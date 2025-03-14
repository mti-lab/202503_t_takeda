import os
import json
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# モデルとプロセッサのロード
model_id = "Salesforce/blip2-flan-t5-xl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

processor2 = Blip2Processor.from_pretrained(model_id)
blip2 = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

# JSONファイルと画像フォルダのパス
input_json = "your_data_search_space.json"  # データベースのJSONファイル 例: VisDial_v1.0_queries_val.json or flickr30k_search_space.json　など
output_file = "your_output_file.json"  # キャプションを保存するJSONファイル

# キャプションを保存する辞書を初期化
captions_dict = {}

# 以前の進捗がある場合、途中までのデータをロード
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        captions_dict = json.load(f)

# JSONファイルを読み込む
with open(input_json, "r") as f:
    image_list = json.load(f)

# バッチサイズの設定
batch_size = 8

# tqdmを使って進行状況を表示
batched_data = [image_list[i:i + batch_size] for i in range(0, len(image_list), batch_size)]

for batch_entries in tqdm(batched_data, desc="Generating captions"):
    image_batch = []
    valid_keys = []

    for img_key in batch_entries:
        image_path = os.path.join(".", img_key)  # 画像フォルダはカレントディレクトリ

        # 既にキャプションが生成されている場合はスキップ
        if img_key in captions_dict:
            continue

        # 画像を開く
        try:
            image = Image.open(image_path).convert("RGB")
            image_batch.append(image)
            valid_keys.append(img_key)
        except (UnidentifiedImageError, FileNotFoundError):
            print(f"Skipping unsupported or missing image: {img_key}")
            continue

    # バッチが空であれば次に進む
    if not image_batch:
        continue

    # プロンプトの準備
    prompt = ["A photo of "] * len(image_batch)

    # 画像とテキストをエンコード
    inputs = processor2(images=image_batch, text=prompt, return_tensors="pt", padding=True).to(device)

    # モデルで生成
    outputs = blip2.generate(
        **inputs, 
        min_length=30,
        max_length=60,
        num_beams=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=2.0
    )

    # 生成されたキャプションを辞書に追加
    for key, output in zip(valid_keys, outputs):
        captions_dict[key] = processor2.decode(output, skip_special_tokens=True)

    # バッチごとに途中経過を保存
    temp_file = output_file + ".tmp"
    try:
        with open(temp_file, "w") as f:
            json.dump(captions_dict, f, indent=4)
        os.replace(temp_file, output_file)  # 安全に保存
    except Exception as e:
        print(f"Error saving intermediate file: {e}")

# 最終的な結果を保存
try:
    with open(output_file, "w") as f:
        json.dump(captions_dict, f, indent=4)
    print("キャプションの生成が完了しました。")
except Exception as e:
    print(f"Error saving final file: {e}")
