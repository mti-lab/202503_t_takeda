import csv
import os
import sys
import torch
import clip
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import openai
import random
from tqdm import tqdm  # tqdmをインポート
import time
from torch.nn.functional import normalize
from typing import Any, Optional
from transformers import AutoProcessor, BlipForImageTextRetrieval
import ast  # 文字列からtupleを復元するために使用
import time


# settingsモジュールをインポート
import settings

OPEN_API_KEY = settings.OPEN_API_KEY
openai.api_key = OPEN_API_KEY

# モデルとプリプロセス関数のロード
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class BlipForRetrieval(BlipForImageTextRetrieval):
    def get_text_features(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)
        return text_feat

    def get_image_features(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_feat = normalize(self.vision_proj(vision_outputs[0][:, 0, :]), dim=-1)
        return image_feat


class ImageEmbedder:
    def __init__(self, model, preprocessor):
        self.model = model
        self.processor = preprocessor

def CLIP_ZERO_SHOT_BASELINE():
    # Install CLIP library from https://github.com/openai/CLIP
    model, preprocess = clip.load("ViT-B/32", device='cpu')
    # model, preprocess = clip.load("ViT-B/16", device='cpu')
    model = model.to(device)
    image_embedder = ImageEmbedder(lambda img: model.encode_image(img), lambda path: preprocess(Image.open(path)))
    # Note that CLIP supports only 77 tokens!! this is just a baseline.
    dialog_encoder = lambda text: model.encode_text(clip.tokenize(text, truncate=True).to(device))

    return dialog_encoder, image_embedder


def BLIP_ZERO_SHOT_BASELINE():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlipForRetrieval.from_pretrained("Salesforce/blip-itm-large-coco")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")

    model = model.to(device)

    image_embedder = ImageEmbedder(lambda img: model.get_image_features(img),
                                   lambda path: processor(images=Image.open(path), return_tensors='pt'))
    # if blip, Copus.getitem is image = self.preprocessor(self.corpus[i])['pixel_values']
    dialog_encoder = lambda text: model.get_text_features(**processor(text=text,
                                                                      padding=True,
                                                                      truncation=True,
                                                                      return_tensors="pt"))
    return dialog_encoder, image_embedder


def generate_caption_from_image_url(image_id: str, retries: int = 3, delay: int = 5) -> str:
    """
    GPT-4に画像URLを与えてキャプションを生成する関数。
    リトライ機能付きで、エラー発生時に複数回試行する。
    """
    image_url = "http://images.cocodataset.org/" + image_id
    for attempt in range(retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Generate a caption for the provided image. Please provide only the caption, without extra text."},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=50
            )
            # 生成されたキャプションを取得
            generated_caption = response.choices[0].message.content.strip()
            
            # キャプションが不正確または空の場合に対処
            if not generated_caption or generated_caption.startswith("Error") or generated_caption.isnumeric():
                print(f"Invalid caption received for {image_url}: {generated_caption}")
                raise ValueError("Invalid caption")
            
            print(f"Generated caption for {image_url}: {generated_caption}")
            return generated_caption

        except Exception as e:
            print(f"Error generating caption for {image_url} on attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # リトライ間の待機時間
            else:
                print(f"All attempts failed for {image_url}. Providing default caption.")
                return "An image with unrecognizable content"  # デフォルトのキャプション



def generate_modification_text(target_detailed_caption, reference_detailed_caption, retries=5, delay=10):
    """
    GPT-4を使用して修正テキストを生成する関数。
    エラー発生時には再試行し、一定時間待機します。
    """
    prompt = f"""
        Compare the following two image descriptions:
        1. {target_detailed_caption}
        2. {reference_detailed_caption}

        Identify one specific and concrete difference between the two descriptions. 
        Write a modification instruction to adjust the second description to match the unique details of the first description. 
        Use specific elements directly related to the content of the descriptions and avoid phrases like "first description" or "second description."
        Output only the instruction in imperative form.
    """
    
    for attempt in range(retries):
        try:
            # GPT-4のレスポンスを非同期で取得
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                max_tokens=50
            )
            print(f"Generated modification text: {response.choices[0].message.content}")
            return response.choices[0].message.content
        except openai.OpenAIError as e:
            print(f"Error generating modification text: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)  # 指定時間待機して再試行
    print(f"Failed to generate modification text after {retries} attempts.")
    return None


def get_image_features(image_id, corpus_vectors, search_space):
    """
    search_spaceからimage_idを探し、その画像に対応する特徴量を返す
    """
    corpus_ids, corpus_features = corpus_vectors
    index = search_space.index(image_id)
    return corpus_features[index].clone().detach().to(device)

# 画像キャプションの取得関数
def get_caption_from_target_detailed_file(image_id, detailed_target_captions):
    
    if image_id not in detailed_target_captions:
        print(f"Caption not found for image ID: {image_id}")
        return None
    return detailed_target_captions[image_id]

# 選択画像の詳細なキャプションを取得する関数
def get_caption_from_selected_detailed_file(image_id, detailed_selected_captions):

    if image_id not in detailed_selected_captions:
        print(f"Caption not found for image ID: {image_id}")
        return None
    return detailed_selected_captions[image_id]

def safe_json_dump(data, filename, mode='w'):
    # 辞書のキーがタプルの場合は文字列に変換し、それ以外はそのまま使用
    data_with_str_keys = {str(k) if isinstance(k, tuple) else k: v for k, v in data.items()}
    temp_filename = filename + ".temp"
    with open(temp_filename, mode) as f:
        json.dump(data_with_str_keys, f, ensure_ascii=False, indent=4)
    os.replace(temp_filename, filename)  # Atomically replace the file
    

# search_space_fileから画像IDをロード
def load_search_space(search_space_file):
    with open(search_space_file, 'r') as f:
        search_space = json.load(f)
    return search_space

# 画像検索を行う関数
def image_retrieval(text_feature, search_space, save_path='top_10_similar_images.jpg'):
    corpus_ids, corpus_features = corpus_vectors  # corpus_vectorsをインデックスと特徴量に分解

    # Normalize the text feature (query)
    text_feature = torch.nn.functional.normalize(text_feature, dim=-1)

    # Normalize the corpus features (images)
    corpus_features = torch.nn.functional.normalize(corpus_features, dim=-1)

    # Compute cosine similarities between the text feature and corpus features using matrix multiplication
    similarities = (text_feature @ corpus_features.T).squeeze(0).cpu().numpy()

    # Get image IDs from search_space and pair them with their similarity scores
    image_similarities = [(search_space[index], similarities[index]) for index in range(len(corpus_ids))]

    # Sort the images by similarity in descending order
    images = sorted(image_similarities, key=lambda x: x[1], reverse=True)

    return images



# 特定の画像IDのランキング位置を取得
def get_rank(image_id, images):
    for rank, (current_id, _) in enumerate(images, start=0):
        if current_id == image_id:
            return rank
    return -1  # image_idがリストに存在しない場合    

import time

# GPTを使用して修正テキストから新たなクエリを生成
def generate_new_query(ref_img_caption, modification_text, gpt_model="gpt-3.5-turbo", retries=5, delay=10):
    """
    GPTを用いて修正テキストに基づき新たなクエリを生成する関数
    エラー発生時には再試行し、一定時間待機します。
    """
    messages = [
        {"role": "system", "content": "A user is performing image retrieval. The user provides a reference image caption and a modification for the retrieved image to refine the search. Generate a new query reflecting this modification. Only return queries. Additional descriptive text such as ‘new query:’ will not be included."},
        {"role": "user", "content": f"Reference image caption: {ref_img_caption}"},
        {"role": "user", "content": f"Modification: {modification_text}"}
    ]

    for attempt in range(retries):
        try:
            response = openai.chat.completions.create(
                model=gpt_model,
                messages=messages,
                temperature=0.0,
            )
            print(f"Generated new query: {response.choices[0].message.content}")
            return response.choices[0].message.content
        except openai.OpenAIError as e:
            print(f"Error generating new query: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)  # 指定時間待機して再試行
    
    print(f"Failed to generate new query after {retries} attempts.")
    return None


# 視覚的特徴量を考慮してクエリを更新
def update_query_with_visual_features(current_query_features, selected_image_feature, unselected_image_features, new_text_query, alpha=0.3, beta=0.3, gamma=0.4, dialog_encoder=None):

    with torch.no_grad():
        text_features = dialog_encoder(new_text_query)  # Use the dialog_encoder for both CLIP and BLIP

    # 特徴量を正規化
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    current_query_features = torch.nn.functional.normalize(current_query_features, dim=-1)
    selected_image_feature = torch.nn.functional.normalize(selected_image_feature, dim=-1)
    unselected_image_features = torch.nn.functional.normalize(unselected_image_features, dim=-1)

    # 選択された画像の特徴量に正の重みを掛ける
    weighted_selected_feature = alpha * selected_image_feature

    # 未選択の画像の特徴量の平均に負の重みを掛ける
    weighted_unselected_features = -beta * torch.mean(unselected_image_features, dim=0)
    
    weighted_text_features = gamma * text_features

    # クエリの視覚特徴量とテキスト特徴量を加算して更新
    updated_query_features = current_query_features + weighted_text_features + weighted_selected_feature + weighted_unselected_features

    # 更新されたクエリ特徴量も正規化
    updated_query_features = torch.nn.functional.normalize(updated_query_features, dim=-1)

    return updated_query_features



# ユーザフィードバックを基にクエリを更新して画像検索
def image_retrieval_with_feedback(current_query_features, target_image_id, selected_image_id, unselected_image_ids, corpus_vectors, ref_img_caption, modification_text, search_space, new_text_query, dialog_encoder=None):
    # corpus_vectorsをインデックスと特徴量に分解
    corpus_ids, corpus_features = corpus_vectors

    # search_spaceからインデックスに基づいて対応する特徴量を取得
    selected_index = search_space.index(selected_image_id)
    selected_image_feature = corpus_features[selected_index]

    # 未選択の画像の特徴量を取得
    unselected_image_features = torch.stack([
        corpus_features[search_space.index(img_id)].clone().detach().to(device)
        for img_id in unselected_image_ids
    ])

    updated_query_features = update_query_with_visual_features(
        current_query_features,
        selected_image_feature,
        unselected_image_features,
        new_text_query,
        alpha=0.08,
        beta=0.29,
        gamma=0.44,
        dialog_encoder=dialog_encoder    
    )

    updated_images = image_retrieval(updated_query_features, search_space, save_path=f'top_10_similar_images_updated.jpg')

    return updated_query_features, updated_images

def get_or_generate_modification_text(image_id, selected_image_id, modification_texts, ref_caption, modification_text_output_file, target_caption):
    key = (image_id, selected_image_id)

    # 既に存在する場合はそれを返す
    if key in modification_texts:
        return modification_texts[key]["modification_text"]

    # 存在しない場合は新たに生成
    modification_text = generate_modification_text(target_caption, ref_caption)
    modification_text_entry = {
        "modification_text": modification_text,
        "label": 1  # GPT-generated texts
    }

    # 辞書に追加
    modification_texts[key] = modification_text_entry

    # ファイルに保存（キーを文字列に変換して保存）
    # with open(modification_text_output_file, 'w') as f:
    #     json.dump({str(k): v for k, v in modification_texts.items()}, f, ensure_ascii=False, indent=4)
    safe_json_dump(modification_texts, modification_text_output_file)

    return modification_text

# 新しいクエリを取得または生成
def get_or_generate_new_text_query(image_id, selected_image_id, ref_caption, modification_text, new_text_queries, new_text_query_file):
    key = (image_id, selected_image_id)

    # 既に存在する場合はそれを返す
    if key in new_text_queries:
        return new_text_queries[key]["new_text_query"]

    # 存在しない場合は新たに生成
    new_text_query = generate_new_query(ref_caption, modification_text)
    new_text_query_entry = {
        "ref_caption": ref_caption,
        "modification_text": modification_text,
        "new_text_query": new_text_query
    }

    # 辞書に追加
    new_text_queries[key] = new_text_query_entry

    safe_json_dump(new_text_queries, new_text_query_file)

    return new_text_query

# キャプションがすでに存在するか確認し、存在しなければ新たに生成
def get_or_generate_ref_caption(selected_image_id, ref_captions_dict, ref_caption_file):
    if selected_image_id in ref_captions_dict:
        return ref_captions_dict[selected_image_id]  # 既存のキャプションを使用

    # キャプションを生成し、辞書に追加
    ref_caption = generate_caption_from_image_url(selected_image_id)
    ref_captions_dict[selected_image_id] = ref_caption

    safe_json_dump(ref_captions_dict, ref_caption_file)

    return ref_caption


def load_target_caption_features(features_file):
    """
    事前計算されたキャプション特徴量をロードする関数。

    Args:
        features_file (str): キャプション特徴量が保存されたファイルのパス

    Returns:
        dict: {image_id: torch.Tensor} の形式の辞書
    """
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    # 各特徴量をテンソル形式に変換
    caption_features = {image_id: torch.tensor(features, device=device) for image_id, features in features_data.items()}
    print(f"Loaded caption features from {features_file}")
    return caption_features

def load_selected_caption_features(features_file):
    """
    事前計算されたキャプション特徴量をロードする関数。

    Args:
        features_file (str): キャプション特徴量が保存されたファイルのパス

    Returns:
        dict: {image_id: torch.Tensor} の形式の辞書
    """
    with open(features_file, 'r') as f:
        features_data = json.load(f)
    # 各特徴量をテンソル形式に変換
    caption_features = {image_id: torch.tensor(features, device=device) for image_id, features in features_data.items()}
    print(f"Loaded caption features from {features_file}")
    return caption_features

# 候補画像のキャプション
ref_caption_file = 'your_caption_file_path.json'
ref_captions_dict = {}

# 既存の参照キャプションがある場合、それをロードして辞書に追加
if os.path.exists(ref_caption_file):
    with open(ref_caption_file, 'r') as f:
        ref_captions_dict = json.load(f)

# 修正テキストと補助キャプションのファイルパス
modification_text_output_file = 'your_output_modification_text_file_path.json'
new_text_query_output_file = 'your_output_new_text_query_file_path.json'

# modification_textsとnew_text_queriesの辞書を定義
modification_texts = {}
new_text_queries = {}

# 既存のデータがある場合、それをロードして辞書に追加（キーをtupleに変換）
if os.path.exists(modification_text_output_file):
    with open(modification_text_output_file, 'r') as f:
        modification_texts = {ast.literal_eval(k): v for k, v in json.load(f).items()}

if os.path.exists(new_text_query_output_file):
    with open(new_text_query_output_file, 'r') as f:
        new_text_queries = {ast.literal_eval(k): v for k, v in json.load(f).items()}

# キャプション特徴量ファイルのパス ターゲット画像と候補画像のキャプション特徴量 同じになっているが、変更もできる
target_caption_features_file = 'your_target_caption_features_file.json'
selected_caption_features_file = 'your_selected_caption_features_file.json'


# キャプション特徴量をロード
target_caption_features = load_target_caption_features(target_caption_features_file)
selected_caption_features = load_selected_caption_features(selected_caption_features_file)

# ターゲット画像ファイルとキャプションファイル
target_image_file = "your_target_image_file.json" # 例 VisDial_v1.0_queries_val.json
detailed_target_caption_file = 'your_target_caption_file.json'

# corpus_vectorsファイルとsearch_spaceファイル
corpus_vectors_file = "your_corpus_vectors_file.pt"
search_space_file = "your_search_space_file.json"

with open(detailed_target_caption_file, 'r') as f:
    detailed_target_captions = json.load(f)

random.seed(42)

# コーパスベクトルをロード
corpus_vectors = torch.load(corpus_vectors_file, map_location=device)


MODEL_NAME = "your_model_name"
# BLIPを使用するかどうか
use_blip = True
# CSVに結果を書き込む準備
ranks_csv_file = f'target_image_ranks_{MODEL_NAME}.csv'
top_10_csv_file = f'query_top_10_images_{MODEL_NAME}.csv'
# 新たに選択された画像IDと生成されたクエリを保存するCSVファイル
selected_images_csv_file = f'selected_images_and_queries_{MODEL_NAME}.csv'
# ヘッダーの設定
selected_images_headers = ['Query Image', 'Round', 'Selected Image', 'New Query']

# ヘッダーの設定
ranks_headers = ['Target Image ID'] + [f'Rank at Round {i}' for i in range(11)]
top_10_headers = ['Query Image', 'Round'] + [f'Top {i+1} Image' for i in range(10)]

# 実行例
with open(target_image_file, 'r') as f:
    target_data = json.load(f)

# search_spaceをロード
search_space = load_search_space(search_space_file)

txt_processors = None
if use_blip == True:
    dialog_encoder, image_embedder = BLIP_ZERO_SHOT_BASELINE()
else:
    dialog_encoder, image_embedder = CLIP_ZERO_SHOT_BASELINE()

# CSVファイルを作成し、ヘッダーを書き込む
with open(ranks_csv_file, mode='w', newline='') as ranks_file, \
     open(top_10_csv_file, mode='w', newline='') as top_10_file, \
     open(selected_images_csv_file, mode='w', newline='') as selected_images_file:
    ranks_writer = csv.writer(ranks_file)
    top_10_writer = csv.writer(top_10_file)
    selected_images_writer = csv.writer(selected_images_file)

    ranks_writer.writerow(ranks_headers)
    top_10_writer.writerow(top_10_headers)
    selected_images_writer.writerow(selected_images_headers)

    # ターゲット画像ごとに処理を実行
    for target_data_item in tqdm(target_data, desc="Processing target images"):
        image_id = target_data_item['img']
        caption = target_data_item['dialog'][0]
        target_detailed_caption = get_caption_from_target_detailed_file(image_id, detailed_target_captions)

        with torch.no_grad():
            current_query_features = dialog_encoder(caption)


        # 初期の画像検索とランクを取得
        images = image_retrieval(current_query_features, search_space, save_path='top_10_similar_images_0.jpg')
        initial_rank = get_rank(image_id, images)

        # ターゲット画像の視覚的特徴量を取得
        target_image_features = get_image_features(image_id, corpus_vectors, search_space)

        # 各ターゲット画像のランクとトップ10画像を記録するためのリスト
        rank_record = [image_id, initial_rank]
        top_10_record = [image_id, 0] + [img_id for img_id, _ in images[:10]]
        top_10_writer.writerow(top_10_record)
        selected_images_writer.writerow([image_id, 0, image_id, caption])

                        # **検索終了条件**: ターゲット画像が上位10位以内に入った場合に終了
        if initial_rank < 10:
            # 未使用部分を -1 で埋める
            rank_record.extend([-1] * (12 - len(rank_record)))  # 最大ラウンド数は11（0〜10）
            ranks_writer.writerow(rank_record)
            continue # ループを終了
        
        target_caption_feature = target_caption_features[image_id]

        # 修正: 選択された画像の記録用セットを初期化
        selected_images_set = set()

        # 各ラウンドの処理
        for i in tqdm(range(1, 11), desc=f"Processing rounds for {image_id}", leave=False):
            # リクエスト間に待機時間を挿入
            time.sleep(1)  # 1秒待機（必要に応じて調整）

            # Top 10の画像IDを取得（すでに選択された画像を除外）
            top_10_image_ids = [img_id for img_id, _ in images[:10] if img_id not in selected_images_set]
            top_10_caption_features = [selected_caption_features[img_id].squeeze() for img_id in top_10_image_ids]

            # ターゲット特徴量とトップ10キャプション特徴量の類似度を計算
            similarities = (target_caption_feature @ torch.stack(top_10_caption_features).T).cpu().numpy()

            # 類似度が最も高い画像を選択
            selected_index = np.argmax(similarities)
            selected_image_id = top_10_image_ids[selected_index]

            # 選択された画像を記録セットに追加
            selected_images_set.add(selected_image_id)
            
            # 選択された画像のキャプションを取得または生成
            ref_caption = get_or_generate_ref_caption(selected_image_id, ref_captions_dict, ref_caption_file)
            # ref_detailed_caption = get_caption_from_selected_detailed_file(selected_image_id, detailed_selected_captions)
            
            modification_text = get_or_generate_modification_text(image_id, selected_image_id, modification_texts, ref_caption, modification_text_output_file, target_detailed_caption)
                
            # 新しいクエリを生成
            new_text_query = get_or_generate_new_text_query(image_id, selected_image_id, ref_caption, modification_text, new_text_queries, new_text_query_output_file)

            # 選択された画像と生成されたクエリをCSVに記録
            selected_images_writer.writerow([image_id, i, selected_image_id, new_text_query])
            
            # search_spaceを渡してクエリを更新
            current_query_features, images = image_retrieval_with_feedback(
                current_query_features, image_id, selected_image_id, 
                [img_id for img_id, _ in images[:10] if img_id != selected_image_id], 
                corpus_vectors, ref_caption, modification_text, search_space, new_text_query, dialog_encoder=dialog_encoder
            )

            # 更新されたターゲット画像のランクを取得
            updated_rank = get_rank(image_id, images)
            rank_record.append(updated_rank)

            # ラウンドごとのtop10画像IDを保存
            top_10_record = [image_id, i] + [img_id for img_id, _ in images[:10]]

            # top 10の結果をCSVに記録
            top_10_writer.writerow(top_10_record)

                # **検索終了条件**: ターゲット画像が上位10位以内に入った場合に終了
            if updated_rank < 10:
                print(f"Target image {image_id} entered the top 10 at round {i}. Ending search.")
                break  # ループを終了
        
        
        # 未使用部分を -1 で埋める
        if len(rank_record) < 12:  # 最大ラウンド数は11（0〜10）
            rank_record.extend([-1] * (12 - len(rank_record)))
            
        # 各ターゲット画像の全ラウンド結果をCSVに記録
        ranks_writer.writerow(rank_record)

print(f"Rank results saved to {ranks_csv_file}")
print(f"Top 10 images saved to {top_10_csv_file}")
print(f"Selected images and queries saved to {selected_images_csv_file}")

safe_json_dump(ref_captions_dict, ref_caption_file)



safe_json_dump(modification_texts, modification_text_output_file)

safe_json_dump(new_text_queries, new_text_query_output_file)

