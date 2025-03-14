import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BlipForImageTextRetrieval
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time

# コーパス画像のロード
def load_corpus_images(corpus_file_path, data_dir):
    with open(corpus_file_path, 'r') as f:
        corpus_images = json.load(f)
    corpus_image_paths = [os.path.join(data_dir, img) for img in corpus_images]
    return corpus_image_paths

# Datasetクラスの定義
class CorpusDatasetBLIP(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
            return {'image': processed_image, 'id': idx}
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

# DataLoaderの作成
def create_dataloader(dataset, batch_size=8, num_workers=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# コーパスベクトルの作成と保存
def save_corpus_vectors(dataloader, model, device, output_path='corpus_vectors_blip_val2014.pt'):
    corpus_vectors = []
    corpus_ids = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        if batch is None:
            continue

        images = batch['image'].to(device)
        ids = batch['id'].to(device)

        # BLIPのビジョンモデルを使って特徴量を抽出
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=images)
            image_embeds = vision_outputs.last_hidden_state[:, 0, :].to(torch.float32)

            # vision_projを通して正規化
            image_embeds = F.normalize(model.vision_proj(image_embeds), dim=-1)

        # ベクトルとIDを追加
        corpus_vectors.append(image_embeds)
        corpus_ids.append(ids)
        time.sleep(2)

    # 全てのベクトルとIDを連結
    corpus_vectors = torch.cat(corpus_vectors)
    corpus_ids = torch.cat(corpus_ids)

    # IDでソート
    arg_ids = torch.argsort(corpus_ids)
    corpus_vectors = corpus_vectors[arg_ids]
    corpus_ids = corpus_ids[arg_ids]

    # コーパスを保存
    torch.save((corpus_ids, corpus_vectors), output_path)
    print(f"Saved corpus vectors to {output_path}")

# BLIPを使ったコーパス生成
def generate_blip_corpus(corpus_file_path, data_dir, output_dir):
    processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to('cuda' if torch.cuda.is_available() else 'cpu')

    # コーパスをロードしてデータセットを作成
    corpus_image_paths = load_corpus_images(corpus_file_path, data_dir)
    dataset = CorpusDatasetBLIP(corpus_image_paths, processor)

    # DataLoaderの作成
    dataloader = create_dataloader(dataset)
    
    output_path = os.path.join(output_dir, 'corpus_vectors_blip.pt')

    # コーパスベクトルの保存
    save_corpus_vectors(dataloader, model, 'cuda' if torch.cuda.is_available() else 'cpu', output_path)

# 実行例
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    corpus_file_path = 'your_corpus_file_path'
    data_dir = 'your_data_dir'
    output_dir = 'your_output_dir'

    generate_blip_corpus(corpus_file_path, data_dir, output_dir)
