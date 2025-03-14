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

def CLIP_ZERO_SHOT_BASELINE():
    # Install CLIP library from https://github.com/openai/CLIP
    model, preprocess = clip.load("ViT-B/32", device='cpu')
    # model, preprocess = clip.load("ViT-B/16", device='cpu')
    model = model.to(device)
    image_embedder = ImageEmbedder(lambda img: model.encode_image(img), lambda path: preprocess(Image.open(path)))
    # Note that CLIP supports only 77 tokens!! this is just a baseline.
    dialog_encoder = lambda text: model.encode_text(clip.tokenize(text, truncate=True).to(device))

    return dialog_encoder, image_embedder

def generate_text_features_for_captions(captions, existing_features, dialog_encoder, output_file):
    """
    キャプションリストのテキスト特徴量を生成し、指定したファイルに保存する関数

    Args:
        captions (dict): {image_id: caption} の形式の辞書
        existing_features (dict): 既存の特徴量 {image_id: feature} の形式の辞書
        dialog_encoder (function): テキストをエンコードする関数
        output_file (str): 保存先のファイル名
    """
    text_features = existing_features.copy()  # 既存の特徴量をコピー

    for image_id, caption in tqdm(captions.items(), desc="Generating text features"):
        if image_id in existing_features:
            continue  # 既存の特徴量がある場合スキップ
        with torch.no_grad():
            feature = dialog_encoder(caption)
            feature = torch.nn.functional.normalize(feature, dim=-1)
            text_features[image_id] = feature.cpu().numpy().tolist()  # numpy配列をリストに変換

    # 保存
    with open(output_file, 'w') as f:
        json.dump(text_features, f, ensure_ascii=False, indent=4)
    print(f"Text features saved to {output_file}")


# 実行例
captions_file = 'your_caption_file.json'  # キャプションファイル
output_features_file = 'your_output_caption_features.json'    # 保存先の特徴量ファイル

# 既存の特徴量を読み込む
try:
    with open(output_features_file, 'r') as f:
        existing_features = json.load(f)
except FileNotFoundError:
    existing_features = {}

with open(captions_file, 'r') as f:
    captions = json.load(f)

# モデルとプリプロセス関数のロード
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# BLIPまたはCLIPのエンコーダを初期化
dialog_encoder, _ = BLIP_ZERO_SHOT_BASELINE()

# テキスト特徴量を生成して保存
generate_text_features_for_captions(captions, existing_features, dialog_encoder, output_features_file)
