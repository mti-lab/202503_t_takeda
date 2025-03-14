# CCIR: Chat-based Composed Image Retrieval

## 環境構築
```pip install -r requirements.txt```

## データのダウンロード
実験に使用する際のデータセットを以下のサイトからそれぞれダウンロードし、データディレクトリの.gitkeepがある箇所にダウンロードする
[Visual Dialog(Validation set)](https://visualdialog.org/data), [COCO (COCO2017 Val images)](https://cocodataset.org/#home), [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)


## 実験順序

データベースの画像のキャプションを作成する
```make_caption_from_blip2.py```
キャプションを特徴量に変換する
```make_text_features.py```
画像データベースを特徴量に変換する
```prepare_corpus_blip.py or prepare_corpus_clip.py```

そしてCCIRディレクトリの以下のファイルを実行する
```python eval.py```

## 先行研究との比較
先行研究との比較は以下のリポジトリを参考にそれぞれコードを実行する
[ChatIR](https://github.com/levymsn/ChatIR), [PlugIR](https://github.com/Saehyung-Lee/PlugIR), [Pic2Word](https://github.com/google-research/composed_image_retrieval), [MagicLens](https://github.com/google-deepmind/magiclens)