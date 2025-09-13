# キックバイク乗りこなし判定AI

このリポジトリは、子どもがキックバイクに乗っている映像を解析し、
ペダル付き自転車へ移行できるタイミングを推定するプロトタイプです。
詳細な要件定義は [docs/requirements.md](docs/requirements.md) を参照してください。

## 現在の機能概要

- 動画からフレームを抽出し、YOLOv8 + ByteTrack で子どもとバイクを検出・追跡。
- 骨格推定から傾き・姿勢安定度・視線・蹴りリズム等の特徴量を計算。
- 特徴量をもとに乗りこなしスコア（0–100）とカテゴリ（Beginner/Almost/Ready）を推定。
- 品質チェック（被写体の大きさ、ブレ、暗さ、はみ出し）を行い、問題があれば指摘。
- 判定結果を画像に描画し、アドバイス（3件以上）を提示。

## プログラム構成

```
kickbike_analysis/
├── __init__.py
├── analyze_video.py   # 動画解析のエントリーポイント
├── data_loader.py     # 動画読み込みとフレーム抽出
├── feature_extractor.py# YOLOv8 + ByteTrack による特徴量計算
├── quality_checker.py # 画質・構図の簡易チェック
├── scoring.py         # 特徴量からスコアとアドバイスを算出
├── segmentation.py    # 走行シーンの簡易セグメント化 (将来拡張用)
├── infer.py           # 学習済みモデルを用いた推論
├── train.py           # ロジスティック回帰による学習スクリプト
├── prepare_dataset.py # 特徴量前計算スクリプト
└── extract_images.py  # 学習用の切り出し画像作成
```

## 使い方

### 0. 学習用画像を切り出す（任意）

教師データを準備するため、動画から人物+バイクの領域を切り出して画像として保存できます。

```bash
python -m kickbike_analysis.extract_images <video_dir> <out_dir> --limit 1000
```

`<out_dir>` には最大 `--limit` 枚の画像が保存されます。

### 1. 画像認識（特徴量抽出）

`prepare_dataset.py` で動画から人物ごとの特徴量を計算し ``CSV`` や ``npy`` に保存できます。

```bash
python -m kickbike_analysis.prepare_dataset <video_dir>
```

### 2. 学習

`train.py` に特徴量とラベルを格納した ``CSV`` を渡すとロジスティック回帰モデルを学習し ``model.pkl`` を出力します。

```bash
python -m kickbike_analysis.train <dataset.csv> [out_model.pkl]
```

### 3. 推論を試す

学習済みモデルと解析したい動画を指定して推論を実行します。

```bash
python -m kickbike_analysis.infer <video_path> <model.pkl>
```

各人物について `score`, `category`, `advices`, `quality` を含む結果と判定付き画像が出力されます。

## 依存ライブラリ

```
pip install -r requirements.txt
```

YOLOv8 の重みは AGPLv3 ライセンスに基づきます。商用利用時は
Ultralytics 社のライセンス取得または代替モデルの使用をご検討ください。
