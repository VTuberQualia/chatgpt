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
├── infer.py           # 推論ユーティリティ
├── train.py           # 学習スクリプト（骨組みのみ）
└── prepare_dataset.py # 特徴量前計算スクリプト
```

## 使い方

### 推論を試す

```bash
python -m kickbike_analysis.analyze_video <video_path>
```

各人物について `score`, `category`, `advices`, `quality` を含む結果と
判定付き画像が出力されます。簡易ラッパとして `infer.py` も利用できます。

```bash
python -m kickbike_analysis.infer <video_path>
```

### データセット準備

`prepare_dataset.py` で特徴量を前計算して ``.npy`` ファイルに保存できます。

```bash
python -m kickbike_analysis.prepare_dataset <video_dir>
```

### 学習（未実装）

`train.py` は将来の分類器/強化学習モデル学習用の骨組みのみです。
データセット形式が固まり次第、実装を追加する予定です。

## 依存ライブラリ

```
pip install -r requirements.txt
```

YOLOv8 の重みは AGPLv3 ライセンスに基づきます。商用利用時は
Ultralytics 社のライセンス取得または代替モデルの使用をご検討ください。
