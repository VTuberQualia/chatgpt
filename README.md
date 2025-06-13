
# キックバイク乗りこなし判定AI

このリポジトリは、子どもがキックバイクに乗っている映像を解析し、ペダル付き自転車に乗れるようになる時期を推定するプロトタイプです。

## 要件定義

1. **入力データ**: 子どもとキックバイクが写っている動画。角度や大きさは問いません。
2. **出力**: ペダル付き自転車に乗れるかどうかの二択（OK/NG）。
3. **機能要件**
   - 動画からフレームを抽出し、子どもとキックバイクを検出する。
   - バランス、ハンドル操作、蹴り出しの強さなど動きの特徴量を算出する。
   - 準備が整っているかどうかラベル付けされたデータを使い、特徴量と準備度の相関を学習する。
   - 新しい動画を入力すると準備度を推定する推論用スクリプトを提供する。
4. **非機能要件**
   - Python 3 を使用し、映像処理には OpenCV、学習には PyTorch など一般的なパッケージを利用する。
   - 家庭のスマートフォンで撮影された角度や画角がまちまちな動画を想定する。
   - データセットを追加してもコード変更なしで学習と推論を切り替えられるようにする。

## プログラム構成
=======
# Kickbike Readiness Analyzer

This repository provides a prototype for analyzing videos of children riding kickbikes to estimate when they are ready to transition to pedal bicycles.

## Requirements Definition

1. **Input Data**: Video footage of children riding kickbikes. Footage should capture the whole body from a side view whenever possible.
2. **Output**: An estimated readiness score or predicted timeframe for successfully riding a pedal bike.
3. **Functional Requirements**
   - Extract frames from input video and detect the child and kickbike.
   - Track body posture and motion features such as balance, steering control, and push force.
   - Train a machine learning model on labeled examples (ready vs not-ready) to learn correlations between motion features and readiness.
   - Provide an inference script that takes a new video and outputs a readiness prediction.
4. **Non‑Functional Requirements**
   - The system should run on Python 3 and rely on widely available packages (OpenCV for video processing, PyTorch for ML training).
   - Training and inference should be separated so new datasets can be added without code changes.

## Program Structure
