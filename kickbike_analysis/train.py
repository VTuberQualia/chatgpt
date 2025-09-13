"""乗りこなし判定モデルの学習エントリーポイント。

現在は骨組みのみで、特徴量とラベルから分類器や強化学習モデルを
学習する実装は未着手である。今後、要件定義8章で述べられている
オフラインRLや模倣学習を取り入れた学習手順を実装する予定。
"""

from __future__ import annotations

from pathlib import Path


def main(dataset_path: Path, out_path: Path = Path("model.pkl")) -> None:
    """指定ディレクトリから学習データを読み込みモデルを保存する。

    Parameters
    ----------
    dataset_path: Path
        特徴量とラベルを含むディレクトリ。
    out_path: Path
        学習済みモデルの出力先。
    """
    raise NotImplementedError(
        "学習処理は未実装です。データセットの形式が固まり次第追加予定です。"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m kickbike_analysis.train <dataset_dir>")
    else:
        main(Path(sys.argv[1]))
