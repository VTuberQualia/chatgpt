"""乗りこなし判定モデルの学習エントリーポイント。

簡易的な実装として、特徴量とラベルが保存された ``CSV`` ファイルを
読み込みロジスティック回帰分類器を学習し ``joblib`` 形式で保存する。
将来的には要件定義8章で述べられているようなオフラインRLや模倣学習を
取り入れた学習手順へ置き換える予定である。
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _resolve_dataset(path: Path) -> Path:
    """引数がディレクトリの場合は ``data.csv`` を探し、
    ファイルの場合はそのまま返す。"""

    path = Path(path)
    if path.is_dir():
        csv_file = path / "data.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"データセット {csv_file} が見つかりません")
        return csv_file
    if not path.exists():
        raise FileNotFoundError(f"データセット {path} が見つかりません")
    return path


def main(dataset_path: Path, out_path: Path = Path("model.pkl")) -> None:
    """CSV 形式の特徴量とラベルからモデルを学習し保存する。

    Parameters
    ----------
    dataset_path: Path
        ``f1``〜``f6`` と ``label`` 列を含む ``CSV`` ファイル、
        もしくはそれを格納したディレクトリ。
    out_path: Path
        学習済みモデルの出力先。
    """

    csv_file = _resolve_dataset(dataset_path)

    df = pd.read_csv(csv_file)
    X = df.drop(columns=[df.columns[-1]]).values
    y = df[df.columns[-1]].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)

    joblib.dump(pipe, out_path)
    print(f"モデルを保存しました: {out_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m kickbike_analysis.train <dataset.csv> [out_model.pkl]")
    elif len(sys.argv) == 2:
        main(Path(sys.argv[1]))
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]))
