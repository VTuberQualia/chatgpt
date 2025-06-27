"""Script for unsupervised clustering of kickbike videos."""
from pathlib import Path
import sys

from .cluster import save_model, train_clusters


def main(dataset_path: Path, out_path: Path = Path("clusters.pkl")) -> None:
    model = train_clusters(dataset_path)
    save_model(model, out_path)
    print(f"クラスタリング結果を {out_path} に保存しました")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        user_input = input(
            "動画フォルダのパスを入力してください (空欄で 'data' フォルダを使用します): "
        ).strip()
        user_input = user_input.strip('"')
        path = Path(user_input) if user_input else Path("data")

    if path.suffix.lower() == ".mp4":
        path = path.parent

    main(path)
