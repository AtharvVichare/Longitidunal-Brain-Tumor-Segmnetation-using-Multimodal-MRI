"""
Download dataset helpers.

Usage:
    python scripts/download_data.py --dataset lumiere --out_dir /content/lumiere
    python scripts/download_data.py --dataset pretrained_weights --out_dir ./checkpoints
"""
import argparse
import urllib.request
from pathlib import Path

LUMIERE_URL    = "https://ndownloader.figshare.com/files/38249697"
PRETRAINED_URL = ("https://github.com/Project-MONAI/MONAI-extra-test-data/"
                  "releases/download/0.8.1/model_swinvit.pt")


def download_lumiere(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir.parent / "lumiere.zip"

    print("Downloading LUMIERE ...")
    urllib.request.urlretrieve(LUMIERE_URL, zip_path)

    import zipfile
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    zip_path.unlink()
    print(f"LUMIERE ready at {out_dir}")


def download_pretrained_weights(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "swin_unetr_pretrained.pt"
    print("Downloading pretrained Swin UNETR weights ...")
    urllib.request.urlretrieve(PRETRAINED_URL, save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["lumiere", "pretrained_weights"], required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    if args.dataset == "lumiere":
        download_lumiere(args.out_dir)
    elif args.dataset == "pretrained_weights":
        download_pretrained_weights(args.out_dir)
