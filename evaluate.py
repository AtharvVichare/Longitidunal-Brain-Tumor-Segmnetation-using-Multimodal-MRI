"""
Evaluation entry point.

Usage:
    python evaluate.py --config configs/config.yaml \
                       --checkpoint checkpoints/longidipro_brain.pt
"""
import argparse
import yaml
import torch

from data.dataset import build_loaders
from models.encoder import build_encoder
from models.longidipro import LongiDiProBrain
from evaluation.evaluator import evaluate
from utils.utils import load_checkpoint


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, test_loader = build_loaders(
        cfg["data"]["registry_path"],
        val_frac=cfg["data"]["val_frac"],
        test_frac=cfg["data"]["test_frac"],
        seed=cfg["data"]["seed"],
    )

    encoder = build_encoder(
        feature_size=cfg["model"]["feature_size"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        device=device,
    )
    model = LongiDiProBrain(encoder, cfg).to(device)
    load_checkpoint(model, None, args.checkpoint, device)

    print("\n--- Validation Set ---")
    evaluate(model, val_loader, device)

    print("\n--- Test Set ---")
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args)
