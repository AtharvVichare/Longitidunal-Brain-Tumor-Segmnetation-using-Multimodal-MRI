"""
Main training entry point.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --resume checkpoints/longidipro_epoch10.pt
"""
import argparse
import yaml
import torch

from data.dataset import build_loaders
from models.encoder import build_encoder, load_pretrained_weights, download_pretrained_weights
from models.longidipro import LongiDiProBrain
from training.losses import CombinedLoss
from training.trainer import Trainer
from utils.utils import load_checkpoint


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, _ = build_loaders(
        cfg["data"]["registry_path"],
        val_frac=cfg["data"]["val_frac"],
        test_frac=cfg["data"]["test_frac"],
        seed=cfg["data"]["seed"],
    )

    # Encoder
    encoder = build_encoder(
        feature_size=cfg["model"]["feature_size"],
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        device=device,
    )
    weights_path = "checkpoints/swin_unetr_pretrained.pt"
    download_pretrained_weights("checkpoints")
    encoder = load_pretrained_weights(encoder, weights_path, device)

    # Full model
    model = LongiDiProBrain(encoder, cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["max_epochs"]
    )

    # Resume
    if args.resume:
        load_checkpoint(model, optimizer, args.resume, device)

    # Loss
    loss_fn = CombinedLoss(
        lambdas=cfg["training"]["lambdas"],
        boundary_cfg=cfg["training"]["boundary"],
        device=device,
    )

    # Train
    trainer = Trainer(model, optimizer, scheduler, loss_fn,
                      train_loader, val_loader, cfg, device)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
