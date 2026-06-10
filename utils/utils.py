import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count > 0 else 0


def save_checkpoint(model, optimizer, epoch, best_dice, path):
    torch.save({
        "epoch":     epoch,
        "best_dice": best_dice,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from epoch {ckpt['epoch']} | Best Dice: {ckpt['best_dice']:.4f}")
    return ckpt["epoch"], ckpt["best_dice"]
