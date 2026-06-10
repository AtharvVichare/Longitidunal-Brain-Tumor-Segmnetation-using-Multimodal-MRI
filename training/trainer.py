import time
import torch
from utils.utils import AverageMeter, save_checkpoint


class Trainer:
    def __init__(self, model, optimizer, scheduler, loss_fn,
                 train_loader, val_loader, cfg, device):
        self.model        = model
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.loss_fn      = loss_fn
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.history      = {"total": [], "seg": [], "boundary": []}
        self.best_dice    = 0.0

    def train(self):
        max_epochs           = self.cfg["training"]["max_epochs"]
        freeze_epochs        = self.cfg["training"]["freeze_encoder_epochs"]
        save_every           = self.cfg["training"]["save_every"]
        ckpt_dir             = self.cfg["training"]["checkpoint_dir"]

        for epoch in range(max_epochs):
            # Freeze encoder for first N epochs
            for p in self.model.encoder.parameters():
                p.requires_grad = (epoch >= freeze_epochs)

            train_losses = self._train_epoch(epoch)

            for k in ["total", "seg", "boundary"]:
                self.history[k].append(train_losses[k])

            print(
                f"Epoch {epoch}/{max_epochs} | "
                f"Loss={train_losses['total']:.4f} | "
                f"Seg={train_losses['seg']:.4f} | "
                f"Bnd={train_losses['boundary']:.4f}"
            )

            self.scheduler.step()

            if (epoch + 1) % save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch, self.best_dice,
                    f"{ckpt_dir}/longidipro_epoch{epoch}.pt"
                )

    def _train_epoch(self, epoch):
        self.model.train()
        meter = {k: AverageMeter() for k in ["total", "seg", "boundary", "static", "ortho", "reverse"]}
        t0    = time.time()

        for i, batch in enumerate(self.train_loader):
            item = batch[0]
            if item["images"].shape[0] < 2:
                continue  # need at least 2 timepoints for reverse loss

            images     = item["images"].to(self.device)
            masks      = item["masks"].to(self.device)
            boundaries = item["boundaries"].to(self.device)
            days       = item["days"].to(self.device)

            self.optimizer.zero_grad()
            outputs       = self.model(images, days)
            loss, details = self.loss_fn.compute(
                outputs, images, masks, boundaries, days, self.model
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg["training"]["grad_clip"]
            )
            self.optimizer.step()

            for k in meter:
                meter[k].update(details.get(k, 0))

            print(
                f"  Epoch {epoch} [{i+1}/{len(self.train_loader)}] "
                f"loss={details['total']:.4f} seg={details['seg']:.4f} "
                f"bnd={details['boundary']:.4f} | {time.time()-t0:.1f}s",
                end="\r"
            )

        print()
        return {k: meter[k].avg for k in meter}
