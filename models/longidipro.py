import torch
import torch.nn as nn
import torch.nn.functional as F

from models.disentanglement import DisentanglementModule
from models.temporal import TimeEmbedding, BidirectionalTemporalTransformer
from models.heads import ClassificationHead, ProgressionHead


class LongiDiProBrain(nn.Module):
    def __init__(self, encoder, cfg):
        super().__init__()
        d = cfg["model"]

        self.encoder     = encoder
        self.disentangle = DisentanglementModule(in_dim=768, out_dim=d["d_model"])
        self.time_embed  = TimeEmbedding(d_model=d["d_model"])
        self.temporal_tf = BidirectionalTemporalTransformer(
            d_model=d["d_model"],
            nhead=d["nhead"],
            num_layers=d["num_transformer_layers"],
            dropout=d["dropout"],
        )
        self.clf_head  = ClassificationHead(in_dim=d["d_model"], num_classes=d["num_classes"])
        self.prog_head = ProgressionHead(in_dim=d["d_model"])

    def forward(self, images, days):
        """
        images: (T, 4, D, H, W)
        days:   (T,)

        Returns dict with all outputs and intermediate features.
        """
        T = images.shape[0]
        seg_logits  = []
        bottlenecks = []

        for t in range(T):
            img_t = images[t].unsqueeze(0)              # (1,4,D,H,W)
            seg_t = self.encoder(img_t)                 # (1,3,D,H,W)
            seg_logits.append(seg_t.squeeze(0))

            # Lightweight bottleneck proxy: flatten spatial dims of seg features
            feat_t = seg_t.flatten(2).permute(0, 2, 1)  # (1, N, 3)
            feat_t = F.pad(feat_t, (0, 768 - feat_t.shape[-1]))  # (1, N, 768)
            bottlenecks.append(feat_t.squeeze(0))       # (N, 768)

        seg_logits  = torch.stack(seg_logits)            # (T, 3, D, H, W)
        bottlenecks = torch.stack(bottlenecks)           # (T, N, 768)

        # Disentangle each timepoint
        statics, dynamics = [], []
        for t in range(T):
            s, d = self.disentangle(bottlenecks[t].unsqueeze(0))
            statics.append(s.squeeze(0))
            dynamics.append(d.squeeze(0))
        statics  = torch.stack(statics)   # (T, N, d_model)
        dynamics = torch.stack(dynamics)  # (T, N, d_model)

        # Add time embeddings + temporal transformer
        dynamics      = self.time_embed(dynamics, days)
        temporal_feat = self.temporal_tf(dynamics)       # (T, N, d_model)

        # Output heads
        clf_out            = self.clf_head(temporal_feat)
        rano_out, ttp_out  = self.prog_head(temporal_feat)

        return {
            "seg_logits":    seg_logits,     # (T, 3, D, H, W)
            "temporal_feat": temporal_feat,  # (T, N, d_model)
            "statics":       statics,        # (T, N, d_model)
            "dynamics":      dynamics,       # (T, N, d_model)
            "clf_out":       clf_out,        # (1, num_classes)
            "rano_out":      rano_out,       # (1, 3)
            "ttp_out":       ttp_out,        # (1, 1)
        }
