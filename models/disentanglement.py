import torch
import torch.nn as nn
import torch.nn.functional as F


class DisentanglementModule(nn.Module):
    def __init__(self, in_dim=768, out_dim=384):
        super().__init__()
        self.static_head = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.dynamic_head = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, feat):
        # feat: (B, N, in_dim)
        return self.static_head(feat), self.dynamic_head(feat)


# ---------- Losses ----------

def static_consistency_loss(s_t, s_t1):
    """Static features must be identical across timepoints."""
    return F.mse_loss(s_t, s_t1)


def orthogonality_loss(static, dynamic):
    """Static and dynamic features must be uncorrelated."""
    B, N, C = static.shape
    s = F.normalize(static.reshape(B * N, C),  dim=-1)
    d = F.normalize(dynamic.reshape(B * N, C), dim=-1)
    return (s * d).sum(dim=-1).pow(2).mean()


def reverse_loss(d_fwd, d_rev, s_fwd, s_rev):
    """
    Progression-aware reverse trick:
    - Dynamic features must flip when input order is reversed.
    - Static features must remain unchanged.
    """
    return F.mse_loss(d_fwd, -d_rev) + F.mse_loss(s_fwd, s_rev)
