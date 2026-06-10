import torch
import torch.nn.functional as F


def sample_boundary_voxels(features, boundary_mask, seg_mask, n_samples=256):
    """
    features:      (C, D, H, W) — decoder feature map
    boundary_mask: (D, H, W)    — True where boundary voxel
    seg_mask:      (D, H, W)    — sub-region labels (1=WT, 2=TC, 3=ET)

    Returns: feats (N, C), labels (N,)  or  None, None if no boundary voxels
    """
    boundary_idx = boundary_mask.nonzero(as_tuple=False)  # (K, 3)
    if len(boundary_idx) == 0:
        return None, None

    n    = min(n_samples, len(boundary_idx))
    perm = torch.randperm(len(boundary_idx))[:n]
    idx  = boundary_idx[perm]                                          # (n, 3)

    feats  = features[:, idx[:, 0], idx[:, 1], idx[:, 2]].T           # (n, C)
    labels = seg_mask[idx[:, 0], idx[:, 1], idx[:, 2]]                # (n,)
    return feats, labels


def supcon_loss(features, labels, temperature=0.07):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    features: (N, C) — will be L2 normalized internally
    labels:   (N,)   — sub-region class identities
    """
    if features is None or len(features) < 2:
        return torch.tensor(0.0, device=features.device if features is not None else "cpu")

    features = F.normalize(features, dim=-1)
    sim      = torch.matmul(features, features.T) / temperature  # (N, N)

    labels   = labels.unsqueeze(1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0)

    self_mask = torch.eye(len(features), device=features.device).bool()
    sim       = sim.masked_fill(self_mask, -9e15)

    log_prob  = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos     = pos_mask.sum(dim=1).clamp(min=1)
    loss      = -(pos_mask * log_prob).sum(dim=1) / n_pos
    return loss.mean()
