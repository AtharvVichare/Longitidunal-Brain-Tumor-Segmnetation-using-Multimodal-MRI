import torch.nn as nn


class ClassificationHead(nn.Module):
    """Predicts tumor grade (LGG / GBM) from temporal features."""

    def __init__(self, in_dim=384, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, temporal_feat):
        # temporal_feat: (T, N, d_model) — use last timepoint
        x = temporal_feat[-1]      # (N, d_model)
        x = x.T.unsqueeze(0)       # (1, d_model, N)
        return self.net(x)         # (1, num_classes)


class ProgressionHead(nn.Module):
    """Predicts RANO status (3-class) and Time-to-Progression (regression)."""

    def __init__(self, in_dim=384):
        super().__init__()
        self.shared = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.rano_head = nn.Linear(128, 3)  # stable / response / progression
        self.ttp_head  = nn.Linear(128, 1)  # time-to-progression

    def forward(self, temporal_feat):
        # Use mean over all timepoints
        x = temporal_feat.mean(dim=0).T.unsqueeze(0)  # (1, d_model, N)
        x = self.shared(x)
        return self.rano_head(x), self.ttp_head(x)    # (1,3), (1,1)
