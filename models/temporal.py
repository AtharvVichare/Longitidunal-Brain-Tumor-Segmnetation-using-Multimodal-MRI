import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, d_model=384):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, dynamic_feat, days):
        # dynamic_feat: (T, N, d_model) | days: (T,)
        T, N, d = dynamic_feat.shape
        emb = torch.zeros(T, d, device=dynamic_feat.device)
        for i in range(0, d, 2):
            denom = 10000 ** (i / d)
            emb[:, i] = torch.sin(days / denom)
            if i + 1 < d:
                emb[:, i + 1] = torch.cos(days / denom)
        emb = emb.unsqueeze(1).expand(T, N, d)  # (T, N, d)
        return dynamic_feat + self.proj(emb)


class BidirectionalTemporalTransformer(nn.Module):
    def __init__(self, d_model=384, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.forward_tf = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.reverse_tf = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj       = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # x: (T, N, d_model)
        T, N, d = x.shape
        x_flat  = x.reshape(1, T * N, d)

        fwd = self.forward_tf(x_flat)
        rev = self.reverse_tf(torch.flip(x_flat, dims=[1]))
        rev = torch.flip(rev, dims=[1])

        combined = torch.cat([fwd, rev], dim=-1)  # (1, T*N, 2d)
        out      = self.proj(combined)             # (1, T*N, d)
        return out.reshape(T, N, d)
