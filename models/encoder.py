import torch
from monai.networks.nets import SwinUNETR

PRETRAINED_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/"
    "releases/download/0.8.1/model_swinvit.pt"
)


def build_encoder(feature_size=48, in_channels=4, out_channels=3,
                  use_checkpoint=True, device="cuda"):
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=use_checkpoint,
    ).to(device)
    return model


def load_pretrained_weights(model, weights_path, device="cuda"):
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"Pretrained weights loaded. Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    return model


def download_pretrained_weights(save_path):
    import urllib.request
    print(f"Downloading pretrained weights to {save_path} ...")
    urllib.request.urlretrieve(PRETRAINED_URL, save_path)
    print("Done.")
