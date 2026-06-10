import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class LongiGliomaDataset(Dataset):
    def __init__(self, registry_path, split="train", val_frac=0.15, test_frac=0.15, seed=42):
        df           = pd.read_csv(registry_path)
        all_patients = df["patient_id"].unique()

        rng = np.random.default_rng(seed)
        rng.shuffle(all_patients)

        n      = len(all_patients)
        n_test = int(n * test_frac)
        n_val  = int(n * val_frac)

        if   split == "test": patients = all_patients[:n_test]
        elif split == "val":  patients = all_patients[n_test:n_test + n_val]
        else:                 patients = all_patients[n_test + n_val:]

        self.df       = df[df["patient_id"].isin(patients)]
        self.patients = list(self.df["patient_id"].unique())

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid        = self.patients[idx]
        timepoints = self.df[self.df["patient_id"] == pid].sort_values("days")

        images, masks, boundaries, days = [], [], [], []
        for _, row in timepoints.iterrows():
            d = np.load(row["path"])
            images.append(d["image"])
            masks.append(d["mask"])
            boundaries.append(d["boundary"])
            days.append(row["days"])

        return {
            "patient_id": pid,
            "images":     torch.tensor(np.stack(images),     dtype=torch.float32),  # (T,4,D,H,W)
            "masks":      torch.tensor(np.stack(masks),      dtype=torch.long),      # (T,D,H,W)
            "boundaries": torch.tensor(np.stack(boundaries), dtype=torch.uint8),     # (T,D,H,W)
            "days":       torch.tensor(days,                 dtype=torch.float32),   # (T,)
        }


def collate_fn(batch):
    # T varies per patient — return list of dicts, not stacked tensors
    return batch


def build_loaders(registry_path, val_frac=0.15, test_frac=0.15, seed=42, num_workers=2):
    train_ds = LongiGliomaDataset(registry_path, "train", val_frac, test_frac, seed)
    val_ds   = LongiGliomaDataset(registry_path, "val",   val_frac, test_frac, seed)
    test_ds  = LongiGliomaDataset(registry_path, "test",  val_frac, test_frac, seed)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} patients")
    return train_loader, val_loader, test_loader
