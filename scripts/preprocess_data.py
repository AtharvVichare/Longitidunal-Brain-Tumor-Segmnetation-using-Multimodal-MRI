"""
Run preprocessing for LUMIERE and UCSF datasets.

Usage:
    python scripts/preprocess_data.py \
        --lumiere_dir /path/to/lumiere \
        --ucsf_dir    /path/to/ucsf \
        --out_dir     /path/to/processed
"""
import argparse
import pandas as pd
from pathlib import Path

from data.preprocess import run_preprocessing


def build_registry(processed_dir, out_path):
    import numpy as np
    records = []
    for npz_path in sorted(Path(processed_dir).rglob("*.npz")):
        parts      = npz_path.parts
        dataset    = parts[-3]
        patient_id = parts[-2]
        d          = np.load(npz_path)
        records.append({
            "dataset":    dataset,
            "patient_id": patient_id,
            "timepoint":  npz_path.stem,
            "days":       int(d["days"][0]),
            "has_mask":   bool(d["mask"].max() > 0),
            "path":       str(npz_path),
        })
    registry = pd.DataFrame(records)
    registry.to_csv(out_path, index=False)
    print(registry.groupby("dataset")["patient_id"].nunique())
    print(f"Registry saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lumiere_dir", type=str, default=None)
    parser.add_argument("--ucsf_dir",    type=str, default=None)
    parser.add_argument("--out_dir",     type=str, required=True)
    parser.add_argument("--registry",    type=str, default="registry.csv")
    args = parser.parse_args()

    if args.lumiere_dir:
        run_preprocessing(args.lumiere_dir, "lumiere", args.out_dir)
    if args.ucsf_dir:
        run_preprocessing(args.ucsf_dir, "ucsf", args.out_dir)

    build_registry(args.out_dir, args.registry)
