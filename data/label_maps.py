# Common label space: 0=background, 1=WT, 2=TC, 3=ET, 4=RC

LABEL_MAPS = {
    "brats2021": {0: 0, 1: 2, 2: 1, 4: 3},
    "lumiere":   {0: 0, 1: 1, 2: 2, 3: 3},
    "ucsf":      {0: 0, 1: 3, 2: 2, 3: 1, 4: 4},
}


def harmonize_labels(mask, dataset_name):
    import numpy as np
    label_map = LABEL_MAPS[dataset_name]
    out = np.zeros_like(mask)
    for src, tgt in label_map.items():
        out[mask == src] = tgt
    return out
