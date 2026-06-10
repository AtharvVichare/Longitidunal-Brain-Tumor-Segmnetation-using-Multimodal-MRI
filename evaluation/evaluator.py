import numpy as np
import torch


def dice_score(pred, target):
    inter = (pred * target).sum()
    return (2 * inter / (pred.sum() + target.sum() + 1e-8)).item()


def evaluate(model, loader, device):
    model.eval()
    all_wt, all_tc, all_et = [], [], []

    with torch.no_grad():
        for batch in loader:
            item       = batch[0]
            images     = item["images"].to(device)
            masks      = item["masks"].to(device)
            boundaries = item["boundaries"].to(device)
            days       = item["days"].to(device)
            T          = images.shape[0]

            outputs    = model(images, days)
            seg_logits = outputs["seg_logits"]

            for t in range(T):
                pred = seg_logits[t].argmax(dim=0)
                gt   = masks[t]

                wt_pred = (pred > 0).float();  wt_gt = (gt > 0).float()
                tc_pred = ((pred == 2) | (pred == 3)).float()
                tc_gt   = ((gt   == 2) | (gt   == 3)).float()
                et_pred = (pred == 3).float();  et_gt = (gt == 3).float()

                all_wt.append(dice_score(wt_pred, wt_gt))
                all_tc.append(dice_score(tc_pred, tc_gt))
                all_et.append(dice_score(et_pred, et_gt))

    results = {
        "dice_wt":  np.mean(all_wt),
        "dice_tc":  np.mean(all_tc),
        "dice_et":  np.mean(all_et),
        "dice_avg": np.mean([np.mean(all_wt), np.mean(all_tc), np.mean(all_et)]),
    }

    print(f"Dice  WT: {results['dice_wt']:.4f} | "
          f"TC: {results['dice_tc']:.4f} | "
          f"ET: {results['dice_et']:.4f} | "
          f"Avg: {results['dice_avg']:.4f}")
    return results
