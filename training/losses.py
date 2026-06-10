import torch
import torch.nn as nn
from monai.losses import DiceLoss

from models.disentanglement import static_consistency_loss, orthogonality_loss, reverse_loss
from models.boundary import sample_boundary_voxels, supcon_loss


class CombinedLoss:
    def __init__(self, lambdas, boundary_cfg, device):
        self.lambdas      = lambdas
        self.boundary_cfg = boundary_cfg
        self.device       = device

        self.seg_loss_fn = DiceLoss(to_onehot_y=True, softmax=True, num_classes=4)
        self.ce_loss_fn  = nn.CrossEntropyLoss()
        self.mse_loss_fn = nn.MSELoss()

    def compute(self, outputs, images, masks, boundaries, days,
                model, grade_labels=None, rano_labels=None, ttp_labels=None):

        seg_logits    = outputs["seg_logits"]
        statics       = outputs["statics"]
        dynamics      = outputs["dynamics"]
        clf_out       = outputs["clf_out"]
        rano_out      = outputs["rano_out"]
        ttp_out       = outputs["ttp_out"]
        T             = seg_logits.shape[0]

        # Segmentation
        l_seg = sum(
            self.seg_loss_fn(
                seg_logits[t].unsqueeze(0),
                masks[t].unsqueeze(0).unsqueeze(0).float()
            )
            for t in range(T)
        ) / T

        # Static consistency
        l_static = sum(
            static_consistency_loss(statics[t], statics[t + 1])
            for t in range(T - 1)
        ) / max(T - 1, 1)

        # Orthogonality
        l_ortho = sum(
            orthogonality_loss(statics[t].unsqueeze(0), dynamics[t].unsqueeze(0))
            for t in range(T)
        ) / T

        # Reverse trick — run reversed pair through model
        l_reverse = torch.tensor(0.0, device=self.device)
        if T >= 2:
            pair_rev  = torch.flip(images[:2], dims=[0])
            days_rev  = torch.flip(days[:2],   dims=[0])
            with torch.no_grad():
                rev_out = model(pair_rev, days_rev)
            l_reverse = reverse_loss(
                dynamics[0].unsqueeze(0), rev_out["dynamics"][0].unsqueeze(0),
                statics[0].unsqueeze(0),  rev_out["statics"][0].unsqueeze(0),
            )

        # Boundary SupCon
        l_boundary = torch.tensor(0.0, device=self.device)
        n_samples  = self.boundary_cfg["n_samples"]
        temperature = self.boundary_cfg["temperature"]
        for t in range(T):
            feats, labels = sample_boundary_voxels(
                seg_logits[t].detach(), boundaries[t].bool(), masks[t], n_samples
            )
            if feats is not None:
                l_boundary = l_boundary + supcon_loss(feats, labels, temperature)
        l_boundary = l_boundary / T

        # Classification (masked if no labels)
        l_clf = torch.tensor(0.0, device=self.device)
        if grade_labels is not None:
            l_clf = self.ce_loss_fn(clf_out, grade_labels)

        # Progression (masked if no labels)
        l_prog = torch.tensor(0.0, device=self.device)
        if rano_labels is not None:
            l_prog = l_prog + self.ce_loss_fn(rano_out, rano_labels)
        if ttp_labels is not None:
            l_prog = l_prog + self.mse_loss_fn(ttp_out.squeeze(), ttp_labels.float())

        lm = self.lambdas
        total = (lm["seg"]      * l_seg      +
                 lm["boundary"] * l_boundary +
                 lm["static"]   * l_static   +
                 lm["ortho"]    * l_ortho    +
                 lm["reverse"]  * l_reverse  +
                 lm["clf"]      * l_clf      +
                 lm["prog"]     * l_prog)

        return total, {
            "total":    total.item(),
            "seg":      l_seg.item(),
            "boundary": l_boundary.item(),
            "static":   l_static.item(),
            "ortho":    l_ortho.item(),
            "reverse":  l_reverse.item(),
        }
