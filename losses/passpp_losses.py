# losses/passpp_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class ProtoBank:
    def __init__(self, feat_dim: int):
        self.feat_dim = feat_dim
        self.mu = {}  # class_id -> tensor [D]
        self.radius = 0.1  # default; can be estimated

    @torch.no_grad()
    def update_from_features(self, feats: torch.Tensor, labels: torch.Tensor):
        # feats [N,D], labels [N]
        for c in labels.unique().tolist():
            sel = labels == c
            if sel.any():
                mu_c = feats[sel].mean(dim=0)
                self.mu[c] = mu_c.detach().clone()

    @torch.no_grad()
    def estimate_radius(self, feats: torch.Tensor, labels: torch.Tensor):
        # average intra-class std as a crude radius
        vals = []
        for c in labels.unique().tolist():
            sel = labels == c
            if sel.sum() > 1:
                std_c = feats[sel].std(dim=0).norm().item()
                vals.append(std_c)
        if vals:
            self.radius = float(sum(vals) / len(vals))


def explicit_protoaug(mu: torch.Tensor, r: float) -> torch.Tensor:
    return mu + torch.randn_like(mu) * r


def hardness_aug(mu: torch.Tensor, z_new_batch: torch.Tensor, lam: float = 0.7) -> torch.Tensor:
    # find nearest new feature to mu
    if z_new_batch.numel() == 0:
        return mu
    dists = torch.cdist(mu[None, :], z_new_batch)  # [1, Bn]
    idx = dists.argmin().item()
    z_star = z_new_batch[idx]
    return lam * mu + (1.0 - lam) * z_star


def kd_l2(feat_t: torch.Tensor, feat_prev: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(feat_t, feat_prev)


def total_loss(logits_new: torch.Tensor, targets_new: torch.Tensor,
               logits_old_aug: torch.Tensor | None,
               alpha: float, L_kd: torch.Tensor | None, beta: float) -> torch.Tensor:
    # CE for new (or view-specific) samples
    L_new = F.cross_entropy(logits_new, targets_new)
    L_old = torch.tensor(0.0, device=logits_new.device)
    if logits_old_aug is not None:
        # pseudo old targets are their original class ids
        L_old = F.cross_entropy(logits_old_aug[0], logits_old_aug[1])  # (logits, labels)
    loss = L_new + alpha * L_old
    if L_kd is not None:
        loss = loss + beta * L_kd
    return loss