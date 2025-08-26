import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from models.backbone_deit import DeiTSBackbone
from models.heads.sst_heads import SSTHeads
from utils.rotation import expand_sst_views
from losses.passpp_losses import ProtoBank, explicit_protoaug, hardness_aug, kd_l2, total_loss

DEVICE = torch.device('cpu')  # force CPU-only
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 1
ALPHA = 0.5  # weight for old loss
BETA = 0.5   # weight for KD
LAM_HARD = 0.7
NUM_CLASSES = 2  # tiny smoke test with classes {0, 1}


def build_tiny_cifar10(root: str, train: bool):
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    ds = datasets.CIFAR10(root=root, train=train, download=True, transform=tfm)
    # keep only classes 0 and 1, and only ~20 samples each
    idx = [i for i, (_, y) in enumerate(ds) if y in (0,1)]
    idx = idx[:40]  # ~40 images total
    return Subset(ds, idx)


def train_epoch(backbone, heads, proto_bank, prev_backbone=None):
    ds = build_tiny_cifar10('./data', train=True)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    optimizer = optim.AdamW(list(heads.parameters()), lr=3e-4)
    backbone.eval()  # freeze backbone for speed/stability in smoke test
    for p in backbone.parameters():
        p.requires_grad_(False)

    # collect features once to init prototypes
    with torch.no_grad():
        all_feat, all_lab = [], []
        for img, lab in dl:
            img = img.to(DEVICE)
            feat = backbone(img)
            all_feat.append(feat.cpu())
            all_lab.append(lab)
        all_feat = torch.cat(all_feat, dim=0)
        all_lab = torch.cat(all_lab, dim=0)
        proto_bank.update_from_features(all_feat, all_lab)
        proto_bank.estimate_radius(all_feat, all_lab)

    # real training loop (single pass over the same tiny loader)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    for img, lab in tqdm(dl, desc='train'):  # each step uses SST expansion
        img = img.to(DEVICE)
        lab = lab.to(DEVICE)

        # expand to 4 views
        img4, lab4, view4 = expand_sst_views(img, lab)
        img4, lab4, view4 = img4.to(DEVICE), lab4.to(DEVICE), view4.to(DEVICE)

        # forward backbone (no grad for speed)
        with torch.no_grad():
            feat4 = backbone(img4)  # [4B, 384]

        # new-class logits via correct view head
        logits_new = heads.forward_view(feat4, view4)

        # create old-augmented logits (protoAug + hardness) for classes we have prototypes for
        logits_old_aug = None
        if proto_bank.mu:
            aug_feats, aug_labs = [], []
            # use only samples of the current mini-batch (new set) for hardness
            z_new_batch = feat4.detach()
            for c, mu_c in proto_bank.mu.items():
                mu_c = mu_c.to(DEVICE)
                z_tilde = explicit_protoaug(mu_c, proto_bank.radius)
                z_hard = hardness_aug(mu_c, z_new_batch, lam=LAM_HARD)
                aug_feats.extend([z_tilde, z_hard])
                aug_labs.extend([torch.tensor(c, device=DEVICE), torch.tensor(c, device=DEVICE)])
            aug_feats = torch.stack(aug_feats, dim=0)  # [M,384]
            aug_labs = torch.stack(aug_labs, dim=0)    # [M]
            # For simplicity, send old-aug features through view 0 head
            logits_old = heads.heads[0](aug_feats)
            logits_old_aug = (logits_old, aug_labs)

        # KD between current features and previous backbone (if provided)
        L_kd = None
        if prev_backbone is not None:
            with torch.no_grad():
                feat_prev = prev_backbone(img4)
            L_kd = kd_l2(feat4, feat_prev)

        # total loss
        loss = total_loss(logits_new, lab4, logits_old_aug, alpha=ALPHA, L_kd=L_kd, beta=BETA)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print('Training step finished.')


def test_epoch(backbone, heads):
    ds = build_tiny_cifar10('./data', train=False)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    for img, lab in dl:
        img = img.to(DEVICE)
        lab = lab.to(DEVICE)
        # produce 4 rotated features
        feats_by_view = []
        from utils.rotation import VIEWS, rotate_tensor_224
        for deg in VIEWS:
            img_rot = torch.stack([rotate_tensor_224(x, deg) for x in img])
            feat = backbone(img_rot)
            feats_by_view.append(feat)
        logits = heads.predict_ensemble(feats_by_view)
        pred = logits.argmax(dim=1)
        correct += (pred == lab).sum().item()
        total += lab.numel()
    acc = correct / max(total, 1)
    print(f"Tiny test accuracy: {acc:.3f} on {total} samples")


def main():
    torch.set_num_threads(max(1, os.cpu_count() // 2))  # be nice on CPU

    backbone = DeiTSBackbone(pretrained=True).to(DEVICE)
    # prev_backbone as frozen copy for KD (optional in smoke test)
    prev_backbone = None  # or deepcopy(backbone).eval()

    heads = SSTHeads(feat_dim=backbone.feature_dim, num_classes=NUM_CLASSES).to(DEVICE)
    proto_bank = ProtoBank(feat_dim=backbone.feature_dim)

    train_epoch(backbone, heads, proto_bank, prev_backbone=prev_backbone)
    test_epoch(backbone, heads)

if __name__ == '__main__':
    main()