# utils/rotation.py
import torch
import torchvision.transforms.functional as TF

# view order mapping
VIEWS = [0, 90, 180, 270]


def rotate_tensor_224(x: torch.Tensor, degrees: int) -> torch.Tensor:
    """Rotate a CHW tensor (float [0,1]) by degrees in {0,90,180,270}.
    Keeps size 224×224 using torchvision's rotate (nearest/bilinear safe).
    """
    return TF.rotate(x, degrees)


def expand_sst_views(batch_imgs: torch.Tensor, batch_targets: torch.Tensor):
    """Given a mini-batch [B,3,224,224], returns 4B tensors and view_ids.
    Output:
      imgs_all: [4B,3,224,224]
      targets_all: [4B]
      view_ids: [4B] in {0,1,2,3}
    """
    B = batch_imgs.size(0)
    imgs, targets, views = [], [], []
    for v_idx, deg in enumerate(VIEWS):
        rot = torch.stack([rotate_tensor_224(img, deg) for img in batch_imgs])
        imgs.append(rot)
        targets.append(batch_targets.clone())
        views.append(torch.full((B,), v_idx, dtype=torch.long))
    return torch.cat(imgs, dim=0), torch.cat(targets, dim=0), torch.cat(views, dim=0)