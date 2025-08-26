# models/backbone_deit.py
import timm
import torch.nn as nn
import torch

class DeiTSBackbone(nn.Module):
    """Simple wrapper for DeiT-S/16 via timm (deit_small_patch16_224).
    Ensures forward_features returns CLS vector [B, D].
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        model_name = 'deit_small_patch16_224'
        # num_classes=0 -> return model without head
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # determine feature dim
        if hasattr(self.net, 'embed_dim'):
            self.feat_dim = getattr(self.net, 'embed_dim')
        elif hasattr(self.net, 'num_features'):
            self.feat_dim = getattr(self.net, 'num_features')
        else:
            self.feat_dim = 384

    @property
    def feature_dim(self) -> int:
        return self.feat_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return CLS features [B, D]. timm's forward_features sometimes returns:
          - [B, D] (already CLS), or
          - [B, L, D] (token sequence) -> we take [:, 0, :].
        """
        out = self.net.forward_features(x)
        if isinstance(out, torch.Tensor):
            if out.ndim == 3:
                # token sequence: take CLS (first token)
                return out[:, 0, :]
            elif out.ndim == 2:
                return out
        # fallback: try to call .mean if weird shape
        return out.reshape(out.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
