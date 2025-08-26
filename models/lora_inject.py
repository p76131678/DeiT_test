# models/lora_inject.py
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.r = r
        if r > 0:
            self.A = nn.Linear(base.in_features, r, bias=False)
            self.B = nn.Linear(r, base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.B.weight)
            self.scaling = alpha / r
            for p in self.base.parameters():
                p.requires_grad_(False)
        else:
            self.A = None
            self.B = None
            self.scaling = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0:
            y = y + self.B(self.A(x)) * self.scaling
        return y


def inject_lora_into_vit(model: nn.Module, r: int = 4, alpha: float = 1.0):
    """Replace qkv/proj and optionally MLP fc layers with LoRALinear wrappers.
    Call this AFTER creating the timm ViT model.
    """
    for name, module in model.named_modules():
        # timm ViT blocks usually have .attn.qkv and .attn.proj; MLP is .mlp.fc1/.mlp.fc2
        if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, r=r, alpha=alpha)
        if hasattr(module, 'proj') and isinstance(module.proj, nn.Linear):
            module.proj = LoRALinear(module.proj, r=r, alpha=alpha)
        if hasattr(module, 'fc1') and isinstance(module.fc1, nn.Linear):
            module.fc1 = LoRALinear(module.fc1, r=r, alpha=alpha)
        if hasattr(module, 'fc2') and isinstance(module.fc2, nn.Linear):
            module.fc2 = LoRALinear(module.fc2, r=r, alpha=alpha)
    return model