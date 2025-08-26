# models/heads/sst_heads.py
import torch
import torch.nn as nn

class SSTHeads(nn.Module):
    """Four heads (k-class each) for PASS++ multi-view (0,90,180,270)."""
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.num_views = 4
        self.num_classes = num_classes
        self.heads = nn.ModuleList([nn.Linear(feat_dim, num_classes) for _ in range(self.num_views)])

    def forward_all(self, feat: torch.Tensor) -> torch.Tensor:
        """Return logits for all 4 heads stacked as [B, 4, K]."""
        logits = [h(feat) for h in self.heads]
        return torch.stack(logits, dim=1)

    def forward_view(self, feat: torch.Tensor, view_id: torch.Tensor) -> torch.Tensor:
        """Index into the correct head for each sample by view_id (0..3).
        view_id: LongTensor [B]
        returns logits [B, K]
        """
        out = torch.empty(feat.size(0), self.num_classes, device=feat.device)
        for v in range(4):
            mask = (view_id == v)
            if mask.any():
                out[mask] = self.heads[v](feat[mask])
        return out

    @torch.no_grad()
    def predict_ensemble(self, features_by_view: list) -> torch.Tensor:
        """features_by_view: list of 4 tensors [B, D] for views (0,90,180,270).
        Returns logits averaged over views on K classes: [B, K]."""
        assert len(features_by_view) == 4
        logits_sum = None
        for v, f in enumerate(features_by_view):
            logits_v = self.heads[v](f)
            logits_sum = logits_v if logits_sum is None else logits_sum + logits_v
        return logits_sum / 4.0