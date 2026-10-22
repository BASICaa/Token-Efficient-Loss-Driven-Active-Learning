"""Exact shifted-target exposure accounting and token-normalized updates."""

from __future__ import annotations

from dataclasses import dataclass

from .encoding import supervised_tokens
from .training_safety import require_finite


def limit_targets(encoded: dict, limit: int) -> dict:
    if limit < 1:
        raise ValueError("Every executed example needs positive target supervision.")
    result = {k: v.clone() for k, v in encoded.items()}
    positions = (result["labels"][1:] != -100).nonzero().flatten() + 1
    result["labels"][positions[limit:]] = -100
    if supervised_tokens(result) < 1:
        raise ValueError("Example has no supervised targets.")
    return result


@dataclass
class TokenBudget:
    total: int
    consumed: int = 0

    def __post_init__(self):
        if self.total < 1 or not 0 <= self.consumed <= self.total:
            raise ValueError("Invalid supervised-token budget.")

    @property
    def remaining(self):
        return self.total - self.consumed

    def admit(self, encoded):
        if not self.remaining:
            return None
        return limit_targets(encoded, min(supervised_tokens(encoded), self.remaining))

    def commit(self, count):
        if count < 1 or count > self.remaining:
            raise ValueError("Invalid budget commit.")
        self.consumed += count


def optimizer_group(model, optimizer, group, *, max_grad_norm=1.0):
    """One accumulation group equals a combined target-weighted batch."""
    import torch

    if not group:
        raise ValueError("Empty optimizer group.")
    total = sum(supervised_tokens(x) for x in group)
    if total == 0 or any(supervised_tokens(x) == 0 for x in group):
        raise ValueError("All executed microbatches require positive supervision.")
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    losses = []
    try:
        for encoded in group:
            output = model(**{k: v.unsqueeze(0).to(device) for k, v in encoded.items()})
            loss = require_finite(output.loss)
            losses.append(float(loss.detach().item()))
            (loss * (supervised_tokens(encoded) / total)).backward()
        params = [p for p in model.parameters() if p.requires_grad]
        for p in params:
            if p.grad is not None:
                require_finite(p.grad, "gradient")
        norm = torch.nn.utils.clip_grad_norm_(
            params, max_grad_norm, error_if_nonfinite=True
        )
        require_finite(norm, "gradient norm")
        optimizer.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    return total, losses
