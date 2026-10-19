"""Finite loss/gradient guards shared by standalone and comparison trainers."""

from __future__ import annotations


def require_finite(tensor, name="loss"):
    import torch

    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Nonfinite {name}; optimizer update aborted.")
    return tensor


def install_gradient_guards(model):
    def guard(gradient):
        return require_finite(gradient, "gradient")

    return [p.register_hook(guard) for p in model.parameters() if p.requires_grad]


def guarded_trainer_class(base):
    class FiniteTrainer(base):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            if not (inputs["labels"][..., 1:] != -100).any():
                raise ValueError("Training batch has no supervised prediction targets.")
            result = super().compute_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )
            require_finite(result[0] if return_outputs else result)
            return result

    return FiniteTrainer
