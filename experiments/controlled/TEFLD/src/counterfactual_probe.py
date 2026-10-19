"""Disposable next-round AdamW updates with exact live-state restoration."""

from __future__ import annotations

import hashlib
import random

from .token_budget import optimizer_group


def fingerprint_tensors(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode())
        data = tensor.detach().contiguous().cpu()
        digest.update(str((tuple(data.shape), data.dtype)).encode())
        digest.update(
            data.reshape(-1).view(__import__("torch").uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def virtual_update(model, encoded, *, measure=None, lr=0.0002, seed=0):
    import numpy as np
    import torch

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable adapter parameters for probe.")
    weights = {n: p.detach().clone() for n, p in params}
    gradients = {
        n: None if p.grad is None else p.grad.detach().clone() for n, p in params
    }
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    modes = [(m, m.training) for m in model.modules()]
    rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    before_hash = fingerprint_tensors(params)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        before = measure() if measure else None
        model.train()
        optimizer = torch.optim.AdamW(
            [p for _, p in params], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
        )
        optimizer_group(
            model, optimizer, encoded if isinstance(encoded, list) else [encoded]
        )
        raw_gradients = {
            n: p.grad.detach().float().cpu().clone()
            for n, p in params
            if p.grad is not None
        }
        displacement = {n: (weights[n] - p.detach()).float().cpu() for n, p in params}
        after = measure() if measure else None
        return {
            "gain": None if measure is None else before - after,
            "gradients": raw_gradients,
            "displacement": displacement,
            "state_sha256": before_hash,
        }
    finally:
        with torch.no_grad():
            for n, p in params:
                p.copy_(weights[n])
                p.grad = gradients[n]
            for name, value in buffers.items():
                parent, _, leaf = name.rpartition(".")
                owner = model.get_submodule(parent) if parent else model
                if owner._buffers[leaf].shape == value.shape:
                    owner._buffers[leaf].copy_(value)
                else:
                    owner._buffers[leaf] = value
        for module, mode in modes:
            module.training = mode
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        if fingerprint_tensors(params) != before_hash:
            raise RuntimeError("Probe failed to restore the live adapter.")
