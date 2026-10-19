"""Local gradient features; deliberately not called a LESS/GIST reproduction."""

from __future__ import annotations

import hashlib
import math

from .encoding import encode_sample, supervised_tokens
from .counterfactual_probe import virtual_update
from .training_safety import require_finite


def sketch(tensors, dimensions=1024, seed=20261018):
    """Streaming CountSketch; avoids a parameter-count by dimension matrix."""
    import torch

    result = torch.zeros(dimensions, dtype=torch.float32)
    for name, tensor in sorted(tensors.items()):
        generator = torch.Generator().manual_seed(
            seed ^ int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        )
        flat = tensor.detach().float().cpu().flatten()
        for chunk in flat.split(65536):
            indices = torch.randint(dimensions, (len(chunk),), generator=generator)
            signs = torch.randint(2, (len(chunk),), generator=generator) * 2 - 1
            result.scatter_add_(0, indices, chunk * signs)
    return result


def gradient_summary(
    model, tokenizer, samples, max_length=512, dimensions=1024, include_subspace=False
):
    import torch

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    device = next(model.parameters()).device
    category_vectors = {}
    category_counts = {}
    example_vectors = []
    try:
        for _, sample in samples:
            encoded = encode_sample(sample, tokenizer, max_length)
            with torch.enable_grad():
                loss = require_finite(
                    model(
                        **{k: v.unsqueeze(0).to(device) for k, v in encoded.items()}
                    ).loss
                )
                gradients = torch.autograd.grad(
                    loss, [p for _, p in params], allow_unused=True
                )
            tensors = {n: g for (n, _), g in zip(params, gradients) if g is not None}
            for g in tensors.values():
                require_finite(g, "validation gradient")
            vector = sketch(tensors, dimensions)
            if include_subspace:
                example_vectors.append(vector)
            category_vectors[sample.tag] = (
                category_vectors.get(sample.tag, torch.zeros_like(vector)) + vector
            )
            category_counts[sample.tag] = category_counts.get(sample.tag, 0) + 1
    finally:
        for module, mode in modes:
            module.training = mode
    if not category_vectors:
        raise ValueError("Empty gradient view.")
    means = {k: v / category_counts[k] for k, v in category_vectors.items()}
    result = {
        "overall": torch.stack(list(means.values())).mean(0),
        "categories": means,
        "dimensions": dimensions,
    }
    if include_subspace:
        matrix = torch.stack(example_vectors)
        _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        energy = singular.square()
        rank = (
            int(torch.searchsorted(energy.cumsum(0), energy.sum() * 0.95).item()) + 1
            if float(energy.sum())
            else 0
        )
        result["subspace"] = vh[:rank]
    return result


def candidate_features(
    model, tokenizer, sample, summary, max_length=512, progress=0.0, seed=0
):
    import torch

    encoded = encode_sample(sample, tokenizer, max_length)
    update = virtual_update(model, encoded, seed=seed)
    vector = sketch(update["gradients"], summary["dimensions"])
    delta = sketch(update["displacement"], summary["dimensions"])
    norm = float(vector.norm())
    target = summary["overall"]
    cosine = (
        float(torch.nn.functional.cosine_similarity(vector, target, dim=0))
        if norm and float(target.norm())
        else 0.0
    )
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            loss = float(
                require_finite(
                    model(
                        **{k: v.unsqueeze(0).to(device) for k, v in encoded.items()}
                    ).loss
                ).item()
            )
    finally:
        for module, mode in modes:
            module.training = mode
    features = {
        "candidate_nll": loss,
        "target_tokens": float(supervised_tokens(encoded)),
        "sequence_tokens": float(len(encoded["input_ids"])),
        "progress": float(progress),
        "gradient_norm": norm,
        "gradient_cosine": cosine,
        "adam_gain_projection": float(torch.dot(target, delta)),
        "gradient_available": float(norm > 0),
    }
    for category, target in sorted(summary["categories"].items()):
        features["family_" + category] = float(sample.tag == category)
        features["alignment_" + category] = (
            float(torch.nn.functional.cosine_similarity(vector, target, dim=0))
            if norm and float(target.norm())
            else 0.0
        )
    if "subspace" in summary:
        basis = summary["subspace"]
        projected = basis.T @ (basis @ target)
        features["subspace_rank"] = float(len(basis))
        features["subspace_adam_gain"] = float(torch.dot(projected, delta))
        features["subspace_gradient_energy"] = float(
            (basis @ vector).square().sum()
        ) / max(1e-12, float(vector.square().sum()))
    if not all(math.isfinite(x) for x in features.values()):
        raise FloatingPointError("Nonfinite benefit feature.")
    return features
