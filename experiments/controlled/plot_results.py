"""Rebuild figures from the published numeric snapshot; no models, dataset or API needed."""

import argparse
import json
from pathlib import Path


def plot(snapshot_path, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    methods, order = data["methods"], data["ranking"]
    labels = [methods[a]["label"] for a in order]
    colors = ["#287c69", "#58a08c", "#49699b", "#7894bb", "#aca0be"]

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    values = [methods[a]["macro_nll"] for a in order]
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.2)
    for i, value in enumerate(values):
        ax.text(value + 0.025, i, f"{value:.4f}", va="center")
    ax.set_xlabel("Comparison-validation NLL (lower is better)")
    ax.set_title(
        "100 unique training examples each; same 64 development evaluation rows"
    )
    save(fig, "validation_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for arm in ("sol61_validation", "luna_validation"):
        curve = methods[arm]["training_curve"]
        axes[0].plot(
            [r["unique_examples"] for r in curve],
            [r["curriculum_validation_nll"] for r in curve],
            marker="o",
            label=methods[arm]["label"],
        )
    axes[0].set(
        xlabel="Unique training examples",
        ylabel="Curriculum-validation NLL",
        title="Feedback validation: separate from comparison",
    )
    axes[0].legend(fontsize=8)
    axes[1].axis("off")
    axes[1].text(
        0,
        0.8,
        "Local validation-guided TEFLD:\nimplemented, no measured run yet.\n\nLocal loss-guided TEFLD and Dolly\ndo not score curriculum validation.\n\nThe final holdout remains unscored.",
        fontsize=11,
        va="top",
    )
    save(fig, "curriculum_validation")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    generation = [methods[a]["api_cost_toman"]["generation"] / 1000 for a in order]
    verification = [methods[a]["api_cost_toman"]["verification"] / 1000 for a in order]
    axes[0].barh(labels, generation, label="Generation", color="#49699b")
    axes[0].barh(
        labels, verification, left=generation, label="Verification", color="#dca45b"
    )
    axes[0].invert_yaxis()
    axes[0].set(
        xlabel="Recorded API cost (thousand toman)",
        title="Local compute cost is excluded",
    )
    axes[0].legend(fontsize=8)
    axes[1].barh(
        labels, [methods[a]["answer_token_exposures"] for a in order], color=colors
    )
    axes[1].invert_yaxis()
    axes[1].set(
        xlabel="Supervised answer-token exposures",
        title="Equal examples, unequal training exposure",
    )
    save(fig, "cost_and_training_exposure")

    categories = sorted(methods["dolly_baseline"]["categories"])
    differences = np.array(
        [
            [
                methods[a]["categories"][c] - methods["dolly_baseline"]["categories"][c]
                for c in categories
            ]
            for a in order[1:]
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.imshow(differences, aspect="auto", cmap="RdBu_r", vmin=-0.7, vmax=0.7)
    ax.set_yticks(range(4), labels[1:])
    ax.set_xticks(
        range(8),
        [c.replace("_", " ") for c in categories],
        rotation=35,
        ha="right",
        fontsize=9,
    )
    for i in range(4):
        for j in range(8):
            ax.text(
                j, i, f"{differences[i,j]:+.3f}", va="center", ha="center", fontsize=9
            )
    ax.set_title("Validation loss difference from Dolly; red = higher loss")
    save(fig, "category_differences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.snapshot, args.output)
