"""An opt-in unique-example stopping rule for development comparisons."""

from .candidate_pool import candidate_id


def example_target(protocol, phase, method):
    policy = protocol.get("example_count_control")
    if policy is None:
        return None
    count = policy.get("unique_training_examples")
    if (
        type(count) is not int
        or count < 10
        or phase != "pilot"
        or method
        not in {
            "dolly_baseline",
            "tefld_dataset_loss",
            "tefld_dataset_validation",
            "tefld_validation",
            "tefld_loss",
        }
    ):
        raise ValueError(
            "Unique-example control requires a declared development pilot."
        )
    return count


def cap_new_examples(batch, ledger, target, rng):
    """Keep replay, and randomly choose only the remaining new-example slots.

    Generation still follows the original ten-slot recipe. Extra final-round
    admissions are recorded separately and never counted as trained examples.
    """
    known = {entry["id"] for entry in ledger}
    identities = [candidate_id(sample) for sample in batch]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate training prompts within one batch.")
    remaining = target - len(known)
    if remaining <= 0:
        raise ValueError("The unique-example target has already been reached.")
    new = [i for i, identity in enumerate(identities) if identity not in known]
    if not new:
        raise ValueError("A round must train on at least one new example.")
    chosen = set(rng.sample(new, min(remaining, len(new))))
    keep = [
        i for i, identity in enumerate(identities) if identity in known or i in chosen
    ]
    return [batch[i] for i in keep], [identities[i] for i in new if i not in chosen]
