"""Export already verified admissions for local reuse; makes no provider calls."""

import argparse
from collections import Counter
from pathlib import Path

from TEFLD.src.candidate_pool import candidate_id
from TEFLD.src.dataschema import Training_Sample
from TEFLD.src.helper import read_json_file, stable_json_hash, write_json_file
from TEFLD.src.runtime import file_fingerprint
from TEFLD.src.validation import validate_sample


def export_bank(source, destination):
    source, destination = Path(source), Path(destination)
    admissions = read_json_file(source, {})
    if not admissions:
        raise ValueError("No durable verified admissions found.")
    identities = {item.get("teacher_identity") for item in admissions.values()}
    if len(identities) != 1 or None in identities:
        raise ValueError("Export one declared teacher bank at a time.")
    bank, known = [], set()
    for nonce, item in sorted(admissions.items()):
        sample = Training_Sample.model_validate(item["sample"])
        validate_sample(sample).require()
        review = sample.verification
        if review.get("status") != "accepted_probabilistic_review":
            raise ValueError("Unverified candidate cannot enter a verified bank.")
        fields = ("correct", "supported", "format_valid", "unambiguous")
        if any(review.get("result", {}).get(key) is not True for key in fields):
            raise ValueError("Accepted candidate lacks its passing verifier result.")
        identity = candidate_id(sample)
        if identity in known:
            raise ValueError("Duplicate prompt in admissions; resolve before export.")
        known.add(identity)
        bank.append(sample.model_dump(mode="json"))
    manifest = {
        "source_sha256": file_fingerprint(source),
        "teacher_identity": next(iter(identities)),
        "bank_sha256": stable_json_hash({"bank": bank}),
        "candidates": len(bank),
        "categories": dict(Counter(sample["tag"] for sample in bank)),
        "provider_calls": 0,
        "scope": "Verified training candidates only. Protected-split leakage and encoding checks run again before training. No live generation or automatic fallback.",
    }
    for path, value in (
        (destination, bank),
        (destination.with_suffix(".manifest.json"), manifest),
    ):
        old = read_json_file(path, None)
        if old is not None and old != value:
            raise ValueError("Export destination changed; use a fresh bank path.")
        write_json_file(path, value)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admissions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import json

    print(json.dumps(export_bank(args.admissions, args.output), indent=2))


if __name__ == "__main__":
    main()
