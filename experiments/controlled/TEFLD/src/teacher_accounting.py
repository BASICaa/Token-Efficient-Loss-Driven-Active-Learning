"""Shared study caps and attributed API usage; estimates never imply a wallet balance."""

from collections import defaultdict
import math
from pathlib import Path

from .helper import read_json_file


def study_spend(root):
    totals = defaultdict(float)
    for path in Path(root).rglob("usage.json"):
        ledger = read_json_file(path, {})
        if not isinstance(ledger, dict) or not isinstance(ledger.get("requests"), list):
            continue
        for event in ledger["requests"]:
            for currency in ("toman", "usd"):
                totals[currency] += event.get(
                    "cost_" + currency, event.get("reserved_cost_" + currency, 0)
                )
    return dict(totals)


def require_study_budget(root, currency, reserve):
    budget = read_json_file(Path(root) / "study_budget.json", {})
    limit = budget.get("max_spend")
    if budget.get("approved") is not True or budget.get("currency") != currency:
        raise PermissionError(
            "This new teacher study requires an explicitly approved shared study_budget.json before paid calls."
        )
    if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("Study budget must be finite and positive.")
    spent = study_spend(root).get(currency, 0)
    if spent + reserve > limit:
        raise RuntimeError("Shared teacher-study spend ceiling reached.")
    return limit


def summarize_teacher_costs(root):
    root = Path(root)
    groups = {}
    requests = 0
    known = defaultdict(float)
    reserved = defaultdict(float)
    for path in sorted(root.rglob("usage.json")):
        ledger = read_json_file(path, {})
        if not isinstance(ledger, dict) or not isinstance(ledger.get("requests"), list):
            continue
        for event in ledger["requests"]:
            owner = event.get("owner", {"phase": "unattributed_historical"})
            key = tuple(
                str(owner.get(k, "unattributed"))
                for k in ("arm", "phase", "method", "seed", "round")
            ) + (event.get("model", "unknown"), event.get("kind", "unknown"))
            label = " | ".join(key)
            group = groups.setdefault(
                label,
                {
                    "owner": owner,
                    "model": event.get("model"),
                    "kind": event.get("kind"),
                    "requests": 0,
                    "completed": 0,
                    "failed_or_pending": 0,
                    "known_toman": 0.0,
                    "reserved_toman": 0.0,
                    "known_usd": 0.0,
                    "reserved_usd": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "request_seconds": 0.0,
                },
            )
            requests += 1
            group["requests"] += 1
            group[
                (
                    "completed"
                    if event.get("status") == "completed"
                    else "failed_or_pending"
                )
            ] += 1
            usage = event.get("usage") or {}
            group["input_tokens"] += usage.get("prompt_tokens", 0)
            group["output_tokens"] += usage.get("completion_tokens", 0)
            group["request_seconds"] += event.get("seconds", 0)
            for currency in ("toman", "usd"):
                if "cost_" + currency in event:
                    amount = event["cost_" + currency]
                    known[currency] += amount
                    group["known_" + currency] += amount
                else:
                    amount = event.get("reserved_cost_" + currency, 0)
                    reserved[currency] += amount
                    group["reserved_" + currency] += amount
    accesses = []
    for path in sorted(root.rglob("accesses.json")):
        for event in read_json_file(path, []):
            accesses.append(event)
    outcomes = []
    for path in sorted(root.rglob("candidate_admissions.json")):
        rejected = read_json_file(path.with_name("quarantine.json"), [])
        outcomes.append(
            {
                "directory": str(path.parent.relative_to(root)),
                "durable_accepted_candidates": len(read_json_file(path, {})),
                "rejected_attempts": len(rejected),
            }
        )
    runs = []
    for path in sorted(root.rglob("state.json")):
        state = read_json_file(path, {})
        if "config" not in state:
            continue
        runs.append(
            {
                "directory": str(path.parent.relative_to(root)),
                "method": state["config"].get("method"),
                "seed": state["config"].get("seed"),
                "status": state.get("status"),
                "supervised_exposures": state.get("tokens"),
                "compute_accounting": state.get(
                    "compute_accounting", "unavailable_for_historical_run"
                ),
                "unused_final_round_candidates": sum(
                    len(x.get("unused_candidate_ids", []))
                    for x in state.get("rounds", [])
                ),
                "peak_gpu_bytes_last_session": state.get("peak_gpu_bytes"),
            }
        )
    return {
        "status": "recorded_usage_only",
        "requests": requests,
        "known_cost": dict(known),
        "unknown_reservations": dict(reserved),
        "attributed_groups": list(groups.values()),
        "cache_hits": sum(bool(x.get("cache_hit")) for x in accesses),
        "candidate_outcomes": outcomes,
        "student_runs": runs,
        "scope": "Cache hits are accesses, not additional provider charges. Missing historical ownership stays unattributed.",
        "shared_budget": read_json_file(root / "study_budget.json", None),
        "counted_spend": study_spend(root),
    }
