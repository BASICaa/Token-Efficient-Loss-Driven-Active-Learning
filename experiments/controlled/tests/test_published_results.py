import json
from pathlib import Path


def test_snapshot_is_complete_and_reconciles_scores_without_private_data():
    path = Path(__file__).resolve().parents[3] / "results/five-way-100/snapshot.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert len(snapshot["methods"]) == 5
    assert snapshot["local_validation_guided_result"] is None
    assert snapshot["final_test_scored"] is False
    contracts = []
    for method in snapshot["methods"].values():
        assert method["unique_training_examples"] == 100
        rows = method["evaluation_rows"]
        assert len(rows) == 64
        contracts.append(
            [(r["row_id"], r["category"], r["answer_tokens"]) for r in rows]
        )
        means = [
            sum(r["loss"] for r in rows if r["category"] == c) / 8
            for c in method["categories"]
        ]
        assert abs(sum(means) / 8 - method["macro_nll"]) < 1e-12
        assert "checkpoint" not in method
        assert all("instruct" not in r and "gold_summary" not in r for r in rows)
    assert all(contract == contracts[0] for contract in contracts)
    total = sum(sum(m["api_cost_toman"].values()) for m in snapshot["methods"].values())
    assert abs(total - snapshot["total_api_cost_toman"]) < 1e-6
