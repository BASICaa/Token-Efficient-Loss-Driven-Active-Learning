import copy
from collections import Counter
import random

import pytest

import run_bank_recovery as recovery
from TEFLD.src.candidate_pool import CandidatePool, candidate_id
from TEFLD.src.dataschema import Training_Sample
from TEFLD.src.helper import read_json_file, write_json_file
from test_runner import CATEGORIES, SavedTokenizer, fixture_runner


def sample(category, serial):
    return Training_Sample(
        instruct=f"Explain training fixture {serial} in {category}.",
        sample="",
        gold_summary="A fixture answer.",
        tag=category,
        source="generated",
        verification={
            "status": "accepted_probabilistic_review",
            "result": {key: True for key in recovery.CHECKS},
        },
    )


def setup(runner, tmp_path, bank):
    runner.__class__ = recovery.RecoveryRunner
    tokenizer = SavedTokenizer()
    pool = CandidatePool(
        runner.data, tokenizer, runner.teacher, tmp_path / "quarantine.json"
    )
    state = {
        "round": 2,
        "ledger": [],
        "seen": [],
        "weakness": {c: i + 0.1 for i, c in enumerate(CATEGORIES)},
    }
    return tokenizer, pool, state


def invoke(runner, tokenizer, pool, state, bank, seed=42, replay=True):
    return runner._batch(
        state,
        "tefld_loss",
        seed,
        state["round"],
        random.Random(seed * 1000003 + state["round"]),
        pool,
        None,
        tokenizer,
        replay,
        bank,
        None,
        [],
    )


@pytest.mark.parametrize("seed", [1, 42, 123])
@pytest.mark.parametrize("replay", [True, False])
def test_adequate_bank_matches_frozen_engine_without_generation(
    fixture_runner, tmp_path, monkeypatch, seed, replay
):
    runner = fixture_runner("parity")
    bank = [sample(c, i).model_dump(mode="json") for c in CATEGORIES for i in range(20)]
    tokenizer, pool, state = setup(runner, tmp_path, bank)
    for i in range(5):
        item = sample(CATEGORIES[i], 100 + i)
        state["ledger"].append(
            {
                "id": candidate_id(item),
                "sample": item.model_dump(mode="json"),
                "loss": 1.0,
                "loss_round": 1,
                "replay_uses": 0,
                "last_replay_round": None,
            }
        )
    expected_state = copy.deepcopy(state)
    rng = random.Random(seed * 1000003 + state["round"])
    from TEFLD.src.experiment_runner import ExperimentRunner

    expected = ExperimentRunner._batch(
        runner,
        expected_state,
        "tefld_loss",
        seed,
        state["round"],
        rng,
        pool,
        None,
        tokenizer,
        replay,
        bank,
        None,
        [],
    )
    monkeypatch.setattr(
        pool, "create", lambda *a, **kw: pytest.fail("Unnecessary generation")
    )
    actual = invoke(runner, tokenizer, pool, state, bank, seed, replay)
    assert [x.model_dump() for x in actual] == [x.model_dump() for x in expected]
    assert state == expected_state


def test_only_deficits_generated_and_restart_reuses_durable_admissions(
    fixture_runner, tmp_path, monkeypatch
):
    runner = fixture_runner("shortage")
    bank = [sample("classification", 0).model_dump(mode="json")]
    tokenizer, pool, initial = setup(runner, tmp_path, bank)
    required = recovery.required_categories(
        initial, CATEGORIES, random.Random(42 * 1000003 + 2), True
    )
    expected = required - Counter(x["tag"] for x in bank)
    calls = []

    def create(category, nonce, seen, weakness):
        calls.append(category)
        item = sample(category, 1000 + len(calls))
        payload = read_json_file(pool.admissions_path, {})
        payload[nonce] = {
            "teacher_identity": runner.teacher.identity,
            "sample": item.model_dump(mode="json"),
        }
        write_json_file(pool.admissions_path, payload)
        seen.add(candidate_id(item))
        return item

    monkeypatch.setattr(pool, "create", create)
    first = invoke(runner, tokenizer, pool, copy.deepcopy(initial), bank)
    assert Counter(calls) == expected
    assert len(first) == 10
    monkeypatch.setattr(
        pool, "create", lambda *a, **k: pytest.fail("Regenerated durable admission")
    )
    second = invoke(runner, tokenizer, pool, copy.deepcopy(initial), bank)
    assert [candidate_id(x) for x in first] == [candidate_id(x) for x in second]


def test_partial_replenishment_failure_preserves_round_and_accepted_candidates(
    fixture_runner, tmp_path, monkeypatch
):
    runner = fixture_runner("partial")
    bank = [sample("classification", 0).model_dump(mode="json")]
    tokenizer, pool, state = setup(runner, tmp_path, bank)
    before = copy.deepcopy(state)
    calls = [0]

    def create(category, nonce, seen, weakness):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("Shared teacher-study spend ceiling reached.")
        item = sample(category, 1000)
        write_json_file(
            pool.admissions_path,
            {
                nonce: {
                    "teacher_identity": runner.teacher.identity,
                    "sample": item.model_dump(mode="json"),
                }
            },
        )
        return item

    monkeypatch.setattr(pool, "create", create)
    with pytest.raises(RuntimeError, match="spend ceiling"):
        invoke(runner, tokenizer, pool, state, bank)
    assert state == before
    assert len(read_json_file(pool.admissions_path, {})) == 1


def test_recovery_rejects_unverified_admissions(fixture_runner, tmp_path):
    runner = fixture_runner("invalid")
    bank = [sample("classification", 0).model_dump(mode="json")]
    tokenizer, pool, state = setup(runner, tmp_path, bank)
    item = sample("general_qa", 999)
    item.verification["result"]["correct"] = False
    write_json_file(
        pool.admissions_path,
        {
            "bad": {
                "teacher_identity": runner.teacher.identity,
                "sample": item.model_dump(mode="json"),
            }
        },
    )
    with pytest.raises(ValueError, match="verification"):
        invoke(runner, tokenizer, pool, state, bank)


@pytest.mark.parametrize(
    "allow,budget", [(False, 900000), (True, None), (True, 0), (True, float("nan"))]
)
def test_run_requires_explicit_finite_budget_before_work(tmp_path, allow, budget):
    with pytest.raises(ValueError, match="requires"):
        recovery.run(tmp_path, allow_paid=allow, budget=budget)


def test_fork_output_cannot_replace_source(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        recovery.prepare(
            tmp_path, tmp_path, tmp_path / "bank.json", tmp_path / "budget"
        )


def test_prepare_forks_state_and_pins_policy_without_changing_source(
    tmp_path, monkeypatch
):
    import TEFLD.src.experiment_runner as engine
    from test_runner import FixtureData

    class Data(FixtureData):
        def __init__(self, path):
            super().__init__(path)
            self.protocol.update(read_json_file(path, {}))

    monkeypatch.setattr(engine, "SplitData", Data)
    source = tmp_path / "original"
    output = tmp_path / "recovery" / "loss"
    protocol_path = source / "protocol.json"
    write_json_file(
        protocol_path,
        {
            "controlled_entry": {
                "mode": "api-loss",
                "method": "tefld_loss",
                "seed": 42,
                "safety_tokens": 1000,
                "per_category": 8,
            },
            "teacher_comparison": {"arm": "api-loss"},
        },
    )
    original = engine.ExperimentRunner(protocol_path, source)
    bank = [sample("general_qa", 1).model_dump(mode="json")]
    bank_path = tmp_path / "bank.json"
    write_json_file(bank_path, bank)
    write_json_file(
        bank_path.with_suffix(".manifest.json"),
        {
            "bank_sha256": recovery.stable_json_hash({"bank": bank}),
            "teacher_identity": original.teacher.identity,
        },
    )
    write_json_file(
        source / "study_plan.json",
        {
            "core_code_sha256": original.code_hash,
            "protocol_sha256": original.protocol_hash,
        },
    )
    write_json_file(
        source / "verification_audit.json",
        {"status": "passed", "teacher_identity": original.verifier.identity},
    )
    state_path = original.run_directory("pilot", "tefld_loss", 42) / "state.json"
    write_json_file(
        state_path,
        {
            "status": "running",
            "round": 13,
            "tokens": 1234,
            "batch": [],
            "ledger": [bank[0]],
            "config": {
                "code_sha256": original.code_hash,
                "protocol_sha256": original.protocol_hash,
                "bank_sha256": recovery.stable_json_hash({"bank": bank}),
            },
        },
    )
    before = state_path.read_bytes()
    report = recovery.prepare(source, output, bank_path, output.parent)
    assert report["provider_calls_during_prepare"] == 0
    assert report["inherited_unique_examples"] == 1
    assert state_path.read_bytes() == before
    fork = read_json_file(
        output / "arms/api-loss/pilot/tefld_loss_seed_42/state.json", {}
    )
    assert fork["tokens"] == 1234 and fork["round"] == 13
    assert fork["config"]["code_sha256"] != original.code_hash
    assert fork["recovery"]["source_sha256"] == recovery.file_fingerprint(state_path)
    with pytest.raises(ValueError, match="exists"):
        recovery.prepare(source, output, bank_path, output.parent)
