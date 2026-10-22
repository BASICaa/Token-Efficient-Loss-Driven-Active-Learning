import copy
from pathlib import Path
import random

import pytest

from TEFLD.src.candidate_pool import candidate_id
from TEFLD.src.example_budget import cap_new_examples, example_target
from TEFLD.src.helper import read_json_file
from test_runner import dataset_runner, fixture_runner  # noqa: F401


def test_cap_keeps_replay_and_counts_new_examples_only(fixture_runner):
    runner = fixture_runner("cap")
    samples = [runner.data.sample(i, "training_pool") for i in range(10)]
    ledger = [{"id": candidate_id(samples[0])}]
    samples[0].source = "recycled"
    selected, omitted = cap_new_examples(samples, ledger, 4, random.Random(42))
    assert len(selected) == 4 and samples[0] in selected
    assert len(omitted) == 6
    assert candidate_id(samples[0]) not in omitted
    assert cap_new_examples(samples, ledger, 4, random.Random(42)) == (
        selected,
        omitted,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        cap_new_examples([samples[1], samples[1]], ledger, 4, random.Random(42))


@pytest.mark.parametrize(
    "method",
    [
        "dolly_baseline",
        "tefld_dataset_loss",
        "tefld_dataset_validation",
        "tefld_loss",
        "tefld_validation",
    ],
)
def test_exact_100_unique_examples_complete_answers_and_original_evaluation(
    fixture_runner, method
):
    factory = (
        dataset_runner
        if method.startswith("tefld_dataset_")
        else lambda fixture, name: fixture(name)
    )
    runner = factory(fixture_runner, "count_" + method)
    runner.protocol["example_count_control"] = {"unique_training_examples": 100}
    bank = (
        None
        if method == "dolly_baseline" or method.startswith("tefld_dataset_")
        else [
            runner.data.sample(i, "training_pool").model_dump(mode="json")
            for i in range(240)
        ]
    )
    result = runner.run(method, 42, budget=10000, candidate_bank=bank)
    assert result["status"] == "complete"
    assert len({entry["id"] for entry in result["ledger"]}) == 100
    assert result["tokens"] < 10000
    assert not any(r["partial_final_round"] for r in result["rounds"])
    assert all(p["epoch_presentations"] == 3 for p in result["presentations"])
    assert len(result["candidates"]) == 1
    assert result["final_checkpoint"] == result["selected_checkpoint"]
    scores = read_json_file(
        Path(result["final_checkpoint"]["path"]) / "validation.json", {}
    )
    assert [r["row_id"] for r in scores["rows"]] == sorted(
        runner.data.ids["checkpoint_validation"]
    )
    assert "final_test" not in runner.data.access
    assert ("curriculum_validation" in runner.data.access) == (
        method in {"tefld_validation", "tefld_dataset_validation"}
    )
    assert runner.run(method, 42, budget=10000, candidate_bank=bank) == result


def test_count_mode_cannot_silently_finish_at_token_ceiling(fixture_runner):
    runner = fixture_runner("ceiling")
    runner.protocol["example_count_control"] = {"unique_training_examples": 100}
    with pytest.raises(RuntimeError, match="Safety token ceiling"):
        runner.run("dolly_baseline", 42, budget=1)
    path = runner.run_directory("pilot", "dolly_baseline", 42) / "state.json"
    assert read_json_file(path, {})["tokens"] == 0


def test_count_mode_interruption_recovers_without_extra_unique_examples(
    fixture_runner, monkeypatch
):
    runner = fixture_runner("recover_count")
    runner.protocol["example_count_control"] = {"unique_training_examples": 13}
    original = runner._batch

    def fail_next_round(state, *args, **kwargs):
        if state["round"] == 1:
            raise RuntimeError("interruption")
        return original(state, *args, **kwargs)

    monkeypatch.setattr(runner, "_batch", fail_next_round)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.run("dolly_baseline", 42, budget=10000)
    monkeypatch.setattr(runner, "_batch", original)
    result = runner.run("dolly_baseline", 42, budget=10000)
    assert len(result["ledger"]) == 13
    assert len(result["untrained_final_admissions"]) == 7


def test_count_control_is_opt_in_and_development_only():
    assert example_target({}, "primary", "dolly_baseline") is None
    for count in (True, 0, 9, 100.0):
        with pytest.raises(ValueError):
            example_target(
                {"example_count_control": {"unique_training_examples": count}},
                "pilot",
                "dolly_baseline",
            )
    with pytest.raises(ValueError):
        example_target(
            {"example_count_control": {"unique_training_examples": 100}},
            "primary",
            "dolly_baseline",
        )


def test_five_profiles_keep_the_five_methods_and_shared_verifier():
    from run_five_way_training import ARMS, ROOT, profile, read

    base = read(ROOT / "experiment_design_2026-10-18/experiment_protocol_hibana.json")
    before = copy.deepcopy(base)
    assert len(ARMS) == 5
    for arm, (method, teacher) in ARMS.items():
        settings = profile(base, arm)
        assert example_target(settings, "pilot", method) == 100
        assert (
            settings["tefld_controls"]["verifier"]["provider"]
            == base["tefld_controls"]["provider"]
        )
        assert settings["common_training"]["epochs_per_round"] == 3
        assert settings["teacher_comparison"]["generator_factor"] == teacher
    luna = profile(base, "luna_validation")
    assert ARMS["luna_validation"][0] == "tefld_validation"
    assert luna["tefld_controls"]["teacher_model_id"] == "gpt-6-luna"
    assert base == before


def test_offline_preparation_pins_real_64_row_evaluation_and_disables_paid_calls(
    tmp_path, monkeypatch
):
    from run_five_way_training import ARMS, prepare, read
    from TEFLD.src.candidate_pool import Teacher
    from run_five_way_training import ROOT

    if not (
        ROOT / "experiment_runs_two_teachers_dataset_extension/study_plan.json"
    ).exists():
        pytest.skip(
            "Historical raw run inputs are intentionally not distributed; exported result integrity is tested separately."
        )

    monkeypatch.setattr(
        Teacher, "ask", lambda *a, **k: pytest.fail("Unexpected API call")
    )
    root = tmp_path / "prepared"
    plan = prepare(root)
    assert len(plan["evaluation_row_ids"]) == 64
    assert plan["unique_training_examples_per_arm"] == 100
    assert read(root / "study_budget.json")["approved"] is False
    assert len(read(root / "shared_initial_seed_42.json")) == 10
    assert all((root / "protocols" / (arm + ".json")).exists() for arm in ARMS)
    assert prepare(root) == plan


@pytest.mark.parametrize("arm", ["sol61_validation", "sol61_loss", "luna_validation"])
def test_launcher_binds_generator_and_verifier_limits_before_network(
    tmp_path, monkeypatch, arm
):
    from run_five_way_training import ROOT, configure_client_limits, profile, read
    from TEFLD.src.candidate_pool import Teacher
    from TEFLD.src.experiment_runner import ExperimentRunner
    from TEFLD.src.helper import write_json_file
    from TEFLD.src import experiment_runner as runner_module
    from types import SimpleNamespace

    monkeypatch.setattr(runner_module, "SplitData", lambda path: SimpleNamespace(
        protocol_path=Path(path), protocol=read(path)
    ))

    monkeypatch.setattr(
        Teacher, "ask", lambda *a, **k: pytest.fail("Unexpected API call")
    )
    base = read(ROOT / "experiment_design_2026-10-18/experiment_protocol_hibana.json")
    path = tmp_path / "protocol.json"
    write_json_file(path, profile(base, arm))
    runner = ExperimentRunner(path, tmp_path / "study", allow_paid_api=False)
    identities = [runner.teacher.identity, runner.verifier.identity]
    for client in (runner.teacher, runner.verifier):
        # Reproduce the incomplete files written by the previous launcher.
        write_json_file(
            client.directory / "spend_limit.json",
            {"currency": "toman", "max_spend": 550000},
        )
    with pytest.raises(ValueError, match="different provider/model"):
        configure_client_limits(runner)
    configure_client_limits(runner, 550000)
    assert [runner.teacher.identity, runner.verifier.identity] == identities
    for client, ceiling in ((runner.teacher, 400), (runner.verifier, 1200)):
        assert client.spend_ceiling() == 550000
        assert client.max_requests == ceiling
        assert (
            read(client.directory / "spend_limit.json")["teacher_identity"]
            == client.identity
        )


def test_only_known_launcher_only_repair_can_preserve_existing_plan(tmp_path):
    from run_five_way_training import require_study_plan, read
    from TEFLD.src.helper import write_json_file

    prior = {
        "launcher_sha256": "60efd378b42359ecfb2822cf3bb0e5bb770cc01eab439be59219f78b95cc7689",
        "core_code_sha256": "unchanged-core",
        "evaluation_row_ids": [1, 2],
    }
    path = tmp_path / "study_plan.json"
    write_json_file(path, prior)
    repaired = {**prior, "launcher_sha256": "repaired-launcher"}
    with pytest.raises(ValueError, match="fresh"):
        require_study_plan(path, {**repaired, "core_code_sha256": "different-core"})
    assert read(path) == prior
    require_study_plan(path, repaired)
    assert read(path) == repaired
    assert read(tmp_path / "study_plan_before_spend_limit_fix.json") == prior
    assert (
        read(tmp_path / "launcher_spend_limit_fix.json")["core_training_code_changed"]
        is False
    )
    require_study_plan(path, repaired)
    with pytest.raises(ValueError, match="fresh"):
        require_study_plan(
            path, {**repaired, "launcher_sha256": "unrelated-later-change"}
        )
