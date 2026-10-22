import copy
import random

import pytest

import run_selection_diagnostic as engine
import run_validation_replay as replay
from TEFLD.src.helper import read_json_file, write_json_file
from test_selection_diagnostic import study, tiny_cpu_threads


def bank():
    return {
        f"{c}{i}": {"category": c, "answer_tokens": 4, "origin": "fixture"}
        for c in ("weak", "strong")
        for i in range(6)
    }


def inputs():
    records = bank()
    slots = [
        {"length": 4, "warmup_id": None, "replay_slot": True},
        {"length": 4, "warmup_id": None, "replay_slot": False},
    ]
    seen = {"weak0", "weak1", "strong0", "strong1"}
    return records, slots, seen


def test_validation_replay_prefers_weak_category_and_ignores_training_loss():
    records, slots, seen = inputs()
    weak = 0
    for seed in range(150):
        one, traces = replay.select_group(
            records,
            slots,
            seen,
            "validation",
            "validation",
            {"weak": 100, "strong": 0.01},
            {},
            seed,
        )
        two, _ = replay.select_group(
            records,
            slots,
            seen,
            "validation",
            "validation",
            {"weak": 100, "strong": 0.01},
            {i: random.random() for i in records},
            seed,
        )
        assert one == two
        assert one[0] in seen and one[1] not in seen
        assert traces[0]["eligible_categories"] == 2
        assert (
            traces[0]["replay_feedback_source"] == "curriculum_validation_category_nll"
        )
        weak += records[one[0]]["category"] == "weak"
    assert weak > 145


def test_replay_extra_random_draws_do_not_change_new_example_rng():
    records, slots, seen = inputs()
    for seed in range(30):
        a, _ = replay.select_group(
            records,
            slots,
            seen,
            "validation",
            "random",
            {"weak": 2, "strong": 1},
            {},
            seed,
        )
        b, _ = replay.select_group(
            records,
            slots,
            seen,
            "validation",
            "validation",
            {"weak": 2, "strong": 1},
            {},
            seed,
        )
        assert a[1:] == b[1:]


def test_random_replay_does_not_use_validation_to_choose_replayed_example():
    records, slots, seen = inputs()
    for seed in range(30):
        a, _ = replay.select_group(
            records,
            slots,
            seen,
            "validation",
            "random",
            {"weak": 100, "strong": 0.01},
            {},
            seed,
        )
        b, _ = replay.select_group(
            records,
            slots,
            seen,
            "validation",
            "random",
            {"weak": 0.01, "strong": 100},
            {},
            seed,
        )
        assert a[0] == b[0]


@pytest.mark.parametrize(
    "weakness", [{}, {"weak": float("nan"), "strong": 1}, {"weak": -1, "strong": 1}]
)
def test_bad_validation_feedback_rejected(weakness):
    records, slots, seen = inputs()
    with pytest.raises(ValueError):
        replay.select_group(
            records, slots, seen, "validation", "validation", weakness, {}, 42
        )


def test_single_category_replay_is_not_silently_accepted():
    records, slots, _ = inputs()
    with pytest.raises(ValueError, match="real category choice"):
        replay.select_group(
            records,
            slots,
            {"weak0"},
            "validation",
            "validation",
            {"weak": 2, "strong": 1},
            {},
            42,
        )


def test_balanced_warmup_guarantees_two_categories_and_sufficient_new_candidates():
    records = {
        f"id{i}": {
            "category": f"c{i % 4}",
            "answer_tokens": 2 + (i // 4) % 4,
            "origin": "fixture",
        }
        for i in range(120)
    }
    original = engine.make_schedule(records, 42, 10, 4, 2)
    before = copy.deepcopy(original)
    schedule = replay.balanced_schedule(records, original, 2, 42)
    seen = {s["warmup_id"] for group in schedule[:2] for s in group}
    for group in schedule[2:]:
        length = group[0]["length"]
        assert (
            len(
                {
                    records[i]["category"]
                    for i in seen
                    if records[i]["answer_tokens"] == length
                }
            )
            >= 2
        )
        assert group[0]["replay_slot"] and not any(s["replay_slot"] for s in group[1:])
    assert before == original


def test_two_run_end_to_end_and_resumption(study):
    prepare, access = study
    source = prepare()
    output = source.parent / "validation-replay"
    source_plan = (source / "plan.json").read_bytes()
    info = replay.prepare(source, output)
    assert info["new_runs"] == 2 and not info["training_started"]
    assert (source / "plan.json").read_bytes() == source_plan
    core_identity, core_selector = engine.code_identity, engine.select_group
    for policy in replay.POLICIES:
        result = replay.run(output, policy)
        assert result["status"] == "complete"
        assert replay.run(output, policy) == result
        assert (
            engine.code_identity is core_identity
            and engine.select_group is core_selector
        )
    summary = replay.report(output, figures=True)
    assert (
        summary["remaining_runs"] == 0 and not summary["old_scores_directly_comparable"]
    )
    assert len({r["tokens"] for r in summary["results"]}) == 1
    assert len({r["updates"] for r in summary["results"]}) == 1
    assert all(r["minimum_eligible_replay_categories"] >= 2 for r in summary["results"])
    assert "final_test" not in access
    assert (output / "validation_replay_comparison.png").stat().st_size > 1000
    # Reporting an interruption before the second arm's warmup must not mistake
    # an unfinished prefix for an incompatible initial batch.
    path = output / "seed_42/validation__replay_validation/state.json"
    state = read_json_file(path, {})
    state.update(
        status="running", update=0, tokens=0, seen=[], selection_trace=[], updates=[]
    )
    state["evaluations"] = state["evaluations"][:1]
    write_json_file(path, state)
    assert replay.report(output, figures=False)["remaining_runs"] == 1


def test_extension_hooks_restored_on_failure():
    a, b = engine.code_identity, engine.select_group
    with pytest.raises(RuntimeError):
        with replay.extension():
            raise RuntimeError("fixture")
    assert engine.code_identity is a and engine.select_group is b


def test_interrupted_validation_replay_matches_uninterrupted_run(study, monkeypatch):
    prepare, _ = study
    source = prepare()
    clean, broken = source.parent / "clean", source.parent / "broken"
    replay.prepare(source, clean)
    replay.prepare(source, broken)
    replay.run(clean, "validation")
    original = engine.optimizer_step
    calls = [0]

    def stop(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError("fixture interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "optimizer_step", stop)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        replay.run(broken, "validation")
    monkeypatch.setattr(engine, "optimizer_step", original)
    replay.run(broken, "validation")
    a, b = [
        read_json_file(root / "seed_42/validation__replay_validation/state.json", {})
        for root in (clean, broken)
    ]
    assert a["selection_trace"] == b["selection_trace"]
    assert a["updates"] == b["updates"]
    assert a["evaluations"] == b["evaluations"]


def test_preparation_rejects_modified_source_before_creating_output(study):
    prepare, _ = study
    source = prepare()
    with (source / "bank.json").open("a") as f:
        f.write(" ")
    output = source.parent / "invalid"
    with pytest.raises(ValueError, match="changed"):
        replay.prepare(source, output)
    assert not output.exists()


@pytest.mark.parametrize("report_only", [False, True])
def test_launcher_uses_two_separate_policy_calls_or_report_only(tmp_path, report_only):
    import json
    from pathlib import Path
    import shutil
    import subprocess

    shell = shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        pytest.skip("PowerShell required")
    quote = lambda p: "'" + str(p).replace("'", "''") + "'"
    launcher = Path(replay.__file__).with_suffix(".ps1")
    calls = tmp_path / "calls.json"
    harness = tmp_path / "harness.ps1"
    harness.write_text(
        "$ErrorActionPreference='Stop'\n$global:calls=@()\nfunction global:python {$global:calls+=,@($args);$global:LASTEXITCODE=0}\n"
        f"& {quote(launcher)} -Output {quote(tmp_path)} {'-ReportOnly' if report_only else ''}\n"
        f"ConvertTo-Json -InputObject $global:calls -Depth 8 | Set-Content {quote(calls)} -Encoding utf8\n",
        encoding="utf-8-sig",
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-File", str(harness)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    recorded = json.loads(calls.read_text(encoding="utf-8-sig"))
    assert [c[1] for c in recorded] == (
        ["report"] if report_only else ["run", "run", "report"]
    )
    if not report_only:
        assert [c[c.index("--policy") + 1] for c in recorded[:2]] == [
            "random",
            "validation",
        ]
    assert all("--allow-paid-api" not in c for c in recorded)
