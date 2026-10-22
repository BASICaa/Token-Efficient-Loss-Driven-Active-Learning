import copy
from pathlib import Path
import shutil
import subprocess

import pytest

import prepare_optimizer_study as optimizer
from TEFLD.src.helper import read_json_file, write_json_file
from TEFLD.src.runtime import file_fingerprint
from test_replay_preparation import source_study


def prepared(tmp_path, monkeypatch):
    source = source_study(tmp_path, monkeypatch)
    monkeypatch.setattr(optimizer, "code_identity", lambda: "fixture-code")
    plan_path = source / "plan.json"
    plan = read_json_file(plan_path, {})
    plan["settings"]["optimizer_lifecycle"] = "reset-per-round"
    write_json_file(plan_path, plan)
    path = source / "seed_42/random__replay_off/state.json"
    state = read_json_file(path, {})
    state["identity"]["plan_sha256"] = file_fingerprint(plan_path)
    for observation in state["evaluations"]:
        observation["checkpoint"].update(macro_nll=1.8, token_weighted_nll=2.0)
        observation["warmup_probe"] = {"macro_nll": 2.0}
        observation["tokens"] = observation["update"] * 4
    for update in state["updates"]:
        update.update(optimizer_reset=True, learning_rate=0.0002, clipped=False)
    write_json_file(path, state)
    return source


def test_prepares_one_fresh_run_with_unchanged_bank_schedule_and_control(
    tmp_path, monkeypatch
):
    source = prepared(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in source.rglob("*.json")}
    destination = tmp_path / "continuous"
    result = optimizer.prepare(source, destination)
    assert (
        result["new_runs"] == 1
        and not result["training_started"]
        and result["provider_calls"] == 0
    )
    old, new = [read_json_file(p / "plan.json", {}) for p in (source, destination)]
    assert new["settings"]["optimizer_lifecycle"] == "continuous"
    assert new["settings"]["seeds"] == [42] and new["settings"]["replays"] == ["off"]
    assert new["schedules"]["42"] == old["schedules"]["42"]
    assert new["tokens_by_seed"]["42"] == old["tokens_by_seed"]["42"]
    assert (destination / "bank.json").read_bytes() == (
        source / "bank.json"
    ).read_bytes()
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not list(destination.glob("seed_*/**/state.json"))
    assert optimizer.report(destination, figures=False)["remaining_runs"] == 1
    with pytest.raises(ValueError, match="exists"):
        optimizer.prepare(source, destination)


def completed(tmp_path, monkeypatch):
    source = prepared(tmp_path, monkeypatch)
    destination = tmp_path / "continuous"
    optimizer.prepare(source, destination)
    control = read_json_file(destination / "reset_control.json", {})
    state = copy.deepcopy(control)
    state["identity"]["plan_sha256"] = file_fingerprint(destination / "plan.json")
    for step in state["updates"]:
        step["optimizer_reset"] = False
    state["evaluations"][-1]["checkpoint"]["macro_nll"] = 1.7
    path = destination / "seed_42/random__replay_off/state.json"
    write_json_file(path, state)
    return destination, path


def test_reports_same_data_contrast_and_figures_without_model(tmp_path, monkeypatch):
    destination, _ = completed(tmp_path, monkeypatch)
    result = optimizer.report(destination)
    assert result["same_selected_data_and_order"] and result["remaining_runs"] == 0
    assert result["delta_nll_continuous_minus_reset"] == pytest.approx(-0.1)
    assert (destination / "optimizer_comparison.png").stat().st_size > 1000
    assert result["final_test_scored"] is False


@pytest.mark.parametrize(
    "mutation", ["data", "tokens", "initial", "identity", "control"]
)
def test_report_rejects_incompatible_comparison(tmp_path, monkeypatch, mutation):
    destination, path = completed(tmp_path, monkeypatch)
    state = read_json_file(path, {})
    if mutation == "data":
        state["selection_trace"][-1]["candidate_id"] = "different"
    if mutation == "tokens":
        state["tokens"] -= 1
    if mutation == "initial":
        state["evaluations"][0]["checkpoint"]["macro_nll"] = 99
    if mutation == "identity":
        state["identity"]["plan_sha256"] = "different"
    if mutation == "control":
        with (destination / "reset_control.json").open("a") as file:
            file.write(" ")
    write_json_file(path, state)
    with pytest.raises(ValueError):
        optimizer.report(destination, figures=False)


def test_launcher_invokes_only_random_off_and_report(tmp_path, monkeypatch):
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if not shell:
        pytest.skip("PowerShell required")
    source = prepared(tmp_path, monkeypatch)
    destination = tmp_path / "continuous"
    optimizer.prepare(source, destination)
    launcher = Path(optimizer.__file__).with_name("run_continuous_optimizer.ps1")
    calls = tmp_path / "calls.json"
    quote = lambda p: "'" + str(p).replace("'", "''") + "'"
    harness = tmp_path / "harness.ps1"
    harness.write_text(
        "$ErrorActionPreference='Stop'\n$global:calls=@()\nfunction global:python { $global:calls+=,@($args); $global:LASTEXITCODE=0 }\n"
        f"& {quote(launcher)} -Output {quote(destination)}\n"
        f"ConvertTo-Json -InputObject $global:calls -Depth 8 | Set-Content {quote(calls)} -Encoding utf8\n",
        encoding="utf-8-sig",
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-File", str(harness)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    import json

    actual = json.loads(calls.read_text(encoding="utf-8-sig"))
    assert len(actual) == 2
    assert actual[0][0:2] == ["run_selection_diagnostic.py", "run"]
    assert actual[0][actual[0].index("--selection") + 1] == "random"
    assert actual[0][actual[0].index("--replay") + 1] == "off"
    assert actual[1][0:2] == ["prepare_optimizer_study.py", "report"]
