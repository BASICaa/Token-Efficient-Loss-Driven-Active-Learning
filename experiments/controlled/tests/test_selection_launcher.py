"""Exercise the real PowerShell launcher with a fake Python command only."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

SHELL = shutil.which("powershell") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(
    not SHELL, reason="PowerShell is required for launcher checks"
)
LAUNCHER = Path(__file__).resolve().parents[1] / "run_selection_comparison.ps1"


def quoted(path):
    return "'" + str(path).replace("'", "''") + "'"


def fixture_plan(tmp_path, corrupt=False):
    root = tmp_path / "study"
    root.mkdir()
    plan = {
        "settings": {"seeds": [42, 43, 44], "replays": ["off"], "updates": 25},
        "tokens_by_seed": {"42": 1000, "43": 1100, "44": 1200},
    }
    plan_path = root / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    for seed, selection, status, update in [
        (42, "random", "complete", 25),
        (42, "loss", "complete", 25),
        (42, "validation", "complete", 25),
        (43, "random", "complete", 25),
        (43, "loss", "running", 12),
    ]:
        path = root / f"seed_{seed}" / f"{selection}__replay_off" / "state.json"
        path.parent.mkdir(parents=True)
        state = {
            "identity": {
                "seed": seed,
                "selection": selection,
                "replay": "off",
                "plan_sha256": digest,
            },
            "status": status,
            "update": update,
            "tokens": (
                plan["tokens_by_seed"][str(seed)] if status == "complete" else 500
            ),
        }
        if corrupt and seed == 43 and selection == "loss":
            state["identity"]["plan_sha256"] = "wrong"
        path.write_text(json.dumps(state), encoding="utf-8")
    return root


def invoke(tmp_path, *, switch="", fail=False, corrupt=False):
    root = fixture_plan(tmp_path, corrupt)
    calls = tmp_path / "calls.json"
    harness = tmp_path / "harness.ps1"
    harness.write_text(
        "$ErrorActionPreference='Stop'\n"
        "$global:mockCalls=@()\n"
        "function global:python {\n"
        " $global:mockCalls+=,@($args)\n"
        f" $global:LASTEXITCODE={1 if fail else 0}\n"
        "}\n"
        "try {\n"
        f" & {quoted(LAUNCHER)} -Output {quoted(root)} {switch}\n"
        "} finally {\n"
        f" ConvertTo-Json -InputObject $global:mockCalls -Depth 8 | Set-Content -LiteralPath {quoted(calls)} -Encoding utf8\n"
        "}\n",
        encoding="utf-8-sig",
    )
    result = subprocess.run(
        [SHELL, "-NoProfile", "-File", str(harness)], capture_output=True, text=True
    )
    recorded = json.loads(calls.read_text(encoding="utf-8-sig"))
    return result, recorded


def test_launches_only_five_unfinished_arms_as_separate_invocations(tmp_path):
    result, calls = invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "4/9 complete; 5 remaining" in result.stdout
    assert len(calls) == 6
    assert [call[1] for call in calls] == ["run"] * 5 + ["report"]
    runs = [
        (str(call[call.index("--seed") + 1]), call[call.index("--selection") + 1])
        for call in calls[:-1]
    ]
    assert runs == [
        ("43", "loss"),
        ("43", "validation"),
        ("44", "random"),
        ("44", "loss"),
        ("44", "validation"),
    ]
    assert all("--allow-paid-api" not in call for call in calls)


def test_status_only_does_not_invoke_python(tmp_path):
    result, calls = invoke(tmp_path, switch="-StatusOnly")
    assert result.returncode == 0, result.stderr
    assert "4/9 complete; 5 remaining" in result.stdout
    assert calls == []


def test_report_only_never_starts_training(tmp_path):
    result, calls = invoke(tmp_path, switch="-ReportOnly")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1 and calls[0][1] == "report"


def test_failure_stops_sweep_and_preserves_resume_instruction(tmp_path):
    result, calls = invoke(tmp_path, fail=True)
    assert result.returncode != 0
    assert len(calls) == 1
    assert "Committed checkpoints are preserved" in result.stderr


def test_changed_identity_is_rejected_before_training(tmp_path):
    result, calls = invoke(tmp_path, corrupt=True)
    assert result.returncode != 0
    assert "identity differs" in result.stderr
    assert calls == []
