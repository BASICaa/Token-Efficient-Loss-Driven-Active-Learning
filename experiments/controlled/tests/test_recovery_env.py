import os
import sys

import pytest

import recovery_env


def test_file_key_loaded_without_other_variables_or_interpolation(tmp_path, monkeypatch):
    monkeypatch.delenv("HIBANA_API_KEY", raising=False)
    monkeypatch.delenv("UNRELATED_RECOVERY_FIXTURE", raising=False)
    path = tmp_path / ".env"
    path.write_text('HIBANA_API_KEY="fixture-${UNRELATED_RECOVERY_FIXTURE}"\nUNRELATED_RECOVERY_FIXTURE=ignore\n', encoding="utf-8-sig")
    recovery_env.load_credentials(path)
    assert os.environ["HIBANA_API_KEY"] == "fixture-${UNRELATED_RECOVERY_FIXTURE}"
    assert "UNRELATED_RECOVERY_FIXTURE" not in os.environ


def test_existing_process_key_has_priority(tmp_path, monkeypatch):
    monkeypatch.setenv("HIBANA_API_KEY", "session-fixture")
    path = tmp_path / ".env"
    path.write_text("HIBANA_API_KEY=file-fixture\n", encoding="utf-8")
    recovery_env.load_credentials(path)
    assert os.environ["HIBANA_API_KEY"] == "session-fixture"


@pytest.mark.parametrize("content", ["", "HIBANA_API_KEY=\n", "HIBANA_API_KEY='  '\n"])
def test_missing_or_empty_key_fails(tmp_path, monkeypatch, content):
    monkeypatch.delenv("HIBANA_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text(content)
    with pytest.raises(ValueError, match="missing or empty"):
        recovery_env.load_credentials(path)


def test_check_hides_key_and_never_invokes_training(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HIBANA_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("HIBANA_API_KEY=private-fixture-key\n")
    monkeypatch.setattr(sys, "argv", ["recovery_env.py", "--env-file", str(path), "--check"])
    monkeypatch.setattr(recovery_env.runpy, "run_path", lambda *a, **k: pytest.fail("Started recovery"))
    recovery_env.main()
    output = capsys.readouterr()
    assert "private-fixture-key" not in output.out + output.err
    assert "No API calls" in output.out


def test_recovery_arguments_forwarded_only_to_recovery_entrypoint(tmp_path, monkeypatch):
    monkeypatch.delenv("HIBANA_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("HIBANA_API_KEY=private-fixture-key\n")
    args = ["run", "--output", "fixture", "--allow-paid-api", "--teacher-budget-toman", "900000"]
    monkeypatch.setattr(sys, "argv", ["recovery_env.py", "--env-file", str(path), "--", *args])
    calls = []
    def invoke(target, run_name):
        calls.append((target, run_name, list(sys.argv)))
    monkeypatch.setattr(recovery_env.runpy, "run_path", invoke)
    recovery_env.main()
    assert len(calls) == 1
    assert calls[0][0].endswith("run_bank_recovery.py")
    assert calls[0][1] == "__main__" and calls[0][2][1:] == args
