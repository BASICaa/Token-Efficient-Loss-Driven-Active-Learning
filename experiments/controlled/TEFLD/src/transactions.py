"""Redo-journal commits for one section's ledger/vault/state/result bundle.

Only one writer may own a section. Locks fail closed on a second writer; an
abandoned lock is released automatically by the OS. Prepared bundles replay
identically after process death; completed bundles never overwrite later state.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import threading
from pathlib import Path

_LOCAL = threading.local()


@contextmanager
def section_lock(directory: Path):
    directory = Path(directory).resolve()
    key = str(directory)
    active = getattr(_LOCAL, "active", set())
    if key in active:
        yield
        return
    directory.mkdir(parents=True, exist_ok=True)
    handle = open(directory / ".writer.lock", "a+b")
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _LOCAL.active = active | {key}
        yield
    finally:
        _LOCAL.active = active
        # Closing releases the OS lock even when acquisition/commit failed.
        handle.close()


def recover(directory: Path, *, locked=False):
    from .helper import read_json_file, write_json_file

    journal = directory / "round_commit.json"
    if not journal.exists():
        return
    payload = read_json_file(journal, {})
    if payload.get("status") != "prepared":
        return
    if not locked:
        with section_lock(directory):
            return recover(directory, locked=True)
    for name, content in payload["files"].items():
        if (
            name
            not in {
                "OverAllData.json",
                "failure_vault.json",
                "pipeline_state.json",
                payload["result_file"],
            }
            or Path(name).name != name
        ):
            raise ValueError("Invalid transaction target.")
        write_json_file(directory / name, content)
    payload["status"] = "committed"
    write_json_file(journal, payload)


def commit(directory: Path, files: dict, result_file: str):
    from .helper import write_json_file

    with section_lock(directory):
        recover(directory, locked=True)
        write_json_file(
            directory / "round_commit.json",
            {
                "status": "prepared",
                "result_file": result_file,
                "files": files,
            },
        )
        recover(directory, locked=True)
