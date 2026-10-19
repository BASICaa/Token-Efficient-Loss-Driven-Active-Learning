import json

import pytest

from TEFLD.src import validation


@pytest.fixture
def local_validation(tmp_path, monkeypatch):
    target = tmp_path / "frozen.json"
    monkeypatch.setattr(
        validation, "get_shared_validation_path", lambda section: target
    )
    monkeypatch.setattr(
        validation,
        "generate_shared_validation_samples",
        lambda **kwargs: pytest.fail("Unexpected generation"),
    )
    return target


def write_source(path, answer="4"):
    path.write_text(
        json.dumps(
            [
                {
                    "sample": "",
                    "instruct": "What is two plus two?",
                    "gold_summary": answer,
                    "tag": "math",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_imported_local_validation_is_frozen_and_never_generated(
    tmp_path, local_validation
):
    source = tmp_path / "source.json"
    write_source(source)
    samples = validation.ensure_shared_validation_samples(
        section_id="fixture", model_id="unused", validation_file=source
    )
    assert len(samples) == 1
    saved = json.loads(local_validation.read_text(encoding="utf-8"))
    assert saved["metadata"]["source"] == "user_provided_validation"
    assert saved["metadata"]["source_sha256"]
    assert (
        validation.ensure_shared_validation_samples(
            section_id="fixture", model_id="unused"
        )
        == samples
    )
    assert (
        validation.ensure_shared_validation_samples(
            section_id="fixture", model_id="unused", validation_file=source
        )
        == samples
    )
    before = local_validation.read_bytes()
    write_source(source, answer="5")
    with pytest.raises(ValueError, match="different frozen validation"):
        validation.ensure_shared_validation_samples(
            section_id="fixture", model_id="unused", validation_file=source
        )
    assert local_validation.read_bytes() == before


def test_import_does_not_replace_a_generated_validation_set(tmp_path, local_validation):
    source = tmp_path / "source.json"
    write_source(source)
    samples = validation.import_shared_validation_file("fixture", source)
    samples[0].gold_summary = "A different original target"
    validation.save_shared_validation_samples(
        "fixture", samples, metadata={"source": "generated_fixed_shared_validation"}
    )
    with pytest.raises(ValueError, match="different frozen validation"):
        validation.import_shared_validation_file("fixture", source)


def test_import_rejects_missing_reference_before_saving(tmp_path, local_validation):
    source = tmp_path / "source.json"
    write_source(source, answer="")
    with pytest.raises(ValueError, match="nonempty"):
        validation.import_shared_validation_file("fixture", source)
    assert not local_validation.exists()
