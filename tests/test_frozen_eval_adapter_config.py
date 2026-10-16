import pytest

import json

from mri_rl.frozen_eval import adapter_metadata, prepare_resume, sha256


def test_adapter_checkpoint_requires_config_and_records_hashes(tmp_path):
    checkpoint = tmp_path / "after_adapters.safetensors"
    checkpoint.write_bytes(b"fixture weights")
    with pytest.raises(FileNotFoundError, match="config"):
        adapter_metadata(checkpoint)
    config = tmp_path / "adapter_config.json"
    config.write_text('{"fine_tune_type":"lora"}')
    metadata = adapter_metadata(checkpoint)
    assert metadata == {
        "adapter_checkpoint_sha256": sha256(checkpoint),
        "adapter_config_sha256": sha256(config),
    }
    assert adapter_metadata(None)["adapter_checkpoint_sha256"] is None


def test_resume_skips_only_matching_complete_rows_and_repairs_torn_tail(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"real-image-fixture")
    records = [{"record_id": "a", "prompt": "question a", "image": str(image)},
               {"record_id": "b", "prompt": "question b", "image": str(image)}]
    expected = {"model_path": "/pinned/model", "max_tokens": 256,
                "manifest_sha256": "pinned-manifest"}
    good = {**expected, "record_id": "a", "prompt": "question a",
            "image_sha256": sha256(image), "response": "answer"}
    output = tmp_path / "predictions.jsonl"
    output.write_bytes((json.dumps(good) + "\n" + '{"record_id":"b"').encode())
    assert prepare_resume(output, records, expected) == {"a"}
    assert [json.loads(line)["record_id"] for line in output.read_text().splitlines()] == ["a"]
    assert output.with_name(output.name + ".partial").exists()


def test_resume_rejects_changed_decoding_or_image(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    records = [{"record_id": "a", "prompt": "question", "image": str(image)}]
    expected = {"max_tokens": 256}
    output = tmp_path / "predictions.jsonl"
    output.write_text(json.dumps({"record_id": "a", "prompt": "question",
                                  "image_sha256": sha256(image), "response": "answer",
                                  "max_tokens": 128}) + "\n")
    with pytest.raises(ValueError, match="configuration mismatch"):
        prepare_resume(output, records, expected)
    expected["max_tokens"] = 128
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="image checksum mismatch"):
        prepare_resume(output, records, expected)
