import pytest

from mri_rl.frozen_eval import adapter_metadata, sha256


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
