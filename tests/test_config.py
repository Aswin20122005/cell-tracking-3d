import pytest
import yaml

from src.utils import load_config


@pytest.mark.parametrize("path", ["config/default.yaml", "config/synthetic.yaml"])
def test_shipped_configs_are_valid(path):
    assert load_config(path)["seed"] == 0


def test_missing_keys_are_reported_together(tmp_path):
    cfg = yaml.safe_load(open("config/synthetic.yaml"))
    del cfg["train"]["lr"]
    del cfg["detect"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="train.lr.*detect.peak_threshold"):
        load_config(str(path))


def test_patch_size_must_be_3d(tmp_path):
    cfg = yaml.safe_load(open("config/synthetic.yaml"))
    cfg["patch"]["size"] = [32, 32]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="patch.size"):
        load_config(str(path))
