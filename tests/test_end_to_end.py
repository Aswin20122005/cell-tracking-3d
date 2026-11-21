"""Train -> predict -> LOEO on synthetic data with a tiny model (CPU, a few seconds)."""

import copy

import pandas as pd
import pytest

from src.predict import run as predict
from src.train import train
from src.utils import load_config
from src.validation.loeo import loeo


@pytest.fixture
def cfg(tmp_path):
    c = load_config("config/synthetic.yaml")
    c["train"].update(epochs=1, patches_per_volume=4, checkpoint_dir=str(tmp_path / "ckpt"))
    c["model"].update(base_channels=4)
    return c


def test_train_then_predict_with_tta_and_tracking(cfg, tmp_path):
    ckpt = train(cfg)
    assert ckpt.endswith("detector_s0.pt")
    out = tmp_path / "pred.csv"
    predict(cfg, "unet", use_tta=True, use_ensemble=False, do_track=True, out_path=str(out))
    df = pd.read_csv(out)
    assert list(df.columns) == ["embryo", "t", "z", "y", "x"] or df.empty


def test_dog_baseline_finds_the_synthetic_cells(cfg, tmp_path, capsys):
    out = tmp_path / "dog.csv"
    predict(cfg, "dog", use_tta=False, use_ensemble=False, do_track=False, out_path=str(out))
    f1 = float(capsys.readouterr().out.split("f1=")[1].split()[0])
    assert f1 > 0.9


def test_loeo_runs_one_fold_per_embryo(cfg):
    c = copy.deepcopy(cfg)
    c["data"]["train_embryos"] = ["embryo_01"]
    c["data"]["val_embryos"] = ["embryo_02"]
    mean, std = loeo(c)
    assert 0.0 <= mean <= 1.0 and std >= 0.0
