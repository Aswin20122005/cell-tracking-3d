import pandas as pd
import pytest

from src.evaluate import run


def write(tmp_path, name, rows, cols):
    path = tmp_path / name
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return str(path)


def test_perfect_predictions_score_one(tmp_path):
    gt = write(tmp_path, "gt.csv", [(0, 1, 2, 3), (1, 4, 5, 6)], ["t", "z", "y", "x"])
    pred = write(tmp_path, "p.csv", [("e1", 0, 1, 2, 3), ("e1", 1, 4, 5, 6)], ["embryo", "t", "z", "y", "x"])
    assert run(pred, gt, tol=1.0)["f1"] == 1.0


def test_empty_predictions_give_zero_recall(tmp_path):
    gt = write(tmp_path, "gt.csv", [(0, 1, 2, 3)], ["t", "z", "y", "x"])
    pred = write(tmp_path, "p.csv", [], ["embryo", "t", "z", "y", "x"])
    res = run(pred, gt, tol=1.0)
    assert res["recall"] == 0.0 and res["fn"] == 1


def test_multiple_embryos_require_choosing_one(tmp_path):
    gt = write(tmp_path, "gt.csv", [(0, 1, 2, 3)], ["t", "z", "y", "x"])
    pred = write(tmp_path, "p.csv", [("e1", 0, 1, 2, 3), ("e2", 0, 9, 9, 9)], ["embryo", "t", "z", "y", "x"])
    with pytest.raises(SystemExit, match="--embryo"):
        run(pred, gt, tol=1.0)
    assert run(pred, gt, tol=1.0, embryo="e1")["f1"] == 1.0
