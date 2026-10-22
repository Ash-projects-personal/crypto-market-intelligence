import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_metrics():
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    assert m["dataset"]["candles_raw"] > 2500
    r = m["results"]
    for name in ("logistic_regression", "gradient_boosting", "ensemble_soft_vote",
                 "baseline_always_up", "baseline_momentum_ret7"):
        assert 0.0 <= r[name]["accuracy"] <= 1.0
    # the always up baseline accuracy must equal the actual up rate by construction
    assert abs(r["baseline_always_up"]["accuracy"] - r["baseline_always_up"]["actual_up_rate"]) < 1e-9


def test_predictions_rows():
    import csv
    rows = list(csv.DictReader(open(ROOT / "outputs" / "predictions.csv")))
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    assert len(rows) == m["test_rows"]
