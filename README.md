# Crypto market intelligence

Not financial advice. This is a research exercise in time series
forecasting. Backtest results do not predict future returns.

I rebuilt this project from scratch. The old version ran on synthetic
data, planted the effect it claimed to find in that data, had an
ensemble step that returned one model's output on both branches so the
second model never counted, and reported a directional accuracy that
its own committed metrics file contradicted. All of that is gone. This
version uses real exchange data, a real ensemble, real baselines, and
it reports a result I would rather hide: the models do not beat the
dumbest possible baseline.

## Dataset

BTC-USD daily candles from the Coinbase Exchange public API, saved as
a snapshot in `data/btc_usd_daily.csv` so the run is reproducible.
2,852 daily candles, 1 Jan 2019 to 22 Oct 2026, fetched 22 Oct 2026.
`src/download_data.py` is the exact code that fetched it.

There is no sentiment data in this version. The old version claimed a
Reddit and VADER pipeline that was never in the code. I removed the
claim instead of rebuilding that pipeline, because I have no real
sentiment dataset here and I will not plant one again.

## Task and method

Predict one thing: whether the BTC close 7 days from now is higher
than today's close. Features use only the past: returns over 1, 7, 14
and 30 days, 14 and 30 day volatility, price against its 20 and 50 day
moving averages, 7 day volume change, and a 14 day RSI.

Split is chronological and fixed before fitting: first 70% of usable
rows train (1,957 rows), last 30% test (839 rows, from 29 Jun 2024).
Models: logistic regression and gradient boosting. The ensemble
averages their predicted probabilities, so both models genuinely feed
the final call. Baselines: always predict up, and a momentum rule that
predicts the sign of the last 7 day return.

## Results

From `outputs/metrics.json`. Test period actual up rate: 52.8%.

| Model | Accuracy | Balanced accuracy |
|---|---|---|
| Logistic regression | 48.9% | 48.2% |
| Gradient boosting | 49.7% | 49.9% |
| Ensemble (soft vote) | 51.8% | 51.8% |
| Baseline: always up | 52.8% | 50.0% |
| Baseline: momentum | 48.5% | 48.3% |

The ensemble is the best of my three models and it still loses to
always predicting up. A binomial test of the ensemble against a coin
flip gives p = 0.15, so even its small edge over 50% is not
statistically meaningful on this test set. Plain conclusion: with
these features, on this period, I found no real directional edge.
Price and volume history alone, at a 7 day horizon, did not carry a
usable signal in this run.

Per day predictions are in `outputs/predictions.csv`.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py
python src/run_analysis.py
python -m pytest tests/
```

## Files

- `src/download_data.py` pages the Coinbase API and saves the snapshot.
- `src/run_analysis.py` builds features, trains, evaluates, writes
  `outputs/metrics.json`, `outputs/predictions.csv` and a price chart
  with the split marked.
- `EXPLAIN.md` explains the method and why the negative result matters.
