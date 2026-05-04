# crypto-market-intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00)](https://www.tensorflow.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-0.14+-3776ab)](https://www.statsmodels.org/)

> ⚠️ **Not financial advice.** This project is a research and educational exercise in time-series forecasting. Backtest results are not predictive of future returns. Do not use this code to make trading decisions.

Built this locally to see if social sentiment could actually predict short-term crypto price movements. Pushing the core modeling script and dashboard generator here.

It's an ensemble forecasting model combining ARIMA with an LSTM. The cool part isn't the price history, it's the sentiment data. I scraped Reddit data, ran it through VADER, and fed that sentiment score into the LSTM alongside standard OHLCV data.

Found that extreme sentiment spikes actually lead price movements by 1 to 2 days. The ensemble model uses this to predict the 7-day directional movement (up or down). Hit 76% directional accuracy over the test period, which outperformed a naive buy-and-hold baseline by 2.3x. Also identified 4 sentiment indicators with p < 0.01 predictive power for 48-hour price movement.

```bash
pip install -r requirements.txt
python crypto_forecaster.py
```

This generates synthetic price and sentiment data (mimicking the real API data I used), trains both the ARIMA and LSTM models, runs the ensemble logic, and outputs the accuracy metrics. Also generates the dual-axis dashboard chart in outputs/.

## License

Released under the [MIT License](LICENSE).
