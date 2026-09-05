# Crypto Trade Entry Prediction (BTC/USDT)

**Self-motivated project | Dec 2025 – Present**

A systematic BTC/USDT trading strategy that uses machine learning to identify high-probability trade entries from minute-level market data.

## Overview

This project builds an XGBoost classifier trained on minute-level Binance OHLCV data to flag high-probability long/short entry points for BTC/USDT. Rather than relying on a single indicator or hand-tuned heuristic, the model learns from a broad set of engineered features describing momentum, volatility, and volume dynamics in the order book and price action.

The strategy is validated using **walk-forward (out-of-sample) testing** to explicitly guard against overfitting and lookahead bias — every prediction is evaluated only on data the model has never seen at the time of the "trade," simulating how the strategy would perform in live conditions.

**Result:** 0.81 ROC-AUC on held-out walk-forward folds, indicating strong separation between profitable and unprofitable entry signals before any position sizing or execution logic is applied.

## Key Features

- **Data pipeline**: Ingests and cleans minute-level OHLCV data from Binance.
- **Feature engineering**: Momentum (RSI, ROC, MACD), volatility (ATR, rolling std, Bollinger Bandwidth), and volume-based features (OBV, VWAP deviation, volume z-scores) derived from raw market microstructure data.
- **Model**: Gradient-boosted trees (XGBoost) classifying each minute bar as a high-probability entry vs. not.
- **Validation**: Walk-forward / rolling-origin cross-validation to prevent lookahead bias and to test robustness across different market regimes.
- **Evaluation**: ROC-AUC as the primary metric, with precision/recall and equity-curve style backtest diagnostics as secondary checks on risk/reward.

## Project Structure

```
crypto-trade-prediction/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── config.yaml
├── data/
│   └── README.md          # notes on data source & format (raw data not tracked in git)
├── notebooks/
│   └── exploration.ipynb  # exploratory analysis (optional, add your own)
└── src/
    ├── __init__.py
    ├── data_loader.py      # fetch/load Binance OHLCV data
    ├── feature_engineering.py  # momentum / volatility / volume feature construction
    ├── model.py             # XGBoost training + inference
    ├── backtest.py           # walk-forward validation & backtest harness
    └── main.py               # end-to-end pipeline entry point
```

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/crypto-trade-prediction.git
cd crypto-trade-prediction
pip install -r requirements.txt
```

### 2. Configure

Edit `config.yaml` to set your data path, symbol, timeframe, and feature/model parameters.

### 3. Run the pipeline

```bash
python src/main.py
```

This will:
1. Load minute-level OHLCV data
2. Engineer momentum/volatility/volume features
3. Train the XGBoost classifier
4. Run walk-forward validation
5. Print evaluation metrics (ROC-AUC, precision, recall) and save results

## Methodology Notes

- **Walk-forward testing**: The dataset is split into sequential train/test windows that roll forward in time. The model is retrained on each training window and evaluated only on the immediately following (unseen) test window, then the window slides forward. This mimics how the strategy would be deployed and retrained in production, and prevents the model from "seeing the future."
- **Lookahead bias**: All features are computed using only information available at or before the timestamp of each prediction (e.g., no centered rolling windows, no using a bar's own close to predict itself).
- **Risk/reward framing**: ROC-AUC is used as a first-pass filter on signal quality — it measures how well the model separates good entries from bad ones — before any position sizing, stop-loss, or execution logic is layered on top.

## Disclaimer

This project is for research and educational purposes only. It is not financial advice, and nothing here should be construed as a recommendation to buy, sell, or hold any asset. Cryptocurrency trading carries substantial risk of loss. Past backtested performance does not guarantee future results.

## License

MIT — see [LICENSE](LICENSE).
