"""
main.py

End-to-end pipeline:
  1. Load raw OHLCV data
  2. Engineer features + labels
  3. Run walk-forward validation
  4. Report metrics
"""

import os
import yaml
import pandas as pd

from feature_engineering import build_feature_set
from backtest import walk_forward_validate


def load_config(path="config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    raw_path = cfg["data"]["raw_data_path"]
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. "
            f"Run src/data_loader.py first to download OHLCV data."
        )

    print("Loading raw data...")
    raw_df = pd.read_csv(raw_path, parse_dates=["open_time", "close_time"])

    print("Engineering features...")
    feature_df = build_feature_set(raw_df, cfg["features"])
    print(f"  {len(feature_df):,} rows after feature engineering.")

    print("Running walk-forward validation...")
    results = walk_forward_validate(
        feature_df,
        model_params=cfg["model"]["params"],
        val_cfg=cfg["validation"],
    )

    os.makedirs(os.path.dirname(cfg["output"]["results_path"]), exist_ok=True)
    results.to_csv(cfg["output"]["results_path"], index=False)

    print("\nWalk-forward results:")
    print(results[["fold", "test_start", "test_end", "roc_auc", "precision", "recall"]])
    print(f"\nMean ROC-AUC: {results['roc_auc'].mean():.4f}")


if __name__ == "__main__":
    main()
