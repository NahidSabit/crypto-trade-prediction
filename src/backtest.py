"""
backtest.py

Walk-forward (rolling-origin) validation harness.

The dataset is split into sequential train/test windows that slide forward
in time. For each fold:
  1. Train on a fixed-length historical window.
  2. Test only on the immediately following, unseen window.
  3. Slide both windows forward and repeat.

This prevents the model from ever being evaluated on data that precedes its
training window in time, which is the core defense against lookahead bias
in a trading context.
"""

import pandas as pd

from .model import train_model, evaluate_model


def walk_forward_validate(df: pd.DataFrame, model_params: dict, val_cfg: dict) -> pd.DataFrame:
    df = df.sort_values("open_time").reset_index(drop=True)

    train_window = pd.Timedelta(days=val_cfg["train_window_days"])
    test_window = pd.Timedelta(days=val_cfg["test_window_days"])
    step = pd.Timedelta(days=val_cfg["step_days"])

    start_time = df["open_time"].iloc[0]
    end_time = df["open_time"].iloc[-1]

    results = []
    fold = 0
    cursor = start_time + train_window

    while cursor + test_window <= end_time:
        train_mask = (df["open_time"] >= cursor - train_window) & (df["open_time"] < cursor)
        test_mask = (df["open_time"] >= cursor) & (df["open_time"] < cursor + test_window)

        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask]

        if len(train_df) < 100 or len(test_df) < 20:
            cursor += step
            continue

        model, feature_cols = train_model(train_df, model_params)
        metrics = evaluate_model(model, test_df, feature_cols)
        metrics.update({
            "fold": fold,
            "train_start": train_df["open_time"].iloc[0],
            "train_end": train_df["open_time"].iloc[-1],
            "test_start": test_df["open_time"].iloc[0],
            "test_end": test_df["open_time"].iloc[-1],
        })
        results.append(metrics)

        fold += 1
        cursor += step

    return pd.DataFrame(results)
