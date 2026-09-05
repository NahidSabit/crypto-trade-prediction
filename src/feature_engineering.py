"""
feature_engineering.py

Builds momentum, volatility, and volume features from raw minute-level OHLCV
data, and constructs the binary "good entry" label used for training.

All features are computed using only past/current information at each
timestamp (no centered windows, no forward fill from the future) to avoid
lookahead bias.
"""

import numpy as np
import pandas as pd


def add_momentum_features(df: pd.DataFrame, windows) -> pd.DataFrame:
    for w in windows:
        df[f"roc_{w}"] = df["close"].pct_change(w)
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"sma_dist_{w}"] = (df["close"] - df[f"sma_{w}"]) / df[f"sma_{w}"]

    # RSI (14-period, computed once — standard convention)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df["close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    return df


def add_volatility_features(df: pd.DataFrame, windows) -> pd.DataFrame:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    for w in windows:
        df[f"atr_{w}"] = true_range.rolling(w).mean()
        df[f"ret_std_{w}"] = df["close"].pct_change().rolling(w).std()
        rolling_mean = df["close"].rolling(w).mean()
        rolling_std = df["close"].rolling(w).std()
        df[f"bb_width_{w}"] = (2 * rolling_std) / rolling_mean

    return df


def add_volume_features(df: pd.DataFrame, windows) -> pd.DataFrame:
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"]

    for w in windows:
        vol_mean = df["volume"].rolling(w).mean()
        vol_std = df["volume"].rolling(w).std()
        df[f"vol_z_{w}"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    return df


def add_label(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    """
    Label a bar as a "good entry" (1) if the forward return over `horizon`
    minutes exceeds `threshold`. Uses only future close price for the LABEL
    (not a feature) — this is standard supervised-learning practice and does
    not leak into the features themselves.
    """
    forward_return = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = (forward_return > threshold).astype(int)
    return df


def build_feature_set(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy().sort_values("open_time").reset_index(drop=True)
    df = add_momentum_features(df, cfg["momentum_windows"])
    df = add_volatility_features(df, cfg["volatility_windows"])
    df = add_volume_features(df, cfg["volume_windows"])
    df = add_label(df, cfg["target_horizon"], cfg["target_return_threshold"])

    # Drop rows with NaNs introduced by rolling windows / forward label
    df = df.dropna().reset_index(drop=True)
    return df
