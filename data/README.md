# Data

Raw OHLCV data is *not* committed to this repository (see `.gitignore`) because it is large and easily re-downloadable.

## Source

Minute-level BTC/USDT OHLCV data pulled from the Binance API via `src/data_loader.py`.

## Expected format

| column      | description                          |
|-------------|---------------------------------------|
| open_time   | Candle open timestamp (UTC)          |
| open        | Open price                           |
| high        | High price                           |
| low         | Low price                            |
| close       | Close price                          |
| volume      | Base asset volume                    |
| close_time  | Candle close timestamp (UTC)         |
| quote_volume| Quote asset volume                   |
| num_trades  | Number of trades in the candle       |

## Fetching data

```bash
python src/data_loader.py --symbol BTCUSDT --interval 1m --start 2023-01-01 --end 2025-12-01 --out data/btcusdt_1m.csv
```

Requires Binance API credentials set as environment variables `BINANCE_API_KEY` and `BINANCE_API_SECRET` (only needed for higher rate limits; public endpoints work without keys for historical klines).
