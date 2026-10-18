# NVDA paper-trading bot

Moving-average crossover bot with a live dashboard. Paper money only, no broker connection.

## Run

```
python bot.py
```

Open http://localhost:8765. Other stocks or crypto: `python bot.py --symbol AMD` or `--symbol BTC-USD`.

Print a 10-year backtest table without the dashboard: `python bot.py --backtest --symbol NVDA`

## Running in the cloud (GitHub Actions + Pages)

`.github/workflows/bot.yml` runs `python bot.py --tick` every 5 minutes on weekdays during market hours.
Each run catches up on finished 5-minute bars, saves the paper account to `state.json` (committed back to the repo,
so the history is in the commit log), and publishes the dashboard to GitHub Pages. Simulator and Backtest data are
pre-built once a day with `python bot.py --build-static`, because Pages can only serve files.

- Run it by hand: Actions tab, "paper-bot", "Run workflow".
- Pause it: Actions tab, "paper-bot", "...", "Disable workflow".
- On the hosted page the Simulator offers 5 pre-built markets per setting and the Backtest is NVDA only. Run locally for everything.

## What it does

- Pulls prices from Yahoo Finance every 60 seconds (free, no API key). CoinGecko only covers crypto, so stocks use Yahoo.
- Live bot trades 5-minute bars with a $10,000 paper account, saved in `state.json`. Reset it from the dashboard.
- **Plain** strategy: buy when SMA 20 crosses above SMA 50, sell when it crosses back below.
- **Swing** strategy (the live bot): buy a close above the 2-day high, sell a close below the 1-day low or on a wide trailing stop. Change it with `--strategy smart` etc.
- **Smart** strategy: same crossover plus a 200 SMA trend filter, RSI guard, ATR stop, trailing stop and volatility-based sizing.
- Backtest tab compares plain, smart and buy-and-hold on daily data.

## What the bot checks before investing

1. **Market trend**: NVDA closed above its 200-day average yesterday (Meb Faber, O'Neil's CAN SLIM). If not, it stays out or pulls out.
2. **Earnings**: no new buys on report day, and it sells before the close (date from Nasdaq).
3. **Daily loss limit**: no new trades after losing 2% in a day.
4. **Breakout**: buys only on a close above the 2-day high.
5. **Size**: invests less when NVDA is unusually volatile (Moreira & Muir).

Tested and dropped because they made it worse: skipping the first 30 minutes, buying only above VWAP,
buying only above today's open, and Minervini's trend template as an on/off switch. The full list with
sources is on the How it works tab.

## Dashboard tabs

- **Live**: the paper bot trading real 5-minute NVDA bars during market hours.
- **Simulator**: runs the live bot's own code on a simulated market, bar by bar, with its reasoning in a log. Markets are built from real NVDA 5-minute days (last 60 days) stitched in random order, with an optional bull or bear trend. It also runs 200 more simulations to show whether a run was typical. Can also replay real past data.
- **Backtest**: compares plain, smart, breakout, pullback and buy-and-hold on daily, 1-hour or 5-minute bars.
- **How it works**: the rules, plus what the tests showed about making it better.

## Simulation results (200 runs of 4 weeks each, swing strategy with all safety rules)

| Market | Bot made money | Typical result | Just holding NVDA |
|---|---|---|---|
| Bull | 72% of runs | +4.0% | +9.1% |
| Choppy | 52% | +0.4% | +0.7% |
| Bear | 34% | 0.0% | -7.0% |

Run `python sim.py` to reproduce (also prints the realism check and confirms the live code and backtest make identical trades).
The data feed was cross-checked against Nasdaq's own price history: every open, high, low and close matched.

## Results so far (10 years of NVDA, daily)

| | Return | Max drawdown | Sharpe |
|---|---|---|---|
| Plain crossover | 794% | -64% | 0.83 |
| Smart | 363% | -38% | 0.73 |
| Buy and hold | 5,045% | -66% | 1.11 |

Neither strategy beats buying and holding NVDA, because NVDA went up about 50x. The smart version
cuts the worst drop roughly in half. On 5-minute bars over the last 60 days the smart bot lost about 7%
while NVDA itself fell, so intraday crossovers mostly get chopped up. This is a learning project, not financial advice.
