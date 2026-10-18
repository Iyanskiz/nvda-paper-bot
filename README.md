# NVDA paper-trading bot

Moving-average crossover bot with a live dashboard. Paper money only, no broker connection.

## Run

```
python bot.py
```

Open http://localhost:8765. Other stocks or crypto: `python bot.py --symbol AMD` or `--symbol BTC-USD`.

Print a 10-year backtest table without the dashboard: `python bot.py --backtest --symbol NVDA`

## Stock scanner (whole market)

`scanner.py` is a second paper account ($10,000) that picks stocks from the entire US market:

1. Each weekday morning it pulls all ~7,000 NYSE/Nasdaq/AMEX stocks from Nasdaq and keeps ~2,100 real companies
   (market cap $2B+, price $5+).
2. It ranks them by how fast and how steadily they climbed over 90 trading days (Clenow: regression slope x R^2).
3. Filters: above the 100-day average, within 25% of the 52-week high, no 15%+ one-day jumps, enough volume.
4. It holds the top 10 equally, only buys while the S&P 500 is above its 200-day average, sells the morning after a
   holding closes below its 100-day average, and fully rebalances every 4 weeks.

Backtests (`python scanner.py --backtest`), $10,000 start, 0.05% cost per trade:

| Test | Scanner | S&P 500 | Equal-weight, same stocks |
|---|---|---|---|
| ~2,100 stocks, Oct 2022 to Oct 2026 | 28.9%/yr, worst drop -27% | 18.6%/yr, -19% | 22.2%/yr, -20% |
| 600 biggest, Oct 2017 to Oct 2026 | 32.6%/yr, worst drop -26% | 12.9%/yr, -34% | 17.8%/yr, -38% |

These use today's stock list, which leaves out companies that collapsed, so all numbers are too rosy. The fair test
is the equal-weight column, which has the same bias. Weekly rebalancing earned much less (12%/yr in the first test),
and trailing stops hurt in both tests.

## Running in the cloud (GitHub Actions + Pages)

`.github/workflows/bot.yml` runs `python bot.py --tick` and `python scanner.py --tick` every 5 minutes on weekdays during market hours. Live numbers go to the `data` branch each run (the dashboard reads them from raw.githubusercontent.com); the page is only republished when its code changes or the daily Simulator/Backtest data is rebuilt.
Each run catches up on finished 5-minute bars, saves the paper account to `state.json` (committed back to the repo,
so the history is in the commit log), and publishes the dashboard to GitHub Pages. Simulator and Backtest data are
pre-built once a day with `python bot.py --build-static`, because Pages can only serve files.

- Vercel: `vercel/vercel.json` is a mirror that proxies the GitHub Pages site, so it always shows the latest run
  and never needs redeploying. Deploy with `vercel deploy vercel --prod`.
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
