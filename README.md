# Stock scanner bot (paper money)

A bot that scans the whole US stock market every day, invests $10,000 of **paper money** in the strongest steady
uptrends, waits in cash when the market is falling, and sells a stock once its trend breaks. No real money, no broker.

Dashboard: https://nvda-paper-bot.vercel.app (mirror: https://iyanskiz.github.io/nvda-paper-bot/)

- **Live portfolio:** the real-time paper account, checked every 5 minutes during market hours, vs the same $10k in the S&P 500.
- **Simulator:** replays real past bull, choppy and bear markets (1, 3 or 6 months) day by day. The bot only sees data up to each day.
- **Backtest:** the same rules on 4 years of ~2,100 stocks and 9 years of the 600 biggest, plus the track record of its predictions.
- **How it works:** every rule, where it came from, and what testing showed.

## The rules (`scanner.py`)

1. Each weekday morning pull all ~7,000 NYSE/Nasdaq/AMEX stocks from Nasdaq; keep ~2,100 real companies ($2B+ market cap, $5+ price).
2. Rank by how fast and how steadily each climbed over 90 trading days (Clenow: regression slope x R^2).
3. Filters: above the 100-day average, within 25% of the 52-week high, no 15%+ one-day jumps, enough volume.
4. Buy the top few (5-10, chosen by its monthly learning) in equal amounts, only while the S&P 500 is above its 200-day average.
5. Sell the morning after a holding closes below its 100-day average, and don't buy it back for 90 days. Full rebalance every few weeks.
6. Decide after the close, trade at the next open, 0.05% cost per trade. Checks every minute during market hours.
7. **Learns monthly:** replays the past 12 months with 27 setting combos (5/7/10 stocks, 60/90/120-day trends,
   rebalance every 3/4/6 weeks) and switches to the best. Walk-forward tested: 50.5%/yr vs 41.3% fixed (9 years),
   33.9% vs 32.8% (4 years); learning from 6 or 24 months was unreliable. It also keeps a journal of its own trades.

## Penny-stock bot (standby, not running)

`penny.py` / `penny_data.py` test day-trading strategies on every listed stock under $5. On 60 days of 5-minute data,
every strategy lost money or roughly broke even after costs (opening-range breakout -36% to -50%, Gap and Go -15%,
short fades -8% to -20% with borrow costs). Penny gap-ups closed below their open 65-72% of the time. Parked until a
strategy survives testing.

## Results ($10,000 start)

| Test | Bot | S&P 500 | Equal amounts in every stock |
|---|---|---|---|
| ~2,100 stocks, Oct 2022 to Oct 2026 | 35.8%/yr, worst drop -30% | 18.6%/yr, -19% | 22.2%/yr, -20% |
| 600 biggest, Oct 2017 to Oct 2026 | 41.2%/yr, worst drop -26% | 12.9%/yr, -34% | 17.8%/yr, -38% |

Its top 7 picks rose over the next month 54-57% of the time, about as often as an average stock, but gained
+2.8% to +3.4% on average vs +1.6% to +1.8%. The edge is bigger winners, not better guessing.

These tests use today's list of stocks, which leaves out companies that collapsed, so they are too rosy. Compare
against the equal-weight column, which has the same bias.

## Running it

```
pip install numpy pandas
python scanner.py --scan            # today's ranking
python scanner.py --backtest        # 5-year backtest
python scanner.py --build-static    # backtest + simulator data for the dashboard (docs/static)
python scanner.py --tick            # one live pass, writes docs/
python serve_docs.py                # preview the dashboard at http://[::1]:8800/
```

## Cloud

`.github/workflows/bot.yml` runs `python scanner.py --tick` every 5 minutes on weekdays during market hours, saves
`scanner_state.json` to `main`, and pushes the live numbers to the `data` branch (the dashboard reads them from
raw.githubusercontent.com). The page is republished only when its code changes or once a day with fresh
Backtest/Simulator data. `vercel/vercel.json` is a Vercel mirror that proxies the GitHub Pages site.

`bot.py` holds the shared data helpers (Yahoo prices, indicators, times) and the retired NVDA bot code.
