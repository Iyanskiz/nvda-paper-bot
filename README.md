# iyanstockbot (paper money)

Built for Iyan Shivshankar.

A bot that scans the whole US stock market every day, invests $10,000 of **paper money** in the strongest steady
uptrends, waits in cash when the market is falling, and sells a stock once its trend breaks. No real money, no broker.

Dashboard: https://iyanstockbot.vercel.app (also https://nvda-paper-bot.vercel.app; mirror: https://iyanskiz.github.io/nvda-paper-bot/)

- **Live portfolio:** the real-time paper account, checked every 30 seconds during market hours, vs the same $10k in the S&P 500.
- **Simulator:** replays real past bull, choppy and bear markets (1, 3 or 6 months) day by day. The bot only sees data up to each day.
- **Backtest:** the same rules on 4 years of ~2,100 stocks and 9 years of the 600 biggest, plus the track record of its predictions.
- **How it works:** every rule, where it came from, and what testing showed.

## The rules (`scanner.py`)

1. Each weekday morning pull all ~7,000 NYSE/Nasdaq/AMEX stocks from Nasdaq; keep ~2,100 real companies ($2B+ market cap, $5+ price).
2. Rank by how fast and how steadily each climbed over 90 trading days (Clenow: regression slope x R^2).
3. Filters: above the 100-day average, within 25% of the 52-week high, no 20%+ one-day jumps, $50M+ traded a day.
4. Buy the top 7, sized by trend strength (score-weighted, max 35% in one stock), only while the S&P 500 is above its 200-day average.
5. Sell the morning after a holding closes below its 100-day average, and don't buy it back for 90 days. Full rebalance every few weeks.
6. Decide after the close, trade at the next open, 0.05% cost per trade. Checks every 30 seconds during market hours.
   **Safety net:** sell right away if a holding falls 25% under its best close since buying (then a 90-day ban).
7. **Smart borrowing:** while the S&P 500 trend is up, 1.5x the account when the market is calm (S&P 500 20-day
   volatility under 15%), 1.25x when normal, 1.0x when wild (over 25%) or in a downtrend; 6%/yr interest on what's borrowed.
   Mean over 20 timings vs flat 1.25x: 70% to 81%/yr (all stocks), 51% to 58% (600 biggest); worst drops about 5 points deeper.
7b. **Swap fading stocks daily:** between rebalances, a holding that drops out of the top 21 is sold at the open and its
   money goes to the strongest stock not held (all stocks 81% to 99%/yr, worst drop -44% to -38%; 600 biggest 57% to 59%,
   -48% to -44%). Idle cash waits in a T-bill ETF (BIL), +0.5-0.8 points a year. Both came from a research workflow of six
   technique families with independent audit and robustness checks (bonds/gold rotation, breadth, residual momentum,
   volatility targeting, ATR stops and pullback entries were rejected).
8. **News:** `news.py` reads Yahoo/Nasdaq/Google headlines and SEC EDGAR filings for holdings and the next candidates.
   Serious red flags block buys and sell holdings; warnings block buys and sell only if the price confirms.
9. **Live rescan:** all ~2,100 stocks (440+ of them foreign companies listed in the US) re-priced every 30 seconds
   (Yahoo spark, ~4 s); it reports, the tested daily rules decide.
10. **Monthly review (report only):** replays the past 12 months with 27 setting combos. It used to switch to the
   winner, but averaged over 4 rebalance timings that earned 30-31%/yr vs 49% for fixed rules (all stocks) and 37-39% vs
   39% (600 biggest), so it now only reports. It also keeps a journal of its own trades.

### Tested and rejected (mean over 20 rebalance timings, no leverage)

| Idea | All stocks, 4y | 600 biggest, 9y |
|---|---|---|
| Current rules (7 stocks, score-weighted, 25% safety net) | 29.8%/yr | 39.2%/yr |
| Same with equal amounts, no safety net | 21.7% | 33.4% |
| Tight stops: 10% / 15% / 20% under the high | 3.1% / 8.5% / 18.1% | 17.7% / 22.1% / 28.4% |
| Sell on a 5% / 8% one-day drop | 3.4% / 13.7% | 12.3% / 25.9% |
| Hold 15 / 20 stocks | 15.5% / 13.4% | 16.1% / 13.7% |

## Stock search

The dashboard's search finds any stock worldwide (`vercel/api/stock.js` proxies Yahoo Finance for quotes, charts and
headlines) and shows the bot's view of it from `universe.json`, which the bot publishes to the `data` branch at each scan.
Deploy the function with `vercel deploy --prod` from the `vercel/` folder.

## Brokerage link (optional, `broker.py`)

Copies the bot's portfolio into an Alpaca account in the same proportions on every check. Add repository secrets
`ALPACA_KEY_ID` and `ALPACA_SECRET_KEY` (Alpaca paper keys to start). Variables: `ALPACA_MODE=live` for real money
(account holder 18+), `ALPACA_MAX_EXPOSURE` (default 1.0, no borrowing). Money moves bank -> Alpaca inside Alpaca's app;
the bot never handles bank details.

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

`.github/workflows/trader.yml` runs `.github/scripts/trade_loop.sh`: one job at a time runs `python scanner.py --tick`
every 30 seconds during market hours, idles otherwise, and starts its own next run before GitHub's 6-hour cap (the
cron schedule is only a backup, since GitHub delays and skips scheduled runs). It saves `scanner_state.json` to `main`
and force-pushes the live numbers to the `data` branch (the dashboard reads them through the GitHub API). The page is republished only when its code changes or once a day with fresh
Backtest/Simulator data. `vercel/vercel.json` is a Vercel mirror that proxies the GitHub Pages site.

`bot.py` holds the shared data helpers (Yahoo prices, indicators, times) and the retired NVDA bot code.
