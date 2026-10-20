"""Penny-stock day-trading bot (paper money): finds listed stocks under $5 that are 'in play' at the open,
buys breakouts, and is always flat by the close.

    python penny.py --backtest     # test the rules on the last ~60 days of 5-minute data
"""
import argparse
import math
from datetime import datetime

import numpy as np

import bot
import penny_data as PD

FIRST_BAR = 570  # 9:30am New York, in minutes after midnight

PARAMS = dict(
    slots=5,            # trade at most this many stocks a day, equal money in each
    min_rv=1.0,         # opening 5-minute volume vs. its 14-day average (1 = normal, 5 = five times normal)
    min_gap=None,       # minimum gap from yesterday's close at the open (0.10 = +10%)
    green_first=True,   # the first 5-minute candle must close above its open
    stop="first_low",   # "first_low" = under the opening candle, or a number = that many ATRs under the entry
    target_r=None,      # take profit at this many times the risk; None = hold to the close
    min_dollar_vol=1e6, # average daily dollars traded (so trades would fill)
    cost=0.0025,        # spread + slippage per side; penny stocks are expensive to trade
    side="long",        # "long" buys breakouts above the opening candle; "short" sells breakdowns below it
    first="green",      # opening candle must be "green" (closed up), "red" (closed down) or "any"
    max_gap=None,       # maximum gap (e.g. -0.10 to only take gap-downs of 10% or more)
    borrow=0.002,       # shorting cost per day for hard-to-borrow penny stocks (often far higher in real life)
)


def day_features(d5, dd):
    """Per stock, per day: the opening candle and what the bot can know at 9:35am."""
    feats = {}
    for sym, d in d5.items():
        days = PD.by_day(d)
        keys = sorted(days)
        firstv = []
        dly = dd.get(sym)
        dly_days = {datetime.fromtimestamp(t, bot.ET).date().isoformat(): i for i, t in enumerate(dly["t"])} if dly else {}
        for k, day in enumerate(keys):
            sl = days[day]
            t0 = d["t"][sl.start]
            m0 = datetime.fromtimestamp(t0, bot.ET)
            if m0.hour * 60 + m0.minute != FIRST_BAR or k == 0:
                firstv.append(np.nan)
                continue
            prev = days[keys[k - 1]]
            prev_close = d["c"][prev.stop - 1]
            o1, h1, l1, c1, v1 = (d[x][sl.start] for x in "ohlcv")
            hist = [x for x in firstv[-14:] if np.isfinite(x) and x > 0]
            firstv.append(v1)
            rv = v1 / np.mean(hist) if len(hist) >= 10 else np.nan
            atr = dvol = np.nan
            j = dly_days.get(day)
            if j is not None and j >= 15:
                hh, ll, cc = dly["h"][j - 14:j], dly["l"][j - 14:j], dly["c"][j - 15:j - 1]
                atr = float(np.mean(np.maximum(hh - ll, np.maximum(abs(hh - cc), abs(ll - cc)))))
                dvol = float(np.mean(dly["c"][j - 14:j] * dly["v"][j - 14:j]))
            feats.setdefault(day, []).append(dict(sym=sym, sl=sl, o1=o1, h1=h1, l1=l1, c1=c1, v1=v1, rv=rv,
                                                   gap=o1 / prev_close - 1, prev_close=prev_close, atr=atr, dvol=dvol))
    return feats


def candidates(rows, p):
    out = []
    for r in rows:
        if not (np.isfinite(r["rv"]) and r["rv"] >= p["min_rv"]):
            continue
        if not (1 <= r["o1"] < 5):
            continue
        if not (np.isfinite(r["dvol"]) and r["dvol"] >= p["min_dollar_vol"]):
            continue
        if p["min_gap"] is not None and r["gap"] < p["min_gap"]:
            continue
        first = p.get("first", "green" if p.get("green_first", True) else "any")
        if first == "green" and not r["c1"] > r["o1"]:
            continue
        if first == "red" and not r["c1"] < r["o1"]:
            continue
        if p.get("max_gap") is not None and r["gap"] > p["max_gap"]:
            continue
        out.append(r)
    return sorted(out, key=lambda r: -r["rv"])


def trade_one(d, r, p, alloc):
    """Long: buy a break above the opening candle's high. Short: sell a break below its low.
    Exit on the stop, the target, or at the close (never holds overnight)."""
    sl, short = r["sl"], p.get("side") == "short"
    entry = None
    for i in range(sl.start + 1, sl.stop):
        if entry is None:
            hit = d["l"][i] <= r["l1"] if short else d["h"][i] >= r["h1"]
            if hit:
                entry = min(d["o"][i], r["l1"]) if short else max(d["o"][i], r["h1"])
                if p["stop"] == "first_low":
                    stop = r["h1"] if short else r["l1"]
                else:
                    a = r["atr"] if np.isfinite(r["atr"]) else entry * 0.05
                    stop = entry + p["stop"] * a if short else entry - p["stop"] * a
                risk = abs(entry - stop)
                if risk <= 0:
                    return None
                target = (entry - p["target_r"] * risk if short else entry + p["target_r"] * risk) if p["target_r"] else None
                t_in = d["t"][i]
            continue
        if (d["h"][i] >= stop) if short else (d["l"][i] <= stop):
            exit_px, why = (max(d["o"][i], stop) if short else min(d["o"][i], stop)), "stop hit"
            break
        if target and ((d["l"][i] <= target) if short else (d["h"][i] >= target)):
            exit_px, why = (min(d["o"][i], target) if short else max(d["o"][i], target)), "profit target hit"
            break
    else:
        if entry is None:
            return None
        exit_px, why = d["c"][sl.stop - 1], "closed at 4pm (never holds overnight)"
    shares = alloc / (entry * (1 + p["cost"]))
    if short:
        pnl = shares * (entry * (1 - p["cost"]) - exit_px * (1 + p["cost"])) - alloc * p.get("borrow", 0)
    else:
        pnl = shares * (exit_px * (1 - p["cost"]) - entry * (1 + p["cost"]))
    return dict(sym=r["sym"], side="SHORT" if short else "BUY", entry=entry, exit=exit_px, pnl=pnl,
                ret=(entry / exit_px - 1) if short else (exit_px / entry - 1), why=why, t=int(t_in), rv=r["rv"], gap=r["gap"])


def backtest(d5, feats, p=PARAMS, cash=bot.START_CASH):
    eq, curve, trades = cash, [], []
    for day in sorted(feats):
        picks = candidates(feats[day], p)[: p["slots"]]
        alloc = eq / p["slots"]
        day_pnl = 0.0
        for r in picks:
            x = trade_one(d5[r["sym"]], r, p, alloc)
            if x:
                x["day"] = day
                trades.append(x)
                day_pnl += x["pnl"]
        eq += day_pnl
        curve.append((day, eq))
    return curve, trades


def summarize(curve, trades, cash=bot.START_CASH):
    if not curve:
        return {}
    v = np.array([c[1] for c in curve])
    days = len(v)
    daily = np.diff(np.concatenate([[cash], v])) / np.concatenate([[cash], v[:-1]])
    peak = np.maximum.accumulate(np.concatenate([[cash], v]))
    wins = [t for t in trades if t["pnl"] > 0]
    return dict(days=days, final=round(v[-1], 2), total=round((v[-1] / cash - 1) * 100, 1),
                per_day=round((v[-1] - cash) / days, 2),
                green_days=round(100 * np.mean(daily > 0), 0),
                worst_day=round(daily.min() * 100, 1), best_day=round(daily.max() * 100, 1),
                maxdd=round(((np.concatenate([[cash], v]) / peak) - 1).min() * 100, 1),
                trades=len(trades), win_rate=round(100 * len(wins) / len(trades), 0) if trades else None,
                days_500=int(np.sum(np.diff(np.concatenate([[cash], v])) >= 500)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    args = ap.parse_args()
    if args.backtest:
        uni = PD.penny_universe()
        syms = [u["symbol"] for u in uni]
        d5, dd = PD.load_intraday(syms, "5m", "60d"), PD.load_intraday(syms, "1d", "1y")
        feats = day_features(d5, dd)
        curve, trades = backtest(d5, feats)
        print(summarize(curve, trades))


if __name__ == "__main__":
    main()
