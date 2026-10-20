"""Momentum stock scanner: ranks the whole US market every day, holds the strongest trends, sells when they break.

    python scanner.py --backtest            # test the rules on 5 years of history
    python scanner.py --scan                # today's ranking, printed

Based on Clenow ("Stocks on the Move"), Jegadeesh & Titman (momentum), George & Hwang (52-week high) and
Da, Gurun & Warachka (smooth trends). Paper trading only.
"""
import argparse
import json
import math
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

import bot

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
FEE = bot.FEE

PARAMS = dict(
    n=7,                 # how many stocks to hold: 7 beat 10 in both backtests (5 earned more but swung harder)
    window=90,           # days in the trend regression
    ma=100,              # must close above this moving average to buy or keep
    max_gap=0.15,        # skip stocks that moved more than 15% in one day lately (news jumps, not trends)
    near_high=0.75,      # must be within 25% of the 52-week high
    min_dollar_vol=20e6, # average daily dollar volume, so trades would really fill
    regime_ma=200,       # new buys only while SPY is above its 200-day average
    exit_rank=2,         # sell if it falls out of the top n * exit_rank at a rebalance
    rebalance_day=2,     # 0=Mon ... 2=Wed (Clenow trades on Wednesdays)
    every_weeks=4,       # full rebalance every 4th week: weekly churned and earned far less in testing
    trail_atr=None,      # optional trailing stop, k x ATR(20) under the highest close
    cooldown_days=90,    # after selling on a broken trend, move on: no rebuy for 90 days (free in tests; "forever" cost 2-4%/yr)
)


# ---------------------------------------------------------------- data

def universe(min_cap=2e9, min_price=5.0):
    """Every NYSE/Nasdaq/AMEX stock from Nasdaq's screener, filtered to real companies."""
    req = urllib.request.Request("https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true",
                                 headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        rows = json.load(r)["data"]["rows"]
    num = lambda s: float(str(s).replace("$", "").replace(",", "") or 0)
    out = []
    for r in rows:
        sym = r["symbol"].strip()
        if sym.isalpha() and num(r["marketCap"]) >= min_cap and num(r["lastsale"]) >= min_price:
            out.append(dict(symbol=sym, name=_short_name(r["name"]), sector=r["sector"] or "", cap=num(r["marketCap"])))
    return sorted(out, key=lambda x: -x["cap"])


def _short_name(name):
    """'Marathon Petroleum Corporation Common Stock' -> 'Marathon Petroleum'."""
    import re
    name = re.split(r"\s+(?:common stock|ordinary shares|american depositary|capital stock|class [a-z]\b)", name, flags=re.I)[0]
    name = re.sub(r"\s*\((?:DE|The)\)|,?\s+(?:Inc\.?|Corporation|Corp\.?|Ltd\.?|plc|N\.V\.|S\.A\.)$", "", name, flags=re.I)
    return name.strip(" ,")[:40]


def load_panel(symbols, rng="2y", cache_tag=None):
    """Daily bars for many symbols, aligned to SPY's trading days. Returns dict of DataFrames O/H/L/C/V."""
    tag = cache_tag or datetime.now(bot.ET).date().isoformat()
    path = os.path.join(CACHE, f"panel_{rng}_{len(symbols)}_{tag}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)

    def get(sym):
        for attempt in range(3):
            try:
                bars, _ = bot.fetch_bars(sym, "1d", rng)
                return sym, bars
            except Exception:
                time.sleep(1 + attempt)
        return sym, None

    with ThreadPoolExecutor(12) as ex:
        got = dict(ex.map(get, ["SPY"] + list(symbols)))
    day = lambda t: datetime.fromtimestamp(t, bot.ET).date().isoformat()
    dates = [day(b["t"]) for b in got["SPY"]]
    frames = {}
    for k in "ohlcv":
        cols = {}
        for sym, bars in got.items():
            if bars:
                cols[sym] = pd.Series({day(b["t"]): b[k] for b in bars})
        frames[k.upper()] = pd.DataFrame(cols).reindex(dates).astype(float)
    os.makedirs(CACHE, exist_ok=True)
    pd.to_pickle(frames, path)
    return frames


# ---------------------------------------------------------------- indicators (all stocks at once)

def indicators(P, p=PARAMS):
    C, O, H, L, V = P["C"], P["O"], P["H"], P["L"], P["V"]
    w = p["window"]
    y = np.log(C)
    t = pd.Series(np.arange(len(C), dtype=float), index=C.index)
    Sy = y.rolling(w).sum()
    Sty = y.mul(t, axis=0).rolling(w).sum()
    Syy = (y * y).rolling(w).sum()
    start = t - (w - 1)
    Sxy = Sty - Sy.mul(start, axis=0)            # sum of k * y over the window, k = 0..w-1
    xbar, Sxx = (w - 1) / 2, w * (w * w - 1) / 12
    slope = (Sxy - xbar * Sy) / Sxx               # log-price change per day
    var = Syy - Sy * Sy / w
    r2 = (slope * slope * Sxx / var).clip(0, 1)
    trend = np.exp(slope * 252) - 1               # annualized pace of the trend
    prev = C.shift(1)
    tr = pd.concat([(H - L), (H - prev).abs(), (L - prev).abs()]).groupby(level=0, sort=False).max().reindex(C.index)
    jump = pd.concat([(C / prev - 1).abs(), (O / prev - 1).abs()]).groupby(level=0, sort=False).max().reindex(C.index)
    spy = C["SPY"]
    return dict(
        score=trend * r2, trend=trend, r2=r2,
        mom12=C.shift(21) / C.shift(252) - 1,      # classic 12-1 month momentum (skip the last month)
        mom6=C / C.shift(126) - 1,
        high52=C / C.rolling(252, min_periods=200).max(),
        ma=C.rolling(p["ma"]).mean(),
        hi=C.rolling(252, min_periods=200).max(),
        gap=jump.rolling(w).max(),
        atr=tr.rolling(20).mean(),
        dvol=(C * V).rolling(20).mean(),
        regime=(spy > spy.rolling(p["regime_ma"]).mean()) if p["regime_ma"] else pd.Series(True, index=spy.index),
        spy_ma=spy.rolling(p["regime_ma"] or 200).mean(),
    )


def ranking(P, ind, t, p=PARAMS, exclude=("SPY",)):
    """Stocks passing every filter on day index t, best first, with the numbers behind each pick."""
    C = P["C"].iloc[t]
    ok = (ind["score"].iloc[t] > 0) & (C > ind["ma"].iloc[t]) & (ind["gap"].iloc[t] < p["max_gap"]) \
        & (C >= p["near_high"] * ind["hi"].iloc[t]) & (ind["dvol"].iloc[t] >= p["min_dollar_vol"])
    ok = ok.drop(list(exclude), errors="ignore")
    scores = ind[p.get("rank_by", "score")].iloc[t][ok[ok].index].dropna().sort_values(ascending=False)
    return list(scores.index)


def why_filtered(P, ind, t, sym, p=PARAMS):
    """Plain-language reason a stock didn't make the list."""
    c = P["C"][sym].iloc[t]
    if not np.isfinite(c):
        return "no price data"
    if not ind["score"][sym].iloc[t] > 0:
        return "not in an uptrend"
    if not c > ind["ma"][sym].iloc[t]:
        return f"below its {p['ma']}-day average"
    if not ind["gap"][sym].iloc[t] < p["max_gap"]:
        return f"jumped more than {p['max_gap']:.0%} in one day recently"
    if not c >= p["near_high"] * ind["hi"][sym].iloc[t]:
        return "too far below its 52-week high"
    if not ind["dvol"][sym].iloc[t] >= p["min_dollar_vol"]:
        return "too thinly traded"
    return None


# ---------------------------------------------------------------- portfolio engine (shared by backtest and live)

def new_book(cash=bot.START_CASH):
    return dict(cash=cash, pos={}, trades=[], equity=[], pending=[])


def equity_of(book, prices):
    return book["cash"] + sum(h["shares"] * prices.get(s, h["last"]) for s, h in book["pos"].items())


def decide(book, P, ind, t, p=PARAMS, force_rebalance=False):
    """At the close of day t: queue sells and buys for the next open. Returns the ranking used."""
    C = P["C"].iloc[t]
    date = P["C"].index[t]
    orders = []
    for s, h in book["pos"].items():
        c = C.get(s)
        if c is not None and np.isfinite(c) and c < ind["ma"][s].iloc[t]:
            orders.append(dict(side="SELL", sym=s, reason=f"closed below its {p['ma']}-day average"))
    ranks = ranking(P, ind, t, p)
    d = datetime.fromisoformat(date)
    is_rebalance = force_rebalance or (d.weekday() == p["rebalance_day"] and
                                       (p.get("every_weeks", 1) == 1 or d.isocalendar()[1] % p["every_weeks"] == 0))
    if is_rebalance:
        keep = set(ranks[: p["n"] * p["exit_rank"]])
        for s in book["pos"]:
            if s not in keep and not any(o["sym"] == s for o in orders):
                orders.append(dict(side="SELL", sym=s, reason=f"fell out of the top {p['n'] * p['exit_rank']}"))
        if bool(ind["regime"].iloc[t]):
            selling = {o["sym"] for o in orders}
            slots = p["n"] - (len(book["pos"]) - len(selling))
            for s in ranks:
                if slots <= 0:
                    break
                if s in book["pos"] or s in selling or book.get("banned", {}).get(s, "") > date:
                    continue
                rank = ranks.index(s) + 1
                gain = P["C"][s].iloc[t] / P["C"][s].iloc[t - p["window"] + 1] - 1
                orders.append(dict(side="BUY", sym=s, rank=rank,
                                   reason=f"#{rank} strongest steady uptrend: {gain:+.0%} in {p['window']} trading days, "
                                          f"smoothness {ind['r2'][s].iloc[t]:.2f} out of 1"))
                slots -= 1
    book["pending"] = orders
    return ranks


def execute(book, prices, date, equity_now, p=PARAMS, atr=None):
    """Fill pending orders at the given prices (the next open, or the live price). Sells first, then buys."""
    for o in [o for o in book["pending"] if o["side"] == "SELL"]:
        h = book["pos"].get(o["sym"])
        px = prices.get(o["sym"])
        if h is None or px is None or not np.isfinite(px):
            continue
        _sell(book, o["sym"], px, date, o["reason"], p)
    slot = equity_now / p["n"]
    for o in [o for o in book["pending"] if o["side"] == "BUY"]:
        px = prices.get(o["sym"])
        if px is None or not np.isfinite(px) or o["sym"] in book["pos"]:
            continue
        alloc = min(book["cash"], slot)
        if alloc < 0.25 * slot:
            break
        sh = alloc / (px * (1 + FEE))
        book["cash"] -= sh * px * (1 + FEE)
        a = atr.get(o["sym"]) if atr else None
        stop = px - p["trail_atr"] * a if p["trail_atr"] and a else None
        book["pos"][o["sym"]] = dict(shares=sh, entry=px, date=date, high=px, stop=stop, last=px, reason=o["reason"])
        book["trades"].append(dict(date=date, side="BUY", sym=o["sym"], price=round(px, 4), shares=round(sh, 4),
                                   reason=o["reason"]))
    book["pending"] = []


def _sell(book, s, px, date, reason, p=PARAMS):
    h = book["pos"].pop(s)
    cd = p.get("cooldown_days")
    if cd and "below" in reason:  # the trend broke: move on, don't chase it back
        until = "9999-12-31" if cd == "forever" else (datetime.fromisoformat(date) + pd.Timedelta(days=cd)).date().isoformat()
        book.setdefault("banned", {})[s] = until
    book["cash"] += h["shares"] * px * (1 - FEE)
    pnl = h["shares"] * (px * (1 - FEE) - h["entry"] * (1 + FEE))
    book["trades"].append(dict(date=date, side="SELL", sym=s, price=round(px, 4), shares=round(h["shares"], 4),
                               pnl=round(pnl, 2), ret=round(px / h["entry"] - 1, 4), held_since=h["date"], reason=reason))


def check_stops(book, lows, opens, date):
    """Trailing stops as resting orders: fill at the stop, or the open if price gapped under it."""
    for s, h in list(book["pos"].items()):
        lo = lows.get(s)
        if h.get("stop") and lo is not None and np.isfinite(lo) and lo <= h["stop"]:
            op = opens.get(s, h["stop"])
            _sell(book, s, min(op, h["stop"]) if np.isfinite(op) else h["stop"], date, "trailing stop hit")


def mark(book, closes, atr, p=PARAMS):
    for s, h in book["pos"].items():
        c = closes.get(s)
        if c is not None and np.isfinite(c):
            h["last"] = c
            h["high"] = max(h["high"], c)
            a = atr.get(s)
            if p["trail_atr"] and a and np.isfinite(a):
                h["stop"] = max(h["stop"] or 0, h["high"] - p["trail_atr"] * a)


# ---------------------------------------------------------------- backtest

def backtest(P, ind, p=PARAMS, start=260):
    book = new_book()
    dates = P["C"].index
    curve = []
    for t in range(start, len(dates)):
        row = lambda k: P[k].iloc[t].to_dict()
        opens, lows, closes = row("O"), row("L"), row("C")
        eq_prev = curve[-1][1] if curve else book["cash"]
        if book["pending"]:
            execute(book, opens, dates[t], eq_prev, p, ind["atr"].iloc[t - 1].to_dict())
        check_stops(book, lows, opens, dates[t])
        mark(book, closes, ind["atr"].iloc[t].to_dict(), p)
        curve.append((dates[t], equity_of(book, closes)))
        if t < len(dates) - 1:
            decide(book, P, ind, t, p, force_rebalance=(t == start))
    return book, curve


PRED_BUCKETS = [("Top 7 (what it buys)", 1, 7), ("Ranked 8-25", 8, 25), ("Ranked 26-100", 26, 100),
                ("Passed filters, ranked lower", 101, 10**6)]


def prediction_stats(P, ind, p=PARAMS, start=260, step=5, horizon=21):
    """How often stocks at each rank really rose over the next month, vs an average stock.
    This is the honest track record behind the bot's 'predictions'."""
    C = P["C"]
    fwd = (C.shift(-horizon) / C - 1)
    out = {name: [] for name, _, _ in PRED_BUCKETS}
    everyone = []
    for t in range(start, len(C) - horizon, step):
        ranks = ranking(P, ind, t, p)
        f = fwd.iloc[t]
        everyone.extend(f.drop("SPY").dropna().clip(-0.9, 3).tolist())
        for name, lo, hi in PRED_BUCKETS:
            out[name].extend(f[ranks[lo - 1:hi]].dropna().clip(-0.9, 3).tolist())
    summary = lambda xs: dict(n=len(xs), up=round(float(np.mean([x > 0 for x in xs])) * 100, 1),
                              avg=round(float(np.mean(xs)) * 100, 2), median=round(float(np.median(xs)) * 100, 2))
    res = {name: summary(xs) for name, xs in out.items() if xs}
    res["Average stock"] = summary(everyone)
    return dict(horizon_days=horizon, buckets=res)


# ---------------------------------------------------------------- simulator: replay real past markets

MARKETS = {  # by the S&P 500's average monthly move over the window
    "bull": "S&P 500 rose more than 1.5% a month",
    "choppy": "S&P 500 moved less than 1.5% a month either way",
    "bear": "S&P 500 fell more than 1.5% a month",
}
SIM_MONTHS = (1, 3, 6)


def classify(P, t0, days):
    spy = P["C"]["SPY"]
    monthly = (spy.iloc[t0 + days] / spy.iloc[t0]) ** (21 / days) - 1
    return "bull" if monthly > 0.015 else "bear" if monthly < -0.015 else "choppy"


def windows(P, months, start=260, step=5):
    """Every real past window of this length (starting each week), grouped by market type."""
    days = months * 21
    out = {k: [] for k in MARKETS}
    for t0 in range(start, len(P["C"]) - days - 1, step):
        out[classify(P, t0, days)].append(t0)
    return out


def run_window(P, ind, t0, days, names, p=PARAMS, pred=None):
    """Trade one real past window with $10,000, recording each day like the live dashboard would show it.
    The bot only uses data up to each day; the next days are hidden from it."""
    book = new_book()
    dates = P["C"].index
    spy0 = P["C"]["SPY"].iloc[t0]
    rets = P["C"].drop(columns=["SPY"]).pct_change().clip(-0.5, 1.0)
    ew = bot.START_CASH
    frames = []
    decide(book, P, ind, t0, p, force_rebalance=True)
    for t in range(t0 + 1, t0 + days + 1):
        row = lambda k: P[k].iloc[t].to_dict()
        opens, lows, closes = row("O"), row("L"), row("C")
        eq_prev = frames[-1]["eq"] if frames else bot.START_CASH
        n_before = len(book["trades"])
        if book["pending"]:
            execute(book, opens, dates[t], eq_prev, p, ind["atr"].iloc[t - 1].to_dict())
        check_stops(book, lows, opens, dates[t])
        mark(book, closes, ind["atr"].iloc[t].to_dict(), p)
        ew *= 1 + float(np.nan_to_num(rets.iloc[t].mean()))
        eq = equity_of(book, closes)
        ranks = decide(book, P, ind, t, p) if t < t0 + days else ranking(P, ind, t, p)
        preds = []
        for i, s in enumerate(ranks[:10], 1):
            preds.append(dict(rank=i, sym=s, name=names.get(s, ""), gain90=round(float(P["C"][s].iloc[t] / P["C"][s].iloc[t - p["window"] + 1] - 1), 3),
                              smooth=round(float(ind["r2"][s].iloc[t]), 2)))
        frames.append(dict(
            date=dates[t], eq=round(eq, 2), spy=round(float(bot.START_CASH * P["C"]["SPY"].iloc[t] / spy0), 2), ew=round(ew, 2),
            regime=bool(ind["regime"].iloc[t]),
            hold=[dict(sym=s, name=names.get(s, ""), value=round(h["shares"] * h["last"], 2), gain=round(h["last"] / h["entry"] - 1, 4),
                       since=h["date"]) for s, h in book["pos"].items()],
            trades=[dict(x, name=names.get(x["sym"], "")) for x in book["trades"][n_before:]],
            plan=[dict(side=o["side"], sym=o["sym"], name=names.get(o["sym"], ""), reason=o["reason"]) for o in book["pending"]],
            preds=preds))
    return frames


def sim_summary(P, ind, ts, days, p=PARAMS):
    """How the bot did across every real past window of a type: the honest 'is this typical' numbers."""
    res, beat = [], 0
    for t0 in ts:
        book = new_book()
        decide(book, P, ind, t0, p, force_rebalance=True)
        eq = bot.START_CASH
        for t in range(t0 + 1, t0 + days + 1):
            row = lambda k: P[k].iloc[t].to_dict()
            if book["pending"]:
                execute(book, row("O"), P["C"].index[t], eq, p)
            check_stops(book, row("L"), row("O"), P["C"].index[t])
            mark(book, row("C"), ind["atr"].iloc[t].to_dict(), p)
            eq = equity_of(book, row("C"))
            if t < t0 + days:
                decide(book, P, ind, t, p)
        r = eq / bot.START_CASH - 1
        spy = P["C"]["SPY"].iloc[t0 + days] / P["C"]["SPY"].iloc[t0] - 1
        res.append(r)
        beat += r > spy
    res = sorted(res)
    q = lambda f: round(res[int(f * (len(res) - 1))] * 100, 1)
    return dict(windows=len(res), profitable=round(100 * sum(x > 0 for x in res) / len(res)), beat_spy=round(100 * beat / len(res)),
                median=q(0.5), p10=q(0.1), p90=q(0.9)) if res else None


def build_sim(P, ind, names, out, per_type=6):
    """Pre-build simulator runs for the hosted dashboard: market type x length, a few real windows each."""
    index = {}
    for months in SIM_MONTHS:
        days = months * 21
        groups = windows(P, months)
        for kind, ts in groups.items():
            if not ts:
                continue
            pick = [ts[int(i * (len(ts) - 1) / max(per_type - 1, 1))] for i in range(min(per_type, len(ts)))]
            pick = sorted(set(pick))
            summary = sim_summary(P, ind, ts, days)
            runs = []
            for k, t0 in enumerate(pick, 1):
                frames = run_window(P, ind, t0, days, names)
                name = f"scansim_{kind}_{months}_{k}.json"
                with open(os.path.join(out, name), "w") as f:
                    json.dump(dict(kind=kind, months=months, start=P["C"].index[t0], frames=frames), f, default=float, separators=(",", ":"))
                runs.append(dict(file=name, start=P["C"].index[t0], end=frames[-1]["date"],
                                 bot=round(frames[-1]["eq"] / bot.START_CASH * 100 - 100, 1),
                                 spy=round(frames[-1]["spy"] / bot.START_CASH * 100 - 100, 1)))
            index[f"{kind}_{months}"] = dict(kind=kind, months=months, about=MARKETS[kind], runs=runs, summary=summary)
    return index


def benchmarks(P, start=260):
    """SPY buy-and-hold, and an equal-weight index of the same universe (carries the same survivorship bias)."""
    C = P["C"].iloc[start:]
    spy = C["SPY"] / C["SPY"].iloc[0] * bot.START_CASH
    rets = C.drop(columns=["SPY"]).pct_change().clip(-0.5, 1.0)
    ew = (1 + rets.mean(axis=1).fillna(0)).cumprod() * bot.START_CASH
    return list(zip(C.index, spy)), list(zip(C.index, ew))


def stats(curve):
    v = np.array([x[1] for x in curve], dtype=float)
    r = np.diff(v) / v[:-1]
    years = len(v) / 252
    peak = np.maximum.accumulate(v)
    yearly = {}
    for (d, x) in curve:
        yearly.setdefault(d[:4], [x, x])[1] = x
    return dict(total=round((v[-1] / v[0] - 1) * 100, 1), cagr=round(((v[-1] / v[0]) ** (1 / years) - 1) * 100, 1),
                maxdd=round(((v / peak) - 1).min() * 100, 1),
                sharpe=round(r.mean() / r.std() * math.sqrt(252), 2) if r.std() else None,
                worst_year=round(min((b / a - 1) * 100 for a, b in yearly.values()), 1),
                final=round(v[-1], 2))


# ---------------------------------------------------------------- learning: re-tune itself from the past year

LEARN_GRID = [(n, w, ew) for n in (5, 7, 10) for w in (60, 90, 120) for ew in (3, 4, 6)]  # stocks, trend days, weeks
LEARN_LOOKBACK = 252   # trading days of experience it learns from (12 months tested best; 6 and 24 were unreliable)
LEARN_EVERY = 28       # calendar days between re-tunes (monthly)


def learn(P=None, base=PARAMS):
    """Replay the last 12 months with each of 27 setting combos and pick the one that made the most.
    Walk-forward tested on 9 years: 50.5%/yr vs 41.3% fixed (600 biggest), 33.9% vs 32.8% (all stocks)."""
    if P is None:
        P = load_panel([u["symbol"] for u in universe()], "5y")
    start = len(P["C"]) - LEARN_LOOKBACK - 1
    inds, table = {}, []
    for n, w, ew in LEARN_GRID:
        p = dict(base, n=n, window=w, every_weeks=ew)
        if w not in inds:
            inds[w] = indicators(P, p)
        book, curve = backtest(P, inds[w], p, start=start)
        v = np.array([c[1] for c in curve])
        r = v[1:] / v[:-1] - 1
        table.append(dict(n=n, window=w, every_weeks=ew, ret=round(float(v[-1] / v[0] - 1) * 100, 1),
                          maxdd=round(float((v / np.maximum.accumulate(v) - 1).min()) * 100, 1),
                          sharpe=round(float(r.mean() / r.std() * math.sqrt(252)), 2) if r.std() else None))
    table.sort(key=lambda x: -x["ret"])
    return dict(at=int(time.time()), since=P["C"].index[start], until=P["C"].index[-1], best=table[0], table=table)


def journal(book):
    """Look back at its own closed trades: what worked, what didn't, and the patterns behind the losses."""
    sells = [t for t in book["trades"] if t["side"] == "SELL"]
    if not sells:
        return dict(trades=0, lessons=["No closed trades yet. Lessons appear after its first sells."])
    wins = [t for t in sells if t["pnl"] > 0]
    losses = [t for t in sells if t["pnl"] <= 0]
    days = lambda t: (datetime.fromisoformat(t["date"]) - datetime.fromisoformat(t["held_since"])).days
    quick = [t for t in losses if days(t) <= 10]
    by_reason = {}
    for t in sells:
        k = "trend broke" if "below" in t["reason"] else "rebalanced out" if "fell out" in t["reason"] else t["reason"]
        by_reason.setdefault(k, []).append(t["ret"])
    lessons = []
    if wins and losses:
        lessons.append(f"Average win {np.mean([t['ret'] for t in wins]) * 100:+.1f}% vs average loss "
                       f"{np.mean([t['ret'] for t in losses]) * 100:+.1f}%: momentum works by letting winners run.")
    if quick:
        lessons.append(f"{len(quick)} of {len(losses)} losses were sold within 10 days of buying: trends that broke right "
                       "away. The 90-day rebuy ban stops it chasing those stocks again.")
    for k, rs in by_reason.items():
        lessons.append(f"Sold because '{k}': {len(rs)} times, average result {np.mean(rs) * 100:+.1f}%.")
    return dict(trades=len(sells), win_rate=round(100 * len(wins) / len(sells)),
                avg_win=round(float(np.mean([t["ret"] for t in wins])) * 100, 1) if wins else None,
                avg_loss=round(float(np.mean([t["ret"] for t in losses])) * 100, 1) if losses else None,
                best=sorted(sells, key=lambda t: -t["ret"])[:3], worst=sorted(sells, key=lambda t: t["ret"])[:3],
                lessons=lessons)


RULES = [
    dict(kept=True, rule="Scan every US stock worth $2B+ and priced $5+ each morning (~2,100 stocks).",
         source="Nasdaq's full stock list", evidence="Covers the whole market, not one company."),
    dict(kept=True, rule="Rank by how fast and how steadily each stock climbed over 90 trading days.",
         source="Clenow, 'Stocks on the Move'; Jegadeesh & Titman (momentum); Da, Gurun & Warachka ('frog in the pan')",
         evidence="Its top picks gained +2.8% to +3.4% a month on average vs +1.6% to +1.8% for an average stock."),
    dict(kept=True, rule="Only buy stocks above their 100-day average, within 25% of their 52-week high, with enough trading.",
         source="Clenow; George & Hwang (52-week high)", evidence="Keeps it to stocks already in healthy uptrends."),
    dict(kept=True, rule="Skip stocks that jumped 15%+ in a single day lately (news spikes, not trends).",
         source="Clenow", evidence="Removing it raised returns in one test but deepened the worst drop to -41%."),
    dict(kept=True, rule="Only buy while the S&P 500 is above its 200-day average; otherwise wait in cash.",
         source="Faber's 10-month rule; O'Neil's 'M' in CAN SLIM; Daniel & Moskowitz on momentum crashes",
         evidence="Without it the worst year in the 9-year test went from -9% to -30%."),
    dict(kept=True, rule="Hold the strongest few in equal amounts and rebalance every few weeks (started at 7 stocks, every 4 weeks).",
         source="Tested here", evidence="7 beat 10 in both tests (36% and 41% a year). Weekly rebalancing earned far less (12%)."),
    dict(kept=True, rule="Learn every month: replay the past 12 months with 27 setting combos and switch to the one that made the most.",
         source="Walk-forward testing (each choice uses only data before it)",
         evidence="9 years: 50.5%/yr vs 41.3% with fixed settings. 4 years, all stocks: 33.9% vs 32.8%. Learning from 6 or 24 months was unreliable."),
    dict(kept=True, rule="Check every minute during market hours using 1-minute prices.",
         source="Your request", evidence="Faster fills at the open; trades still follow the tested daily rules."),
    dict(kept=True, rule="Sell the morning after a holding closes below its 100-day average, then move on: no rebuying it for 90 days.",
         source="Clenow's exit; tested here", evidence="The 90-day ban cost nothing; banning forever cost 2-4% a year."),
    dict(kept=False, rule="Trailing stops (sell on intraday dips).", source="Common trading advice",
         evidence="Cut returns sharply in both tests (as low as 3% a year): strong stocks dip and recover."),
    dict(kept=False, rule="Rank by nearness to the 52-week high alone.", source="George & Hwang",
         evidence="About 0% a year as a stand-alone ranking."),
    dict(kept=False, rule="Loosen the jump filter to 25%.", source="Tested here",
         evidence="+84% a year in one test but driven by a few lucky stocks; not reliable."),
]
REALITY = ("Paper money only. The tests use today's list of stocks, which leaves out companies that collapsed, so real "
           "results would be lower. The bot's top picks rose about as often as an average stock (54-57% of months); "
           "its edge is that its winners win bigger. A study of 19,646 day traders (Chague et al.) found 97% of those "
           "who kept at it lost money: be skeptical of any strategy, including this one.")


# ---------------------------------------------------------------- live paper trading (run every few minutes)

STATE_FILE = os.path.join(HERE, "scanner_state.json")


def _live_prices(symbols):
    """Latest price and today's low for each symbol, from 5-minute bars."""
    def get(sym):
        try:
            bars, quote = bot.fetch_bars(sym, "1m", "1d")  # 1-minute bars: the freshest free price
            if not bars:
                return sym, None
            return sym, dict(price=quote or bars[-1]["c"], low=min(b["l"] for b in bars), open=bars[0]["o"],
                             t=bars[-1]["t"])
        except Exception:
            return sym, None
    with ThreadPoolExecutor(8) as ex:
        return {s: v for s, v in ex.map(get, symbols) if v}


def _market_open(now):
    return now.weekday() < 5 and (9, 30) <= (now.hour, now.minute) < (16, 0)


def live_tick(p=PARAMS, out_dir=None):
    """One pass: morning scan once a day, buy/sell at the first run after the open, watch stops, publish."""
    now = datetime.now(bot.ET)
    today = now.date().isoformat()
    st = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else dict(book=new_book(), days={}, log=[])
    book = st["book"]
    before = (len(book["trades"]), len(st["log"]))
    note = lambda msg: st["log"].append([int(time.time()), msg])

    # learning: once a month (outside market hours) re-test 27 setting combos on the past year, adopt the best
    learned = st.get("learned")
    if (not learned or time.time() - learned["at"] > LEARN_EVERY * 86400) and not _market_open(now):
        try:
            old = learned["best"] if learned else dict(n=p["n"], window=p["window"], every_weeks=p["every_weeks"])
            learned = st["learned"] = learn(base=p)
            b = learned["best"]
            same = (b["n"], b["window"], b["every_weeks"]) == (old["n"], old["window"], old["every_weeks"])
            note(f"Learned from the last 12 months ({learned['since']} to {learned['until']}): best was holding {b['n']} "
                 f"stocks, {b['window']}-day trends, rebalancing every {b['every_weeks']} weeks ({b['ret']:+.1f}% "
                 f"over the year). " + ("Keeping the same settings." if same else "Switching to those settings."))
        except Exception as e:
            note(f"Learning step failed ({type(e).__name__}); keeping the current settings.")
    if st.get("learned"):
        b = st["learned"]["best"]
        p = dict(p, n=b["n"], window=b["window"], every_weeks=b["every_weeks"])

    if st.get("last_scan") != today or st.get("params_used") != p:  # new day, or the rules changed
        if st.get("last_scan") == today:
            note("Rules were updated, rescanning with the new settings.")
        uni = universe()
        names = {u["symbol"]: u for u in uni}
        P = load_panel([u["symbol"] for u in uni], "2y")
        if P["C"].index[-1] == today and now.hour < 16:  # today's bar is still forming: decide on yesterday's close
            P = {k: v.iloc[:-1] for k, v in P.items()}
        ind = indicators(P, p)
        t = len(P["C"]) - 1
        closes = P["C"].iloc[t].to_dict()
        mark(book, closes, ind["atr"].iloc[t].to_dict(), p)
        first = not book["trades"] and not book["pos"]
        ranks = decide(book, P, ind, t, p, force_rebalance=first)
        regime = bool(ind["regime"].iloc[t])
        st["atr"] = {o["sym"]: float(ind["atr"][o["sym"]].iloc[t]) for o in book["pending"] if o["side"] == "BUY"}
        # the 100-day average each holding has to stay above (checked every run, acted on at the close)
        watch = set(book["pos"]) | {o["sym"] for o in book["pending"] if o["side"] == "BUY"}
        st["sell_line"] = {s: round(float(ind["ma"][s].iloc[t]), 4) for s in watch if np.isfinite(ind["ma"][s].iloc[t])}
        st.update(last_scan=today, params_used=dict(p), scan_date=P["C"].index[t], regime=regime, universe_size=len(uni),
                  passed=len(ranks), spy_close=float(P["C"]["SPY"].iloc[t]), spy_ma=float(ind["spy_ma"].iloc[t]))
        rows = []
        for i, s in enumerate(ranks[:40], 1):
            rows.append(dict(rank=i, sym=s, name=names.get(s, {}).get("name", ""), sector=names.get(s, {}).get("sector", ""),
                             price=round(closes[s], 2), gain90=round(float(P["C"][s].iloc[t] / P["C"][s].iloc[t - p["window"] + 1] - 1), 4),
                             smooth=round(float(ind["r2"][s].iloc[t]), 2), of_high=round(float(ind["high52"][s].iloc[t]), 3),
                             held=s in book["pos"]))
        st["top"] = rows
        st["holding_status"] = {s: (why_filtered(P, ind, t, s, p) or f"rank #{ranks.index(s) + 1}" if s in ranks else
                                    why_filtered(P, ind, t, s, p) or "outside the ranking") for s in book["pos"]}
        st["names"] = {s: names.get(s, {}).get("name", "") for s in set(book["pos"]) | {o["sym"] for o in book["pending"]}}
        orders = ", ".join(f"{o['side']} {o['sym']}" for o in book["pending"]) or "no changes"
        note(f"Scanned {len(uni)} stocks ({len(ranks)} passed every filter). Market trend "
             f"{'OK' if regime else 'DOWN: no new buys'}. Plan for the open: {orders}.")
        if "start_spy" not in st:
            st["start_spy"], st["start_date"] = st["spy_close"], st["scan_date"]

    syms = sorted(set(book["pos"]) | {o["sym"] for o in book["pending"]} | {"SPY"})
    live = _live_prices(syms)
    prices = {s: v["price"] for s, v in live.items()}
    if _market_open(now):
        if book["pending"]:
            done = len(book["trades"])
            execute(book, prices, today, equity_of(book, prices), p, st.get("atr"))
            for x in book["trades"][done:]:
                x["at"] = int(time.time())
                note(f"{x['side']} {x['sym']} @ ${x['price']:,.2f}: {x['reason']}")
        lows = {s: v["low"] for s, v in live.items()}
        opens = {s: v["open"] for s, v in live.items()}
        done = len(book["trades"])
        check_stops(book, lows, opens, today)
        for x in book["trades"][done:]:
            x["at"] = int(time.time())
            note(f"SELL {x['sym']} @ ${x['price']:,.2f}: {x['reason']}")
    for s, h in book["pos"].items():
        if s in prices:
            h["last"] = prices[s]
    eq = equity_of(book, prices)

    # the 5-minute check: where every holding stands against its sell line, and what the bot will do
    rows, warn = [], []
    for s, h in book["pos"].items():
        line = st.get("sell_line", {}).get(s)
        px = prices.get(s, h["last"])
        room = px / line - 1 if line else None
        if room is not None and room < 0:
            warn.append(s)
        rows.append(dict(sym=s, price=round(px, 2), line=line, room=round(room, 4) if room is not None else None))
    if not _market_open(now):
        verdict = ("Market closed. " + (f"{len(book['pending'])} orders ready for the next open."
                                         if book["pending"] else "Nothing planned for the open."))
    elif not book["pos"]:
        verdict = "No holdings yet. " + ("Waiting for the market trend to turn up before buying."
                                         if not st.get("regime") else "Buys happen at the next scan's rebalance.")
    elif warn:
        verdict = (f"{', '.join(warn)} below the sell line right now. If still below at the 4pm close, "
                   f"it's sold at tomorrow's open. Everything else: hold.")
    else:
        verdict = f"All {len(book['pos'])} holdings above their sell lines. Decision: hold."
    st["check"] = dict(t=int(time.time()), verdict=verdict, rows=rows)
    if live.get("SPY") and now.weekday() < 5 and now.hour >= 9:
        book["equity"].append([int(time.time()), round(eq, 2), round(live["SPY"]["price"], 2)])
        book["equity"] = book["equity"][-20000:]
        st["days"][today] = [round(eq, 2), round(live["SPY"]["price"], 2)]
    st["journal"] = journal(book)
    st["updated_at"] = int(time.time())
    if (len(book["trades"]), len(st["log"])) != before:  # something worth saving right away (the loop checks this)
        open(os.path.join(HERE, ".changed"), "w").close()
    st["log"] = st["log"][-200:]
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, default=float)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(HERE, "dashboard.html"), "rb") as src, open(os.path.join(out_dir, "index.html"), "wb") as dst:
            dst.write(src.read())
        with open(os.path.join(out_dir, "rules.json"), "w") as f:
            json.dump(dict(rules=RULES, reality=REALITY, params=p), f)
        view = dict(st, equity_now=round(eq, 2), start_cash=bot.START_CASH, params=p,
                    spy_now=prices.get("SPY"), prices={s: round(v, 4) for s, v in prices.items()})
        with open(os.path.join(out_dir, "scanner.json"), "w") as f:
            json.dump(view, f, default=float, separators=(",", ":"))
    print(f"{bot.fmt(time.time())} scanner ok: {len(book['pos'])} holdings, equity ${eq:,.2f}")
    return st


def view():
    """What the dashboard shows, from the saved state (no network)."""
    if not os.path.exists(STATE_FILE):
        return dict(error="The scanner hasn't run yet. Its first scan happens on the next weekday morning.")
    st = json.load(open(STATE_FILE))
    prices = {s: h["last"] for s, h in st["book"]["pos"].items()}
    return dict(st, equity_now=round(equity_of(st["book"], prices), 2), start_cash=bot.START_CASH, params=PARAMS)


def build_backtest(out_path):
    """Backtest results for the dashboard: all tradeable stocks (5y) and the 600 biggest (10y),
    the honest track record of the rankings, and the real-past-market simulator runs."""
    uni = universe()
    names = {u["symbol"]: u["name"] for u in uni}
    res = dict(built_at=int(time.time()), params=PARAMS, tests={})
    for key, syms, rng in (("all_5y", [u["symbol"] for u in uni], "5y"),
                           ("big600_10y", [u["symbol"] for u in uni][:600], "10y")):
        P = load_panel(syms, rng)
        ind = indicators(P)
        book, curve = backtest(P, ind)
        spy, ew = benchmarks(P)
        thin = lambda c: [[d, round(float(v), 2)] for d, v in c[::5]] + [[c[-1][0], round(float(c[-1][1]), 2)]]
        sells = [x for x in book["trades"] if x["side"] == "SELL"]
        res["tests"][key] = dict(
            stocks=len(P["C"].columns) - 1, start=P["C"].index[260], end=P["C"].index[-1],
            stats=dict(scanner=stats(curve), spy=stats(spy), equal_weight=stats(ew)),
            curves=dict(scanner=thin(curve), spy=thin(spy), equal_weight=thin(ew)),
            trades=len(sells), win_rate=round(float(np.mean([x["pnl"] > 0 for x in sells])) * 100) if sells else None,
            best=sorted(sells, key=lambda x: -x["ret"])[:5], worst=sorted(sells, key=lambda x: x["ret"])[:5])
        res["tests"][key]["predictions"] = prediction_stats(P, ind)
        if key == "big600_10y":  # the simulator replays real past windows from the 9-year data
            index = build_sim(P, ind, names, os.path.dirname(out_path))
            with open(os.path.join(os.path.dirname(out_path), "scansim_index.json"), "w") as f:
                json.dump(index, f, default=float, separators=(",", ":"))
    with open(out_path, "w") as f:
        json.dump(res, f, default=float, separators=(",", ":"))
    return res


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--range", default="5y")
    ap.add_argument("--top", type=int, default=0, help="only the N biggest companies (0 = all)")
    ap.add_argument("--tick", action="store_true", help="cloud mode: one live pass, writes docs/scanner.json")
    ap.add_argument("--site", action="store_true", help="write the dashboard files to docs/ without trading")
    ap.add_argument("--build-static", action="store_true", help="write docs/static/scanner_backtest.json")
    args = ap.parse_args()
    if args.tick:
        live_tick(out_dir=os.path.join(HERE, "docs"))
        return
    if args.site:
        docs = os.path.join(HERE, "docs")
        os.makedirs(docs, exist_ok=True)
        with open(os.path.join(HERE, "dashboard.html"), "rb") as src, open(os.path.join(docs, "index.html"), "wb") as dst:
            dst.write(src.read())
        with open(os.path.join(docs, "rules.json"), "w") as f:
            json.dump(dict(rules=RULES, reality=REALITY, params=PARAMS), f)
        with open(os.path.join(docs, "scanner.json"), "w") as f:
            json.dump(view(), f, default=float, separators=(",", ":"))
        print(f"{bot.fmt(time.time())} site files written")
        return
    if args.build_static:
        os.makedirs(os.path.join(HERE, "docs", "static"), exist_ok=True)
        build_backtest(os.path.join(HERE, "docs", "static", "scanner_backtest.json"))
        print(f"{bot.fmt(time.time())} scanner backtest built")
        return
    uni = universe()
    syms = [u["symbol"] for u in uni][: args.top or None]
    t0 = time.time()
    P = load_panel(syms, args.range)
    ind = indicators(P)
    print(f"{len(P['C'].columns) - 1} stocks, {len(P['C'])} days, loaded in {time.time() - t0:.0f}s")
    if args.scan:
        names = {u["symbol"]: u for u in uni}
        t = len(P["C"]) - 1
        for i, s in enumerate(ranking(P, ind, t)[:20], 1):
            print(f"{i:>3} {s:6} {names[s]['name'][:32]:32} trend {ind['trend'][s].iloc[t]:+.0%}/yr  "
                  f"R2 {ind['r2'][s].iloc[t]:.2f}  {P['C'][s].iloc[t] / ind['hi'][s].iloc[t]:.0%} of 52w high")
    if args.backtest:
        book, curve = backtest(P, ind)
        spy, ew = benchmarks(P)
        for name, c in (("scanner", curve), ("SPY", spy), ("equal-weight universe", ew)):
            print(f"{name:24}", stats(c))
        sells = [x for x in book["trades"] if x["side"] == "SELL"]
        print(f"{len(sells)} round trips, win rate {np.mean([x['pnl'] > 0 for x in sells]):.0%}")


if __name__ == "__main__":
    main()
