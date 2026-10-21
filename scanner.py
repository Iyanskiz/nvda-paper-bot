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
    max_gap=0.20,        # skip stocks that moved more than 20% in one day lately (news jumps, not trends)
    near_high=0.75,      # must be within 25% of the 52-week high
    min_dollar_vol=50e6, # $50M+ traded a day: real, liquid trends (beat $20M in testing on both datasets)
    regime_ma=200,       # new buys only while SPY is above its 200-day average
    exit_rank=2,         # sell if it falls out of the top n * exit_rank at a rebalance
    rebalance_day=2,     # 0=Mon ... 2=Wed (Clenow trades on Wednesdays)
    every_weeks=4,       # full rebalance every 4th week: weekly churned and earned far less in testing
    trail_atr=None,      # optional trailing stop, k x ATR(20) under the highest close
    trail_pct=0.25,      # safety net: sell if a holding falls 25% under its best close since buying (checked every 30s)
    weight="score",      # put more money in the strongest trends (score-weighted) instead of equal amounts
    weight_cap=0.35,     # ...but never more than 35% of the account in one stock
    band=0.10,           # only resize a holding at a rebalance if it's 10%+ off its target (saves fees)
    phase=0,             # which week of the cycle it rebalances in
    leverage=1.25,       # while the S&P 500 trend is up, invest 1.25x the account (borrowing 25%); 1.0x when it's down
    margin_rate=0.06,    # yearly interest on borrowed money
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
                                       (p.get("every_weeks", 1) == 1 or
                                        d.isocalendar()[1] % p["every_weeks"] == p.get("phase", 0) % p["every_weeks"]))
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
        if p.get("weight") == "score":
            selling = {o["sym"] for o in orders if o["side"] == "SELL"}
            kept = [s for s in ranks if s in book["pos"] and s not in selling]
            kept += [s for s in book["pos"] if s not in selling and s not in kept]
            book["pending_w"] = target_weights(ind, t, kept + [o["sym"] for o in orders if o["side"] == "BUY"], p)
    if not is_rebalance:
        book["pending_w"] = None
    book["pending"] = orders
    return ranks


def target_weights(ind, t, syms, p=PARAMS):
    """Share of the account for each stock: proportional to its trend score, capped at weight_cap.
    The weights add up to (stocks held / n), so empty slots stay in cash."""
    if not syms:
        return {}
    sc = ind["score"].iloc[t]
    raw = np.array([max(float(sc.get(s, np.nan)), 1e-9) if np.isfinite(sc.get(s, np.nan)) else 1e-9 for s in syms])
    w = raw / raw.sum() * min(len(syms) / p["n"], 1.0)
    cap = p.get("weight_cap")
    for _ in range(20):
        over = w > cap + 1e-12 if cap else np.zeros(len(w), bool)
        if not over.any():
            break
        extra = (w[over] - cap).sum()
        w[over] = cap
        under = w < cap
        if not under.any():
            break
        w[under] += extra * w[under] / w[under].sum()
    return {s: round(float(x), 6) for s, x in zip(syms, w)}


def execute(book, prices, date, equity_now, p=PARAMS, atr=None):
    """Fill pending orders at the given prices (the next open, or the live price). Sells first, then buys."""
    for o in [o for o in book["pending"] if o["side"] == "SELL"]:
        h = book["pos"].get(o["sym"])
        px = prices.get(o["sym"])
        if h is None or px is None or not np.isfinite(px):
            continue
        _sell(book, o["sym"], px, date, o["reason"], p)
    lev = p.get("lev_now", 1.0)
    w = book.pop("pending_w", None)
    if w is not None and p.get("weight") == "score":
        _execute_weighted(book, prices, date, w, lev, p)
        book["pending"] = []
        return
    slot = equity_now * lev / p["n"]
    borrow_room = equity_now * (lev - 1.0)
    for o in [o for o in book["pending"] if o["side"] == "BUY"]:
        px = prices.get(o["sym"])
        if px is None or not np.isfinite(px) or o["sym"] in book["pos"]:
            continue
        alloc = min(book["cash"] + borrow_room, slot)
        if alloc < 0.25 * slot:
            break
        sh = alloc / (px * (1 + FEE))
        book["cash"] -= sh * px * (1 + FEE)
        a = atr.get(o["sym"]) if atr else None
        stop = px - p["trail_atr"] * a if p["trail_atr"] and a else px * (1 - p["trail_pct"]) if p.get("trail_pct") else None
        book["pos"][o["sym"]] = dict(shares=sh, entry=px, date=date, high=px, stop=stop, last=px, reason=o["reason"])
        book["trades"].append(dict(date=date, side="BUY", sym=o["sym"], price=round(px, 4), shares=round(sh, 4),
                                   reason=o["reason"]))
    book["pending"] = []


def _execute_weighted(book, prices, date, w, lev, p):
    """Rebalance to target weights: trim holdings that grew too big, top up ones that are too small, buy new ones."""
    ok = lambda s: prices.get(s) is not None and np.isfinite(prices[s])
    eq = equity_of(book, {s: prices[s] for s in book["pos"] if ok(s)})
    band, buys = p.get("band", 0.10), {o["sym"]: o for o in book["pending"] if o["side"] == "BUY"}
    for s, h in book["pos"].items():
        if s in w and ok(s):
            tgt, val = w[s] * eq * lev, h["shares"] * prices[s]
            if val - tgt > band * tgt:
                sh = (val - tgt) / prices[s]
                h["shares"] -= sh
                book["cash"] += sh * prices[s] * (1 - FEE)
                book["trades"].append(dict(date=date, side="TRIM", sym=s, price=round(prices[s], 4), shares=round(sh, 4),
                                           reason=f"grew past its target ({w[s]:.0%} of the account): taking some profit"))
    avg_w = float(np.mean(list(w.values()))) if w else 1.0 / p["n"]
    for s in list(w) + [x for x in buys if x not in w]:
        if not ok(s):
            continue
        px, h = prices[s], book["pos"].get(s)
        if h is None and s not in buys:
            continue
        tgt = w.get(s, avg_w) * eq * lev
        val = h["shares"] * px if h else 0.0
        if h and tgt - val <= band * tgt:
            continue
        alloc = min(book["cash"] + (lev - 1.0) * eq, tgt - val)
        if alloc < (0.1 if h else 0.25) * tgt:
            continue
        sh = alloc / (px * (1 + FEE))
        book["cash"] -= sh * px * (1 + FEE)
        if h:
            h["entry"] = (h["entry"] * h["shares"] + px * sh) / (h["shares"] + sh)
            h["shares"] += sh
            book["trades"].append(dict(date=date, side="ADD", sym=s, price=round(px, 4), shares=round(sh, 4),
                                       reason=f"strong trend, below its target ({w.get(s, avg_w):.0%} of the account): adding"))
        else:
            book["pos"][s] = dict(shares=sh, entry=px, date=date, high=px, stop=px * (1 - p["trail_pct"]) if p.get("trail_pct") else None,
                                  last=px, reason=buys[s]["reason"])
            book["trades"].append(dict(date=date, side="BUY", sym=s, price=round(px, 4), shares=round(sh, 4),
                                       reason=buys[s]["reason"] + f"; sized at {w.get(s, avg_w):.0%} of the account"))


def _sell(book, s, px, date, reason, p=PARAMS):
    h = book["pos"].pop(s)
    cd = p.get("cooldown_days")
    if cd and ("below" in reason or "safety net" in reason):  # the trend broke: move on, don't chase it back
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
            drop = 1 - h["stop"] / h["high"] if h.get("high") else 0
            _sell(book, s, min(op, h["stop"]) if np.isfinite(op) else h["stop"], date,
                  f"safety net: fell {drop:.0%} under its best close since buying")


def mark(book, closes, atr, p=PARAMS):
    for s, h in book["pos"].items():
        c = closes.get(s)
        if c is not None and np.isfinite(c):
            h["last"] = c
            h["high"] = max(h["high"], c)
            a = atr.get(s)
            if p["trail_atr"] and a and np.isfinite(a):
                h["stop"] = max(h["stop"] or 0, h["high"] - p["trail_atr"] * a)
            elif p.get("trail_pct"):
                h["stop"] = max(h.get("stop") or 0, h["high"] * (1 - p["trail_pct"]))


# ---------------------------------------------------------------- backtest

def backtest(P, ind, p=PARAMS, start=260):
    book = new_book()
    dates = P["C"].index
    curve = []
    p = dict(p)
    for t in range(start, len(dates)):
        p["lev_now"] = p.get("leverage", 1.0) if bool(ind["regime"].iloc[t - 1]) else 1.0
        if book["cash"] < 0:  # interest on borrowed money, one trading day
            book["cash"] -= -book["cash"] * p.get("margin_rate", 0.06) / 252
        row = lambda k: P[k].iloc[t].to_dict()
        opens, lows, closes = row("O"), row("L"), row("C")
        eq_prev = curve[-1][1] if curve else book["cash"]
        if book["pending"] or book.get("pending_w"):
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
        if book["pending"] or book.get("pending_w"):
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
            if book["pending"] or book.get("pending_w"):
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
        k = ("safety net" if "safety net" in t["reason"] else "trend broke" if "below" in t["reason"]
             else "rebalanced out" if "fell out" in t["reason"] else t["reason"])
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
    dict(kept=True, rule="Only stocks trading $50M+ a day, and skip any that jumped 20%+ in a single day lately (news spikes, not trends).",
         source="Clenow (jump filter); training round, Oct 20 2026",
         evidence="Mean over 20 rebalance timings, with leverage: all stocks 35% to 70%/yr (ahead in 95% of timings, both halves of the test, smaller worst drop); 600 biggest 46% to 51%/yr (ahead in 80%). The middle setting was chosen, not the luckiest one."),
    dict(kept=True, rule="Only buy while the S&P 500 is above its 200-day average; otherwise wait in cash.",
         source="Faber's 10-month rule; O'Neil's 'M' in CAN SLIM; Daniel & Moskowitz on momentum crashes",
         evidence="Without it the worst year in the 9-year test went from -9% to -30%."),
    dict(kept=True, rule="Hold the 7 strongest and rebalance every 4 weeks. More holdings earned less: 10, 15 and 20 stocks made 15%, 16% and 13% a year vs 22% for 7 (all stocks), and 23%, 16% and 14% vs 33% (600 biggest).",
         source="Tested here, averaged over 20 rebalance timings", evidence="Weekly rebalancing earned far less (12%). 'As many as it needs' (holding every stock above a score bar) also earned less than 7."),
    dict(kept=True, rule="Bet more on the strongest: each stock's share of the account follows its trend score (capped at 35% in one stock), resized at every rebalance.",
         source="Tested here, averaged over 20 rebalance timings", evidence="Beat equal amounts in 85% of timings (all stocks: 21.7% to 25.8%/yr) and 100% (600 biggest: 33.4% to 36.9%). Still ahead with 3x the trading costs."),
    dict(kept=True, rule="Safety net, checked every 30 seconds: sell a holding the moment it falls 25% under its best close since buying, then don't rebuy it for 90 days.",
         source="Tested here, averaged over 20 rebalance timings", evidence="With score weights: 25.8% to 29.8%/yr and worst drop -51% to -41% (all stocks); 36.9% to 39.2%/yr (600 biggest). It triggers only a few times a year."),
    dict(kept=False, rule="Monthly learning that switches to whatever settings won over the last 12 months.",
         source="Walk-forward test over 4 rebalance timings", evidence="30-31%/yr vs 49% for the fixed rules (all stocks) and 37-39% vs 39% (600 biggest). Its earlier win was luck of one timing. It now runs as a monthly report only."),
    dict(kept=True, rule="While the S&P 500 trend is up, invest 1.25x the account (borrowing 25% at 6% a year); no borrowing when it's down.",
         source="Tested here (smart-risk experiment)",
         evidence="7-stock rules: 32.7% to 40.7%/yr (all stocks) and 41.3% to 51.1%/yr (600 biggest); worst drop -30% to -36% and -26% to -32%. 1.5x earned more but drops reached -44% to -58%."),
    dict(kept=True, rule="Read company news and SEC filings for every holding and the next candidates: serious red flags (bankruptcy, delisting, restatement, going concern, fraud action, FDA rejection) block buys and sell holdings; warnings (offerings, guidance cuts, short reports) block buys and sell only if the price falls 5%+ vs the market.",
         source="Chan 2003; Tetlock et al. 2008; Loughran & McDonald 2011; event studies on offerings, restatements, bankruptcy",
         evidence="Bad news drifts for weeks; good news is priced in within hours, so good news only breaks ties. Not backtested here (no free headline history); a protective rule."),
    dict(kept=True, rule="Rescan all ~2,100 stocks (including 440+ foreign companies listed in the US) with live prices every 30 seconds: live leaderboard, today's movers, sell-line and safety-net warnings.",
         source="Your request", evidence="Re-ranking or swapping buys at the open did not beat the tested daily rules (24-26% vs 24%/yr; 35-41% vs 38%), so the rescan watches and reports but the daily rules decide."),
    dict(kept=True, rule="Check every holding every 30 seconds during market hours using live prices.",
         source="Your request", evidence="Fast fills at the open and the safety net acts within 30 seconds; buys and rebalances still follow the tested daily rules."),
    dict(kept=True, rule="Sell the morning after a holding closes below its 100-day average, then move on: no rebuying it for 90 days.",
         source="Clenow's exit; tested here", evidence="The 90-day ban cost nothing; banning forever cost 2-4% a year."),
    dict(kept=False, rule="Tight stops: sell on a 10-20% dip from the high, a 5-12% drop in one day, or an 8-15% loss from the buy price.", source="Common trading advice",
         evidence="Every version earned less, most of them far less (3-18% a year vs 22% on all stocks): strong stocks dip and recover, and the bot ends up selling low."),
    dict(kept=False, rule="Refill a freed slot right away with the next-best stock instead of waiting for the rebalance.", source="Tested here",
         evidence="About the same return with a deeper worst drop (-39% vs -35%)."),
    dict(kept=False, rule="Portfolio stop: sell everything after the account falls 15-20% from its peak.", source="Tested here",
         evidence="15%: 24% to 15%/yr; 20%: 24% to 23%/yr. It sold near bottoms and missed the rebounds."),
    dict(kept=False, rule="Rank by nearness to the 52-week high alone.", source="George & Hwang",
         evidence="About 0% a year as a stand-alone ranking."),
    dict(kept=False, rule="Borrow more: 1.5x or 1.75x instead of 1.25x.", source="Training round, Oct 20 2026",
         evidence="More profit (all stocks 35% to 39-42%/yr) but the average worst drop went from -42% to -48% and -54%. Too much risk."),
    dict(kept=False, rule="Hold many more stocks (10, 15, 25).", source="Training round, Oct 20 2026",
         evidence="Every step up earned less: 33%, 30%, 26% vs 35%/yr (all stocks); 41%, 35%, 29% vs 46% (600 biggest). It can hold more, but 7 makes the most money."),
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


def market_snapshot(symbols, batch=20, threads=10):
    """Live price and today's high/low for every symbol, ~20 per request via Yahoo's spark endpoint
    (about 3 seconds for all ~2,100). Nasdaq's all-stocks list is a day stale, so it isn't used for live prices."""
    def get(chunk):
        url = ("https://query1.finance.yahoo.com/v8/finance/spark?symbols=" + ",".join(chunk) +
               "&range=1d&interval=1m")
        for k in range(2):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    data = json.load(r)
                out = {}
                for sym, v in data.items():
                    closes = [c for c in (v.get("close") or []) if c is not None]
                    if closes:
                        out[sym] = dict(price=closes[-1], prev=v.get("chartPreviousClose"),
                                        high=max(closes), low=min(closes), t=(v.get("timestamp") or [0])[-1])
                return out
            except Exception:
                time.sleep(1 + k)
        return {}
    chunks = [symbols[i:i + batch] for i in range(0, len(symbols), batch)]
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(get, chunks))
    return {k: v for r in res for k, v in r.items()}


def _today_numbers(st, eq, spy_now, today):
    """Today's profit: equity now vs. the last equity recorded on an earlier day (or the starting cash)."""
    prev_days = sorted(d for d in st.get("days", {}) if d < today)
    start_eq = st["days"][prev_days[-1]][0] if prev_days else bot.START_CASH
    prev_spy = st["days"][prev_days[-1]][1] if prev_days else (st.get("spy_close") if st.get("scan_date", today) < today else None)
    spy_pct = (spy_now / prev_spy - 1) if (spy_now and prev_spy) else None
    return dict(date=today, start_equity=round(start_eq, 2), pnl=round(eq - start_eq, 2),
                pct=round(eq / start_eq - 1, 5), spy_pct=round(spy_pct, 5) if spy_pct is not None else None)


def prices_spy_pct(live, st):
    spy = live.get("SPY")
    prev = st.get("spy_close") if st.get("scan_date", "") < datetime.now(bot.ET).date().isoformat() else None
    return (spy["price"] / prev - 1) if spy and prev else None


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

    # monthly review (outside market hours): replay the past year with 27 setting combos and report what would have won.
    # It does NOT switch settings: in walk-forward tests, chasing last year's winner earned 30-39%/yr vs 39-49% for
    # the fixed rules (averaged over 4 rebalance timings), so the bot adapts through its rules instead.
    learned = st.get("learned")
    if (not learned or time.time() - learned["at"] > LEARN_EVERY * 86400 or learned.get("v") != 2) and now.hour >= 16:
        try:
            learned = st["learned"] = dict(learn(base=p), v=2)
            b = learned["best"]
            mine = next((r for r in learned["table"] if (r["n"], r["window"], r["every_weeks"]) ==
                         (p["n"], p["window"], p["every_weeks"])), None)
            note(f"Monthly review of the last 12 months ({learned['since']} to {learned['until']}): the best combo was "
                 f"{b['n']} stocks, {b['window']}-day trends, every {b['every_weeks']} weeks ({b['ret']:+.1f}%); the current "
                 f"rules made {mine['ret']:+.1f}%. Keeping the current rules: switching to last year's winner lost in testing."
                 if mine else "Monthly review done.")
        except Exception as e:
            note(f"Monthly review failed ({type(e).__name__}); rules unchanged.")

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

    # ---- news: holdings, planned buys and the next candidates (cached 15 min; failures never block trading)
    import news as NEWS
    cand = [r["sym"] for r in st.get("top", [])[:15]]
    watch_news = sorted(set(book["pos"]) | {o["sym"] for o in book["pending"]} | set(cand))
    try:
        st["news"] = NEWS.check(watch_news)
    except Exception as e:
        st.setdefault("news", {})
        note(f"News check failed ({type(e).__name__}); trading on price rules only.")
    nv = st.get("news", {})
    bad = lambda sym, levels=("veto", "caution"): nv.get(sym, {}).get("verdict") in levels
    # planned buys hit by bad news are swapped for the next clean candidate
    if book["pending"] and any(o["side"] == "BUY" and bad(o["sym"]) for o in book["pending"]):
        keep = [o for o in book["pending"] if not (o["side"] == "BUY" and bad(o["sym"]))]
        dropped = [o["sym"] for o in book["pending"] if o["side"] == "BUY" and bad(o["sym"])]
        taken = set(book["pos"]) | {o["sym"] for o in keep}
        for sym in cand:
            if len(keep) - sum(o["side"] == "SELL" for o in keep) >= len(book["pending"]) - sum(o["side"] == "SELL" for o in book["pending"]):
                break
            if sym in taken or bad(sym) or book.get("banned", {}).get(sym, "") > today:
                continue
            row = next(r for r in st["top"] if r["sym"] == sym)
            keep.append(dict(side="BUY", sym=sym, rank=row["rank"], reason=f"#{row['rank']} strongest steady uptrend "
                             f"(replacing a buy blocked by news): {row['gain90']:+.0%} in {p['window']} trading days"))
            taken.add(sym)
        book["pending"] = keep
        for sym in dropped:
            note(f"Skipped buying {sym}: {nv[sym]['summary']}")
    # holdings with a serious red flag are sold; a warning sells only if the price confirms (down 5%+ vs the market)
    for sym in list(book["pos"]):
        v = nv.get(sym, {})
        if v.get("verdict") == "veto" and not any(o["sym"] == sym and o["side"] == "SELL" for o in book["pending"]):
            book["pending"].append(dict(side="SELL", sym=sym, reason="news veto: " + ", ".join(v.get("red_flags") or ["red flag"])))

    syms = sorted(set(book["pos"]) | {o["sym"] for o in book["pending"]} | {"SPY"})
    live = _live_prices(syms)
    prices = {s: v["price"] for s, v in live.items()}
    p = dict(p, lev_now=p.get("leverage", 1.0) if st.get("regime") else 1.0)
    st["risk"] = dict(exposure=p["lev_now"], note=(f"Investing {p['lev_now']:.2f}x the account (borrowing "
                      f"{p['lev_now'] - 1:.0%} at {p.get('margin_rate', 0.06):.0%} a year) while the market trend is up"
                      if p["lev_now"] > 1 else "No borrowing: the market trend is down, so no leverage"))
    if st.get("interest_day") != today and now.weekday() < 5 and book["cash"] < 0:
        interest = -book["cash"] * p.get("margin_rate", 0.06) / 252
        book["cash"] -= interest
        st["interest_paid"] = round(st.get("interest_paid", 0) + interest, 2)
    st["interest_day"] = today
    if _market_open(now):
        # live rescan of every stock: fresh prices for the whole market each minute
        try:
            t0 = time.time()
            snap = market_snapshot([u["symbol"] for u in universe()])
            took = time.time() - t0
            changes = []
            for sym in list(book["pos"]):  # caution news + a 5% drop vs the market = sell now
                v, q, spy = nv.get(sym, {}), snap.get(sym), live.get("SPY")
                if v.get("verdict") == "caution" and q and q.get("prev") and spy:
                    rel = (q["price"] / q["prev"] - 1) - (prices_spy_pct(live, st) or 0)
                    if rel <= -0.05 and not any(o["sym"] == sym for o in book["pending"]):
                        book["pending"].append(dict(side="SELL", sym=sym, reason="bad news confirmed by the price: "
                                                    + ", ".join(v.get("warnings") or ["warning"])))
                        changes.append(f"Selling {sym}: {v.get('summary', '')} and it's down {rel:.0%} vs the market")
            top = []
            for r in st.get("top", [])[:10]:
                q = snap.get(r["sym"], {})
                chg = q["price"] / q["prev"] - 1 if q.get("prev") else None
                top.append(dict(rank=r["rank"], sym=r["sym"], name=r["name"], live_price=q.get("price"),
                                live_change=round(chg, 4) if chg is not None else None, held=r["sym"] in book["pos"]))
            movers = sorted(((v["price"] / v["prev"] - 1, k) for k, v in snap.items() if v.get("prev")), reverse=True)
            for sym in book["pos"]:
                line = st.get("sell_line", {}).get(sym)
                if line and snap.get(sym) and snap[sym]["price"] < line:
                    changes.append(f"{sym} is below its sell line right now; sells at tomorrow's open if it closes there")
            st["rescan"] = dict(t=int(time.time()), scanned=len(snap), took_s=round(took, 1), changes=changes, top=top,
                                gainers=[dict(sym=k, change=round(c, 4)) for c, k in movers[:5]],
                                losers=[dict(sym=k, change=round(c, 4)) for c, k in movers[-5:][::-1]])
        except Exception as e:
            note(f"Live rescan failed ({type(e).__name__}); using the morning scan.")
        if book["pending"] or book.get("pending_w"):
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

    # optional brokerage copy (Alpaca): only when keys are set as GitHub secrets; never blocks the bot
    import broker
    if broker.configured() and (_market_open(now) or st.get("broker", {}).get("t", 0) < time.time() - 1800):
        st["broker"] = broker.sync(book, prices, note)
    elif not broker.configured():
        st["broker"] = dict(connected=False)

    # the 30-second check: where every holding stands against its sell line and safety net, and what the bot will do
    rows, warn = [], []
    for s, h in book["pos"].items():
        line = st.get("sell_line", {}).get(s)
        px = prices.get(s, h["last"])
        room = px / line - 1 if line else None
        if room is not None and room < 0:
            warn.append(s)
        rows.append(dict(sym=s, price=round(px, 2), line=line, room=round(room, 4) if room is not None else None,
                         stop=round(h["stop"], 2) if h.get("stop") else None,
                         stop_room=round(px / h["stop"] - 1, 4) if h.get("stop") else None))
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
        verdict = (f"All {len(book['pos'])} holdings above their sell lines and safety nets. Decision: hold. "
                   "A holding is sold instantly if it falls 25% under its best close since buying.")
    st["check"] = dict(t=int(time.time()), verdict=verdict, rows=rows)
    if live.get("SPY") and now.weekday() < 5 and now.hour >= 9:
        book["equity"].append([int(time.time()), round(eq, 2), round(live["SPY"]["price"], 2)])
        book["equity"] = book["equity"][-20000:]
        st["days"][today] = [round(eq, 2), round(live["SPY"]["price"], 2)]
    st["journal"] = journal(book)
    if now.weekday() < 5:
        st["today"] = _today_numbers(st, eq, prices.get("SPY"), today)
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
