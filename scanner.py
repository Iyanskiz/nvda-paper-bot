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
    n=10,                # how many stocks to hold
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
                if s in book["pos"] or s in selling:
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
        _sell(book, o["sym"], px, date, o["reason"])
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


def _sell(book, s, px, date, reason):
    h = book["pos"].pop(s)
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


# ---------------------------------------------------------------- live paper trading (run every few minutes)

STATE_FILE = os.path.join(HERE, "scanner_state.json")


def _live_prices(symbols):
    """Latest price and today's low for each symbol, from 5-minute bars."""
    def get(sym):
        try:
            bars, quote = bot.fetch_bars(sym, "5m", "1d")
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
    note = lambda msg: st["log"].append([int(time.time()), msg])

    if st.get("last_scan") != today:
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
        st.update(last_scan=today, scan_date=P["C"].index[t], regime=regime, universe_size=len(uni),
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
                note(f"{x['side']} {x['sym']} @ ${x['price']:,.2f}: {x['reason']}")
        lows = {s: v["low"] for s, v in live.items()}
        opens = {s: v["open"] for s, v in live.items()}
        done = len(book["trades"])
        check_stops(book, lows, opens, today)
        for x in book["trades"][done:]:
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
        book["equity"].append([int(time.time()), round(eq, 2)])
        book["equity"] = book["equity"][-20000:]
        st["days"][today] = [round(eq, 2), round(live["SPY"]["price"], 2)]
    st["updated_at"] = int(time.time())
    st["log"] = st["log"][-200:]
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, default=float)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
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
    """Backtest results for the dashboard: all tradeable stocks (5y) and the 600 biggest (10y)."""
    uni = universe()
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
    ap.add_argument("--build-static", action="store_true", help="write docs/static/scanner_backtest.json")
    args = ap.parse_args()
    if args.tick:
        live_tick(out_dir=os.path.join(HERE, "docs"))
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
