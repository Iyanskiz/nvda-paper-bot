"""Training harness on the PRODUCTION engine (scanner.backtest). Every variant runs on both datasets x 20 rebalance
timings (5 weekdays x 4 week phases), and the summary is the mean, so one lucky timing can't fool us.

    python research/train.py r9          (variants are defined in ROUNDS below)

Results go to research/results/<round>.json. "now" is always the live rules (scanner.PARAMS)."""
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import scanner

HERE = os.path.dirname(os.path.abspath(__file__))
DS = {"all_4y": "panel_5y_2089_2026-10-20.pkl", "big600_9y": "panel_10y_600_2026-10-20.pkl"}
OFFSETS = [(day, ph) for ph in range(4) for day in range(5)]
_P, _IND = {}, {}


def panel(name):
    if name not in _P:
        _P[name] = pd.read_pickle(os.path.join(scanner.CACHE, DS[name]))
    return _P[name]


def inds(name, p):
    key = (name, p["window"], p["ma"], p["regime_ma"])
    if key not in _IND:
        _IND[key] = scanner.indicators(panel(name), p)
    return _IND[key]


def run(job):
    lab, name, p = job
    book, curve = scanner.backtest(panel(name), inds(name, p), p)
    v = np.array([c[1] for c in curve])
    h = len(curve) // 2
    s = scanner.stats(curve)
    m = v[::21]
    return lab, name, dict(cagr=float(s["cagr"]), maxdd=float(s["maxdd"]), sharpe=float(s["sharpe"] or 0),
                           h1=float(scanner.stats(curve[:h])["cagr"]), h2=float(scanner.stats(curve[h:])["cagr"]),
                           up_months=float((m[1:] > m[:-1]).mean() * 100),
                           worst_month=float((m[1:] / m[:-1] - 1).min() * 100),
                           trades=sum(1 for x in book["trades"] if x["side"] in ("BUY", "SELL")))


def evaluate(V, out, pool=11):
    jobs = [(lab, name, dict(scanner.PARAMS, **p, rebalance_day=d, phase=ph))
            for lab, p in V.items() for name in DS for d, ph in OFFSETS]
    t0 = time.time()
    with Pool(pool) as pl:
        res = pl.map(run, jobs, chunksize=1)
    raw = {}
    for lab, name, s in res:
        raw.setdefault(lab, {}).setdefault(name, []).append(s)
    base = raw.get("now")
    summ = {}
    for lab, by in raw.items():
        for name, rs in by.items():
            mean = lambda k: round(float(np.mean([x[k] for x in rs])), 2)
            row = {k: mean(k) for k in ("cagr", "maxdd", "sharpe", "h1", "h2", "up_months", "worst_month", "trades")}
            row["worst_dd"] = round(min(x["maxdd"] for x in rs), 1)
            if base:
                row["beat_now"] = round(float(np.mean([x["cagr"] > q["cagr"] for x, q in zip(rs, base[name])]) * 100))
            summ.setdefault(lab, {})[name] = row
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(summary=summ, raw=raw), open(out, "w"), default=float)
    print(f"{len(jobs)} backtests in {time.time() - t0:.0f}s")
    for name in DS:
        print(f"\n== {name} (mean of 20 timings)")
        for lab in V:
            x = summ[lab][name]
            print(f"{lab:40} cagr {x['cagr']:6.1f} h1 {x['h1']:6.1f} h2 {x['h2']:6.1f} dd {x['maxdd']:6.1f} "
                  f"wDD {x['worst_dd']:6.1f} shp {x['sharpe']:4.2f} wMon {x['worst_month']:6.1f} beat {x.get('beat_now', '-')}%",
                  flush=True)
    return summ


ROUNDS = {
    "r9": {
        "now": {},
        "own form: borrow less when losing (50d)": dict(eqc_ma=50, eqc_low=1.0),
        "own form: borrow less when losing (20d)": dict(eqc_ma=20, eqc_low=1.0),
        "own form: 50d, x1.2 when winning": dict(eqc_ma=50, eqc_low=1.0, eqc_boost=1.2, eqc_cap=1.75),
        "ban swapped-out losers 30d": dict(swap_ban=30),
        "ban swapped-out losers 90d": dict(swap_ban=90),
        "safety-net ban 180d": dict(stop_ban_days=180),
        "tighter net on +50% winners (18%)": dict(trail_tiers=[[0.5, 0.18]]),
        "tighter net: +50%->20%, +100%->15%": dict(trail_tiers=[[0.5, 0.20], [1.0, 0.15]]),
        "borrow ladder 1.75/1.5/1.25/1.0": dict(vol_lev=[[0.12, 1.75], [0.20, 1.5], [0.30, 1.25], [9.0, 1.0]]),
        "borrow 1.75 calm / 1.4 / 1.0": dict(vol_lev=[[0.15, 1.75], [0.25, 1.4], [9.0, 1.0]]),
        "6 stocks": dict(n=6),
        "5 stocks": dict(n=5),
        "bet sizes score^1.5": dict(weight_pow=1.5),
        "swap at top 18 (2.5x)": dict(mid_rank=2.5),
    },
}

# round 10: spend the drawdown room the daily swaps freed up. A grid of borrowing ladders x holdings x bet sizing;
# the winner is the highest return whose average worst drop stays inside the old rules' (-44% all, -48% big).
LADDERS = {"L1.5": [[0.15, 1.5], [0.25, 1.25], [9.0, 1.0]],          # today
           "L1.75": [[0.15, 1.75], [0.25, 1.4], [9.0, 1.0]],
           "L2.0": [[0.12, 2.0], [0.18, 1.6], [0.25, 1.3], [9.0, 1.0]],
           "L2.0b": [[0.15, 2.0], [0.25, 1.5], [9.0, 1.0]]}
ROUNDS["r11"] = {   # robustness of the round-10 pick: 1.75x when calm, 1.4x normal, 7 stocks
    "now": {},
    "pick: calm<15% 1.75 / <25% 1.4": dict(vol_lev=[[0.15, 1.75], [0.25, 1.4], [9.0, 1.0]]),
    "calm line 12%": dict(vol_lev=[[0.12, 1.75], [0.25, 1.4], [9.0, 1.0]]),
    "calm line 18%": dict(vol_lev=[[0.18, 1.75], [0.25, 1.4], [9.0, 1.0]]),
    "calm 1.65x": dict(vol_lev=[[0.15, 1.65], [0.25, 1.4], [9.0, 1.0]]),
    "calm 1.85x": dict(vol_lev=[[0.15, 1.85], [0.25, 1.4], [9.0, 1.0]]),
    "normal 1.3x": dict(vol_lev=[[0.15, 1.75], [0.25, 1.3], [9.0, 1.0]]),
    "normal 1.5x": dict(vol_lev=[[0.15, 1.75], [0.25, 1.5], [9.0, 1.0]]),
    "now @ 10% margin interest": dict(margin_rate=0.10),
    "pick @ 10% margin interest": dict(vol_lev=[[0.15, 1.75], [0.25, 1.4], [9.0, 1.0]], margin_rate=0.10),
}
ROUNDS["r10"] = {"now": {}}
for lname, lad in LADDERS.items():
    for n in (5, 6, 7):
        for pw in (1.0, 1.5):
            if lname == "L1.5" and n == 7 and pw == 1.0:
                continue                                   # that's "now"
            ROUNDS["r10"][f"{lname} n{n} pow{pw}"] = dict(vol_lev=lad, n=n, weight_pow=pw)

if __name__ == "__main__":
    rnd = sys.argv[1]
    evaluate(ROUNDS[rnd], os.path.join(HERE, "results", f"{rnd}.json"))
