"""Market simulator for the paper bot.

Builds fake-but-realistic 5-minute NVDA sessions by stitching together real trading days
(block bootstrap): every simulated day is a real NVDA day's intraday path, overnight gap and
volume, replayed in a new order. A scenario can tilt the drift (bull / bear / choppy).
Then it drives the real LiveBot class with a fake clock, so the sim exercises the same code
that trades live.

    python sim.py                    # realism check + equivalence test + Monte Carlo table
"""
import math
import random
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta

import bot

BARS_PER_DAY = 78
WARMUP_DAYS = 5  # history the bot sees before the sim starts (needs 200 bars for the 200 SMA)

# extra drift per day added on top of the de-meaned real moves
SCENARIOS = {
    "bull": 0.004,      # about +8% a month, a strong NVDA stretch
    "choppy": 0.0,      # real NVDA volatility, no trend
    "bear": -0.004,     # about -8% a month
    "history": None,    # real days reshuffled as-is, no adjustment
}

_cache = {}


def real_days(symbol="NVDA"):
    """Last ~60 real sessions of 5-minute bars, grouped by day."""
    if symbol not in _cache:
        bars, _ = bot.fetch_bars(symbol, "5m", "60d")
        days = defaultdict(list)
        for b in bars:
            days[datetime.fromtimestamp(b["t"], bot.ET).date()].append(b)
        _cache[symbol] = [d for _, d in sorted(days.items()) if len(d) == BARS_PER_DAY]
    return _cache[symbol]


def trading_days(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def generate(symbol="NVDA", sim_days=20, scenario="choppy", seed=0, start=date(2026, 10, 19)):
    """Return 5-minute bars: WARMUP_DAYS of history, then sim_days of simulated future."""
    rng = random.Random(seed)
    days = real_days(symbol)
    # overnight gaps and day returns from the real sample, used to de-mean
    gaps = [days[i][0]["o"] / days[i - 1][-1]["c"] for i in range(1, len(days))]
    day_ret = [d[-1]["c"] / d[0]["o"] for d in days]
    mean_log = statistics.mean(math.log(x) for x in day_ret) + statistics.mean(math.log(g) for g in gaps)
    drift = SCENARIOS[scenario]

    total = WARMUP_DAYS + sim_days
    picks = []
    while len(picks) < total:  # blocks of 3 consecutive real days keep volatility clustering
        i = rng.randrange(1, len(days) - 2)
        picks.extend(range(i, i + 3))
    picks = picks[:total]

    dates = trading_days(start - timedelta(days=7), total)  # warm-up days land the week before
    price = days[-1][-1]["c"]
    out = []
    for k, (di, d) in enumerate(zip(picks, dates)):
        src = days[di]
        # remove the sample's average daily move, then add the scenario drift (not on warm-up days)
        tilt = 0.0 if drift is None or k < WARMUP_DAYS else drift - mean_log
        gap = gaps[di - 1]
        open_px = price * gap
        base = src[0]["o"]
        t0 = int(datetime(d.year, d.month, d.day, 9, 30, tzinfo=bot.ET).timestamp())
        for j, b in enumerate(src):
            f = open_px / base * math.exp(tilt * (j + 1) / BARS_PER_DAY)
            out.append(dict(t=t0 + 300 * j, o=b["o"] * f, h=b["h"] * f, l=b["l"] * f,
                            c=b["c"] * f, v=b["v"]))
        price = out[-1]["c"]
    return out


def sim_context(bars, symbol="NVDA"):
    """Daily context for a simulated market: NVDA's real daily closes, then each simulated day's close
    appended, so the 200-day filter reacts to the simulated trend. Real next earnings date applies."""
    real = bot.safe_context(symbol)
    hist = _daily_hist(symbol)
    first = datetime.fromtimestamp(bars[0]["t"], bot.ET).date().isoformat()
    closes = [(d, c) for d, c in hist if d < first]
    by_day = {}
    for b in bars:
        by_day[datetime.fromtimestamp(b["t"], bot.ET).date().isoformat()] = b["c"]
    closes += sorted(by_day.items())
    flags, s50, s200 = bot.trend_flags([d for d, _ in closes], [c for _, c in closes])
    return dict(bot.NO_CONTEXT, **flags, earnings=real.get("earnings"), s200=s200, s50=s50)


def _daily_hist(symbol):
    key = ("daily", symbol)
    if key not in _cache:
        bars, _ = bot.fetch_bars(symbol, "1d", "2y")
        _cache[key] = [(datetime.fromtimestamp(b["t"], bot.ET).date().isoformat(), b["c"]) for b in bars]
    return _cache[key]


# ---------------------------------------------------------------- run the real LiveBot on it

def run_live(bars, preset="smart", symbol="NVDA"):
    """Drive bot.LiveBot bar by bar with a fake clock. Returns replay-format data."""
    cursor = {"i": 0}

    def feed(_sym, _iv, _rng):
        visible = bars[max(0, cursor["i"] + 1 - 5 * BARS_PER_DAY): cursor["i"] + 1]
        return visible, visible[-1]["c"]

    def clock():
        return bars[cursor["i"]]["t"] + 300  # the moment bar i has just closed

    ctx = sim_context(bars, symbol)
    live = bot.LiveBot(symbol, preset, feed=feed, clock=clock, state_file=None, context=lambda _s: ctx)
    live.acct = dict(bot.new_account(), last_t=None)
    live.log = []
    start = WARMUP_DAYS * BARS_PER_DAY - 1
    held, stops = [], []
    for i in range(len(bars)):
        cursor["i"] = i
        if i >= start:
            live.tick()
        a = live.acct
        held.append(a["shares"] > 0)
        stops.append(round(a["stop"], 4) if a["shares"] > 0 and a["stop"] else None)
    p = bot.PRESETS[preset]
    regime, earn = bot.context_series(bars, ctx, p.get("regime_key", "above"))
    ind = bot.indicators(bars, p, regime, earn)
    r = lambda xs: [None if x is None else round(x, 4) for x in xs]
    return dict(symbol=symbol, interval="5m", preset=preset, start=start, params=p,
                bars=bars, fast=r(ind["fast"]), slow=r(ind["slow"]), trend=r(ind["trend"]),
                hi=r(ind["hi20"]), lo=r(ind["lo10"]), regime=regime, earnings=earn, rvol=r(ind["rvol"]),
                stops=stops, held=held, trades=live.acct["trades"], equity=live.acct["equity"],
                log=live.log)


def fast_replay(bars, preset="smart", symbol="NVDA"):
    """Same output as run_live, via the backtest loop. Trades are identical (checked in __main__), just faster.
    Used to pre-build the hosted dashboard."""
    ctx = sim_context(bars, symbol)
    p = bot.PRESETS[preset]
    regime, earn = bot.context_series(bars, ctx, p.get("regime_key", "above"))
    ind = bot.indicators(bars, p, regime, earn)
    acct = bot.new_account()
    start = WARMUP_DAYS * BARS_PER_DAY - 1
    held, stops = [], []
    for i in range(len(bars)):
        if i > start:
            bot.step(acct, bars, ind, i, p)
        held.append(acct["shares"] > 0)
        stops.append(round(acct["stop"], 4) if acct["shares"] > 0 and acct["stop"] else None)
    r = lambda xs: [None if x is None else round(x, 4) for x in xs]
    return dict(symbol=symbol, interval="5m", preset=preset, start=start, params=p,
                bars=bars, fast=r(ind["fast"]), slow=r(ind["slow"]), trend=r(ind["trend"]),
                hi=r(ind["hi20"]), lo=r(ind["lo10"]), regime=regime, earnings=earn, rvol=r(ind["rvol"]),
                stops=stops, held=held, trades=acct["trades"], equity=acct["equity"])


def fast_run(bars, preset="smart", symbol="NVDA"):
    """Same rules via the backtest loop (much faster), for Monte Carlo."""
    p = bot.PRESETS[preset]
    ind = bot.indicators(bars, p, *bot.context_series(bars, sim_context(bars, symbol), p.get("regime_key", "above")))
    acct = bot.new_account()
    for i in range(WARMUP_DAYS * BARS_PER_DAY, len(bars)):
        bot.step(acct, bars, ind, i, p)
    return acct


def monte_carlo(scenario, preset="smart", runs=200, sim_days=20, symbol="NVDA"):
    rets, holds, ntr = [], [], []
    for seed in range(runs):
        bars = generate(symbol, sim_days, scenario, seed)
        acct = fast_run(bars, preset, symbol)
        eq = acct["equity"][-1][1] if acct["equity"] else bot.START_CASH
        rets.append(eq / bot.START_CASH - 1)
        s0 = bars[WARMUP_DAYS * BARS_PER_DAY]["c"]
        holds.append(bars[-1]["c"] / s0 - 1)
        ntr.append(sum(t["side"] == "SELL" for t in acct["trades"]))
    rets.sort()
    q = lambda xs, p: xs[int(p * (len(xs) - 1))]
    return dict(scenario=scenario, preset=preset, runs=runs, weeks=sim_days // 5,
                profitable=round(100 * sum(r > 0 for r in rets) / runs),
                beat_hold=round(100 * sum(r > h for r, h in zip(rets, holds)) / runs),
                median=round(100 * statistics.median(rets), 1),
                p10=round(100 * q(rets, 0.1), 1), p90=round(100 * q(rets, 0.9), 1),
                hold_median=round(100 * statistics.median(holds), 1),
                trades=round(statistics.mean(ntr), 1))


# ---------------------------------------------------------------- realism check

def stats(bars):
    r5 = [math.log(bars[i]["c"] / bars[i - 1]["c"]) for i in range(1, len(bars))
          if bars[i]["t"] - bars[i - 1]["t"] == 300]
    days = defaultdict(list)
    for b in bars:
        days[datetime.fromtimestamp(b["t"], bot.ET).date()].append(b)
    ds = [v for _, v in sorted(days.items())]
    rd = [math.log(ds[i][-1]["c"] / ds[i - 1][-1]["c"]) for i in range(1, len(ds))]
    rng = [(max(b["h"] for b in d) - min(b["l"] for b in d)) / d[0]["o"] for d in ds]
    m, sd = statistics.mean(r5), statistics.pstdev(r5)
    kurt = statistics.mean(((x - m) / sd) ** 4 for x in r5)
    a = [abs(x) for x in r5]
    ma = statistics.mean(a)
    ac = sum((a[i] - ma) * (a[i - 1] - ma) for i in range(1, len(a))) / sum((x - ma) ** 2 for x in a)
    return dict(vol_5m=round(sd * 100, 3), kurtosis_5m=round(kurt, 1), vol_daily=round(statistics.pstdev(rd) * 100, 2),
                day_range=round(statistics.mean(rng) * 100, 2), vol_cluster=round(ac, 2))


if __name__ == "__main__":
    real = [b for d in real_days() for b in d]
    print("Realism: real NVDA (last 60 days) vs simulated (40 days, history scenario, 5 seeds)")
    print("  real:", stats(real))
    for seed in range(3):
        print(f"  sim{seed}:", stats(generate(sim_days=40, scenario="history", seed=seed)))

    print("\nEquivalence: live code path vs backtest loop on the same simulated bars")
    for seed in range(3):
        bars = generate(sim_days=20, scenario="bull", seed=seed)
        live = run_live(bars, "swing")
        fast = fast_run(bars, "swing")
        same = [(t["t"], t["side"], t["price"]) for t in live["trades"]] == \
               [(t["t"], t["side"], t["price"]) for t in fast["trades"]]
        print(f"  seed {seed}: live {len(live['trades'])} fills, backtest {len(fast['trades'])} fills, identical={same}, "
              f"final ${live['equity'][-1][1]:,.2f}")

    print("\nMonte Carlo: 200 simulated 4-week runs per scenario, $10,000 start")
    print(f"  {'scenario':9}{'strategy':10}{'profitable':>11}{'beat hold':>10}{'median':>8}{'worst10%':>9}{'best10%':>8}{'hold med':>9}{'trades':>7}")
    for sc in SCENARIOS:
        for pr in ("swing", "_old"):
            m = monte_carlo(sc, pr)
            print(f"  {sc:9}{pr:10}{m['profitable']:>10}%{m['beat_hold']:>9}%{m['median']:>7}%{m['p10']:>8}%{m['p90']:>7}%{m['hold_median']:>8}%{m['trades']:>7}")
