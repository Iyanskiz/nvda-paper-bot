"""NVDA paper-trading bot: moving-average crossover with a trend filter and ATR risk control.

Run:  python bot.py            (then open http://localhost:8765)
      python bot.py --symbol AMD --port 8765

Paper trading only. No broker connection, no real orders.
"""
import argparse
import gzip
import json
import math
import re
import sys
import os
import threading
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(HERE, "state.json")
START_CASH = 10_000.0
ET = ZoneInfo("America/New_York")

# "plain" is the bot from the assignment: buy on golden cross, sell on death cross.
# "smart" adds the playbook: trend filter, RSI guard, ATR position sizing, ATR stops.
# "breakout" and "pullback" are the two other families from the playbook, kept for comparison.
PRESETS = {
    "plain": dict(kind="cross", fast=20, slow=50, trend=0, rsi_max=101, risk=None, stop_atr=0, trail_atr=0),
    # risk=0.10 with a 2.5 ATR stop means full size in normal volatility and smaller size
    # when the stock gets wild. Tested across NVDA, AMD, SPY, BTC, AAPL, MSFT (10y daily).
    "smart": dict(kind="cross", fast=20, slow=50, trend=200, rsi_max=75, risk=0.10, stop_atr=2.5, trail_atr=5),
    # buy a close above the 20-bar high in an uptrend, sell a close below the 10-bar low
    "breakout": dict(kind="breakout", fast=20, slow=50, trend=200, rsi_max=101, risk=0.10, stop_atr=2.5, trail_atr=5),
    # live bot default. Built for 5-minute bars: buy a close above the 2-day high (156 bars),
    # sell on a close below the 1-day low (78 bars) or a 12 ATR trailing stop. Slow on purpose:
    # in 4-week simulations it won 77% of bull runs vs 36% for "smart", which overtrades on 5m bars.
    "swing": dict(kind="breakout", fast=20, slow=50, trend=0, rsi_max=101, risk=0.10, stop_atr=5,
                  trail_atr=12, hi_n=156, lo_n=78,
                  # rules from the research that held up in testing (see RULES below)
                  regime=True, earnings_guard=True, vol_target=0.30, vol_n=156, day_loss=0.02),
    # buy a short dip (RSI(2) < 10) in an uptrend, sell the bounce back above the 5-bar average
    "pullback": dict(kind="pullback", fast=20, slow=50, trend=200, rsi_max=10, risk=0.10, stop_atr=3, trail_atr=0),
}
FEE = 0.0005  # commission + slippage per side

# What the bot "knows", where it came from, and whether our tests kept it.
RULES = [
    dict(key="breakout", kept=True, source="Donchian / turtle-style breakouts; tested here",
         rule="Buy when price closes above its 2-day high. Strong stocks make new highs.",
         evidence="4-week simulations: profitable in 72% of bull markets (with all safety rules) vs 38% for the 20/50 crossover."),
    dict(key="exit", kept=True, source="Donchian exits; Rayner Teo on ATR trailing stops",
         rule="Sell on a close below the 1-day low, or if a trailing stop 12x ATR under the peak is hit.",
         evidence="Wider trailing stops beat tight ones in every simulated market type."),
    dict(key="regime", kept=True, source="Meb Faber 10-month/200-day rule; O'Neil's 'M' in CAN SLIM",
         rule="Only invest while NVDA closed above its 200-day average yesterday. If it drops below, pull out.",
         evidence="12 stocks, 10 years: average worst drop -69% to -50%; avoided most of PYPL, VFC and META's crashes."),
    dict(key="earnings", kept=True, source="Common advice on Reddit and YouTube: don't gamble on earnings gaps",
         rule="No new buys on earnings day, and sell before the close if a report is due that night.",
         evidence="Risk rule: NVDA often moves 6-10% overnight on reports. Can't be tested in the simulator."),
    dict(key="vol_target", kept=True, source="Moreira & Muir, 'Volatility-Managed Portfolios'",
         rule="Buy fewer shares when NVDA is unusually jumpy (target 30% yearly volatility).",
         evidence="Smaller bad-case losses in all 4 simulated markets; real worst drop -14.7% to -12%."),
    dict(key="day_loss", kept=True, source="r/Daytrading and r/algotrading risk advice",
         rule="Stop opening trades for the day after losing 2%.",
         evidence="Safety net. Rarely triggers because the bot trades slowly."),
    dict(key="costs", kept=True, source="r/algotrading: model costs, 0.1% per round trip minimum",
         rule="Every trade pays 0.05% each way, so the bot only trades when the move is worth it.",
         evidence="A 20/50 crossover on 5-minute bars lost to costs in 62% of bull simulations."),
    dict(key="skip_open", kept=False, source="Zarattini, Barbon & Aziz, 5-minute opening range breakout",
         rule="Skip the first 30 minutes of the session.",
         evidence="Tested: worse in all 4 simulated markets and on real data, so it's off."),
    dict(key="vwap", kept=False, source="YouTube day-trading educators (only buy above VWAP)",
         rule="Only buy above VWAP.", evidence="Tested: slightly worse everywhere, so it's off."),
    dict(key="above_open", kept=False, source="Gao, Han, Li & Zhou, 'Market intraday momentum'",
         rule="Only buy when price is above today's open.", evidence="Tested: slightly worse everywhere, so it's off."),
    dict(key="template", kept=False, source="Mark Minervini's trend template",
         rule="Price above the 50/150/200-day averages, 200-day rising, near the 52-week high.",
         evidence="As an on/off switch it kept you out too much: 3.6% a year vs 11.7% for the 200-day rule."),
]
REALITY = ("A study of 19,646 Brazilian day traders (Chague, De-Losso & Giovannetti) found 97% of those who kept "
           "at it 300+ days lost money. This bot is built to lose small and stay out of bad markets, not to get rich quick.")


# ---------------------------------------------------------------- data

def fetch_bars(symbol, interval="1d", rng="5y"):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?interval={interval}&range={rng}&includePrePost=false")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        res = json.load(r)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    bars = []
    for i, t in enumerate(res.get("timestamp") or []):
        o, h, l, c, v = (q[k][i] for k in ("open", "high", "low", "close", "volume"))
        if None in (o, h, l, c):
            continue
        if interval != "1d" and not symbol.endswith("-USD") and not _regular_session(t):
            continue  # Yahoo tacks a 16:00 closing-print bar onto the newest day; drop it
        bars.append(dict(t=t, o=o, h=h, l=l, c=c, v=v or 0))
    return bars, res["meta"].get("regularMarketPrice")


def _regular_session(t):
    et = datetime.fromtimestamp(t, ET)
    return (9, 30) <= (et.hour, et.minute) < (16, 0)


# ---------------------------------------------------------------- indicators

def sma(xs, n):
    out, s = [None] * len(xs), 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= n:
            s -= xs[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def rsi(xs, n=14):
    out = [None] * len(xs)
    gain = loss = 0.0
    for i in range(1, len(xs)):
        d = xs[i] - xs[i - 1]
        g, l = max(d, 0), max(-d, 0)
        if i <= n:
            gain += g / n
            loss += l / n
        else:
            gain = (gain * (n - 1) + g) / n
            loss = (loss * (n - 1) + l) / n
        if i >= n:
            out[i] = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    return out


def atr(bars, n=14):
    out = [None] * len(bars)
    a = 0.0
    for i, b in enumerate(bars):
        pc = bars[i - 1]["c"] if i else b["c"]
        tr = max(b["h"] - b["l"], abs(b["h"] - pc), abs(b["l"] - pc))
        a = tr if i == 0 else (a * (n - 1) + tr) / n
        if i >= n:
            out[i] = a
    return out


def channel(bars, n, key, fn):
    """Highest high / lowest low of the n bars before bar i (not including i)."""
    vals = [b[key] for b in bars]
    return [fn(vals[i - n:i]) if i >= n else None for i in range(len(bars))]


def session(bars):
    """Per-bar intraday context: bar number since the open, today's open, and VWAP."""
    bar_no, day_open, vwap, day = [], [], [], []
    k = pv = vol = o = None
    last = None
    for b in bars:
        d = datetime.fromtimestamp(b["t"], ET).date()
        if d != last:
            k, pv, vol, o, last = 0, 0.0, 0.0, b["o"], d
        else:
            k += 1
        typical = (b["h"] + b["l"] + b["c"]) / 3
        pv += typical * b["v"]
        vol += b["v"]
        bar_no.append(k)
        day_open.append(o)
        vwap.append(pv / vol if vol else typical)
        day.append(d.isoformat())
    return bar_no, day_open, vwap, day


def realized_vol(xs, n, per_year):
    """Annualized volatility of log returns over the last n bars."""
    out = [None] * len(xs)
    r = [0.0] + [math.log(xs[i] / xs[i - 1]) for i in range(1, len(xs))]
    s = s2 = 0.0
    for i, x in enumerate(r):
        s += x
        s2 += x * x
        if i >= n:
            s -= r[i - n]
            s2 -= r[i - n] ** 2
        if i >= n:
            var = max(s2 / n - (s / n) ** 2, 0.0)
            out[i] = math.sqrt(var * per_year)
    return out


def indicators(bars, p, regime=None, earnings=None):
    c = [b["c"] for b in bars]
    bar_no, day_open, vwap, day = session(bars)
    return dict(fast=sma(c, p["fast"]), slow=sma(c, p["slow"]),
                trend=sma(c, p["trend"]) if p["trend"] else [None] * len(c),
                rsi=rsi(c), rsi2=rsi(c, 2), sma5=sma(c, 5), atr=atr(bars),
                hi20=channel(bars, p.get("hi_n", 20), "h", max), lo10=channel(bars, p.get("lo_n", 10), "l", min),
                bar_no=bar_no, day_open=day_open, vwap=vwap, day=day,
                rvol=realized_vol(c, p.get("vol_n", 390), p.get("bpy", 252 * 78)),
                regime=regime or [None] * len(c), earnings=earnings or [False] * len(c))


# ---------------------------------------------------------------- daily context

_ctx_cache = {}


def earnings_date(symbol):
    """Next earnings date from Nasdaq (estimates come from Zacks). None if unavailable."""
    try:
        req = urllib.request.Request(f"https://api.nasdaq.com/api/analyst/{symbol}/earnings-date",
                                     headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            text = json.load(r)["data"]["reportText"]
        m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}" if m else None
    except Exception:
        return None


def past_earnings(symbol):
    """Last few report dates (Nasdaq). NVDA reports after the close, so the report day is the risky one."""
    try:
        req = urllib.request.Request(f"https://api.nasdaq.com/api/company/{symbol}/earnings-surprise",
                                     headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            rows = json.load(r)["data"]["earningsSurpriseTable"]["rows"]
        out = []
        for row in rows:
            mo, dy, yr = row["dateReported"].split("/")
            out.append(f"{yr}-{int(mo):02d}-{int(dy):02d}")
        return out
    except Exception:
        return []


NO_CONTEXT = dict(above={}, earnings=None, past_earnings=[], s200=None, close=None)


def daily_context(symbol, max_age=3600):
    """Whether the stock closed above its 200-day average on each day, plus the next earnings date."""
    hit = _ctx_cache.get(symbol)
    if hit and time.time() - hit[0] < max_age:
        return hit[1]
    bars, _ = fetch_bars(symbol, "1d", "2y")
    s200 = sma([b["c"] for b in bars], 200)
    above = {datetime.fromtimestamp(b["t"], ET).date().isoformat(): (s is not None and b["c"] > s)
             for b, s in zip(bars, s200)}
    ctx = dict(above=above, s200=s200[-1], close=bars[-1]["c"], earnings=earnings_date(symbol),
               past_earnings=past_earnings(symbol))
    _ctx_cache[symbol] = (time.time(), ctx)
    return ctx


def context_series(bars, ctx):
    """Regime for each bar = above the 200-day at the PREVIOUS daily close (no peeking at today)."""
    days = sorted(ctx["above"])
    regime, earn = [], []
    j, prev = 0, None
    for b in bars:
        d = datetime.fromtimestamp(b["t"], ET).date().isoformat()
        while j < len(days) and days[j] < d:
            prev = ctx["above"][days[j]]
            j += 1
        regime.append(prev)
        earn.append(d == ctx.get("earnings") or d in ctx.get("past_earnings", ()))
    return regime, earn


# ---------------------------------------------------------------- broker + strategy

def new_account():
    return dict(cash=START_CASH, shares=0.0, entry=None, stop=None, high=None,
                rearm_above=None, trades=[], equity=[])


def _buy(acct, bar, price, shares, reason):
    cost = shares * price * (1 + FEE)
    acct["cash"] -= cost
    acct["shares"] = shares
    acct["entry"] = price
    acct["high"] = bar["h"]
    acct["trades"].append(dict(t=bar["t"], side="BUY", price=round(price, 4),
                               shares=round(shares, 4), reason=reason))


def _sell(acct, bar, price, reason):
    sh = acct["shares"]
    acct["cash"] += sh * price * (1 - FEE)
    pnl = sh * (price * (1 - FEE) - acct["entry"] * (1 + FEE))
    acct["trades"].append(dict(t=bar["t"], side="SELL", price=round(price, 4),
                               shares=round(sh, 4), pnl=round(pnl, 2), reason=reason))
    acct.update(shares=0.0, entry=None, stop=None, high=None)


def _exit_signal(p, ind, i, b):
    if p.get("regime") and ind["regime"][i] is False:
        return "NVDA fell below its 200-day average, pulling out"
    if p.get("earnings_guard") and ind["earnings"][i] and ind["bar_no"][i] >= 74:
        return "earnings report tonight, not holding through it"
    k = p["kind"]
    if k == "cross" and ind["fast"][i] < ind["slow"][i]:
        return "death cross (fast < slow)"
    if k == "breakout" and ind["lo10"][i] is not None and b["c"] < ind["lo10"][i]:
        return f"closed below {p.get('lo_n', 10)}-bar low"
    if k == "pullback" and b["c"] > ind["sma5"][i]:
        return "bounced above 5-bar average"
    return None


def _entry_signal(acct, p, ind, i, b):
    f, s, tr = ind["fast"][i], ind["slow"][i], ind["trend"][i]
    k = p["kind"]
    pf, ps = ind["fast"][i - 1], ind["slow"][i - 1]
    crossed = k == "cross" and pf is not None and ps is not None and pf <= ps and f > s
    if crossed:
        acct["rearm_above"] = None  # a fresh cross re-arms entries after a stop-out
    if p["trend"] and (tr is None or b["c"] <= tr):
        return None  # trend filter: long only above the 200-bar average
    if k == "cross":
        r = ind["rsi"][i]
        if r is not None and r >= p["rsi_max"]:
            return None
        if crossed:
            return "golden cross"
        rearmed = acct["rearm_above"] is None or b["c"] > acct["rearm_above"]
        if p["trend"] and f > s and rearmed:
            return "uptrend resumed above 200 SMA"
    elif k == "breakout":
        if ind["hi20"][i] is not None and b["c"] > ind["hi20"][i]:
            return f"broke above {p.get('hi_n', 20)}-bar high"
    elif k == "pullback":
        r2 = ind["rsi2"][i]
        if r2 is not None and r2 < p["rsi_max"]:
            return f"dip in uptrend (RSI2 {r2:.0f})"
    return None


def _blocked(acct, p, ind, i, b):
    """Filters from the research. Returns the reason an entry is vetoed, or None."""
    if p.get("regime") and ind["regime"][i] is False:
        return "NVDA is below its 200-day average"
    if p.get("earnings_guard") and ind["earnings"][i]:
        return "earnings day"
    if p.get("skip_open") and ind["bar_no"][i] < p["skip_open"]:
        return "too early in the session"
    if p.get("vwap") and b["c"] <= ind["vwap"][i]:
        return "price is below VWAP"
    if p.get("above_open") and b["c"] <= ind["day_open"][i]:
        return "price is below today's open"
    if p.get("day_loss"):
        day = ind["day"][i]
        if acct.get("day") != day:
            acct["day"], acct["day_start"] = day, acct["cash"] + acct["shares"] * b["c"]
        if acct["cash"] < acct["day_start"] * (1 - p["day_loss"]):
            return "daily loss limit reached"
    return None


def step(acct, bars, ind, i, p, fill_price=None):
    """Process closed bar i. fill_price overrides the fill (live mode uses the latest quote)."""
    b = bars[i]
    a = ind["atr"][i]
    if ind["fast"][i] is None or ind["slow"][i] is None or a is None:
        return None
    px = fill_price or b["c"]
    action = None

    if acct["shares"] > 0:
        acct["high"] = max(acct["high"], b["h"])
        if p["trail_atr"]:
            acct["stop"] = max(acct["stop"], acct["high"] - p["trail_atr"] * a)
        why = _exit_signal(p, ind, i, b)
        if acct["stop"] and b["l"] <= acct["stop"]:
            # gap below the stop fills at the open, otherwise at the stop
            # a resting stop order: fills at the stop, or at the open if price gapped through it
            exit_px = min(b["o"], acct["stop"])
            acct["rearm_above"] = acct["high"]
            _sell(acct, b, exit_px, "stop hit")
            action = "SELL"
        elif why:
            acct["rearm_above"] = None
            _sell(acct, b, px, why)
            action = "SELL"
    else:
        why = _entry_signal(acct, p, ind, i, b)
        if why and _blocked(acct, p, ind, i, b):
            why = None
        if why:
            equity = acct["cash"]
            if p["risk"]:
                stop = px - p["stop_atr"] * a
                shares = min(equity * p["risk"] / (px - stop), equity / (px * (1 + FEE)))
                acct["stop"] = stop
            else:
                shares = equity / (px * (1 + FEE))
            if p.get("vol_target") and ind["rvol"][i]:
                # volatility targeting: smaller position when the stock is unusually jumpy
                shares *= min(1.0, p["vol_target"] / ind["rvol"][i])
            _buy(acct, b, px, shares, why)
            acct["rearm_above"] = None
            action = "BUY"

    acct["equity"].append([b["t"], round(acct["cash"] + acct["shares"] * b["c"], 2)])
    return action


# ---------------------------------------------------------------- backtest

def metrics(equity, trades, bars_per_year, n_in_market):
    vals = [e[1] for e in equity]
    if len(vals) < 2:
        return {}
    rets = [vals[i] / vals[i - 1] - 1 for i in range(1, len(vals))]
    mean = sum(rets) / len(rets)
    sd = math.sqrt(sum((x - mean) ** 2 for x in rets) / max(len(rets) - 1, 1))
    peak, mdd = vals[0], 0.0
    for v in vals:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    years = len(vals) / bars_per_year
    sells = [t for t in trades if t["side"] == "SELL"]
    wins = [t["pnl"] for t in sells if t["pnl"] > 0]
    losses = [-t["pnl"] for t in sells if t["pnl"] <= 0]
    return dict(
        total_return=round((vals[-1] / vals[0] - 1) * 100, 1),
        cagr=round(((vals[-1] / vals[0]) ** (1 / years) - 1) * 100, 1) if years > 0 else None,
        max_drawdown=round(mdd * 100, 1),
        sharpe=round(mean / sd * math.sqrt(bars_per_year), 2) if sd else None,
        trades=len(sells),
        win_rate=round(len(wins) / len(sells) * 100, 0) if sells else None,
        profit_factor=round(sum(wins) / sum(losses), 2) if losses and sum(losses) else None,
        exposure=round(n_in_market / len(vals) * 100, 0),
        final_equity=round(vals[-1], 2),
    )


def bars_per_year(symbol, interval="1d"):
    crypto = symbol.endswith("-USD")  # crypto trades 24/7
    if interval == "1d":
        return 365 if crypto else 252
    per_day = {"5m": 288, "1h": 24} if crypto else {"5m": 78, "1h": 7}
    return (365 if crypto else 252) * per_day[interval]


# timeframe -> the range Yahoo allows for it (5m bars only go back 60 days)
RANGES = {"5m": ["60d"], "1h": ["730d"], "1d": ["2y", "5y", "10y"]}


def backtest(bars, preset, warmup=200, per_year=252, ctx=None):
    p = PRESETS[preset]
    ind = indicators(bars, p, *(context_series(bars, ctx) if ctx else ()))
    acct, in_mkt = new_account(), 0
    for i in range(warmup, len(bars)):
        step(acct, bars, ind, i, p)
        in_mkt += acct["shares"] > 0
    return acct, metrics(acct["equity"], acct["trades"], per_year, in_mkt)


def buy_and_hold(bars, warmup=200, per_year=252):
    first = bars[warmup]["c"] * (1 + FEE)
    eq = [[b["t"], round(START_CASH * b["c"] / first, 2)] for b in bars[warmup:]]
    return eq, metrics(eq, [], per_year, len(eq))


# ---------------------------------------------------------------- live paper trader

class LiveBot:
    INTERVAL, RANGE, BAR_SEC = "5m", "5d", 300

    def __init__(self, symbol, preset="swing", feed=fetch_bars, clock=time.time, state_file=STATE_FILE,
                 context=None):
        # feed and clock are swappable so the simulator can drive this exact class
        self.symbol, self.preset = symbol, preset
        self.feed, self.clock, self.state_file = feed, clock, state_file
        self.context = context or safe_context
        self.ctx = NO_CONTEXT
        self.lock = threading.Lock()
        self.bars, self.quote, self.ind, self.log, self.error = [], None, {}, [], None
        self.acct = self._load()

    def _load(self):
        if self.state_file and os.path.exists(self.state_file):
            with open(self.state_file) as f:
                st = json.load(f)
            if st.get("symbol") == self.symbol:
                self.log = st.get("log", [])
                return st["acct"]
        return dict(new_account(), last_t=None)

    def _save(self):
        if not self.state_file:
            return
        with open(self.state_file, "w") as f:
            json.dump(dict(symbol=self.symbol, acct=self.acct, log=self.log[-200:]), f)

    def reset(self):
        with self.lock:
            self.acct, self.log = dict(new_account(), last_t=None), []
            self._save()

    def tick(self):
        bars, quote = self.feed(self.symbol, self.INTERVAL, self.RANGE)
        now = self.clock()
        closed = [b for b in bars if b["t"] + self.BAR_SEC <= now]
        p = PRESETS[self.preset]
        ctx = self.context(self.symbol)
        ind = indicators(closed, p, *context_series(closed, ctx))
        with self.lock:
            self.bars, self.quote, self.ind, self.error, self.ctx = closed, quote, ind, None, ctx
            a = self.acct
            if a["last_t"] is None and closed:
                # start fresh from now; don't trade history retroactively
                a["last_t"] = closed[-1]["t"]
                self._note(f"Bot started on {self.symbol} at {fmt(closed[-1]['t'])}, "
                           f"waiting for new 5-minute bars")
            if "start_price" not in a and closed:
                a["start_price"], a["start_t"] = closed[-1]["c"], closed[-1]["t"]
            for i, b in enumerate(closed):
                if b["t"] <= a["last_t"]:
                    continue
                is_latest = i == len(closed) - 1
                act = step(a, closed, ind, i, p, fill_price=quote if is_latest else None)
                a["last_t"] = b["t"]
                # end-of-day snapshot (the last bar of each day wins): [bot equity, NVDA close]
                a.setdefault("days", {})[ind["day"][i]] = [round(a["cash"] + a["shares"] * b["c"], 2), round(b["c"], 2)]
                if act:
                    t = a["trades"][-1]
                    self._note(f"{act} {t['shares']} @ ${t['price']} ({t['reason']})")
            a["equity"] = a["equity"][-20000:]
            self._save()

    def _note(self, msg):
        self.log.append([int(self.clock()), msg])

    def run_forever(self, every=60):
        while True:
            try:
                self.tick()
            except Exception as e:  # keep running through network blips
                with self.lock:
                    self.error = f"{type(e).__name__}: {e}"
                print(f"{fmt(time.time())} tick failed: {self.error}")
            time.sleep(every)

    def snapshot(self):
        with self.lock:
            n = 300
            bars, ind = self.bars[-n:], {k: v[-n:] for k, v in self.ind.items()}
            a = self.acct
            px = self.quote or (bars[-1]["c"] if bars else 0)
            return dict(
                symbol=self.symbol, preset=self.preset, quote=self.quote, error=self.error,
                bars=bars, fast=ind.get("fast"), slow=ind.get("slow"), trend=ind.get("trend"),
                rsi=ind.get("rsi"),
                cash=round(a["cash"], 2), shares=a["shares"], entry=a["entry"], stop=a["stop"],
                equity_now=round(a["cash"] + a["shares"] * px, 2), start_cash=START_CASH,
                start_price=a.get("start_price"), start_t=a.get("start_t"),
                hold_now=round(START_CASH * px / a["start_price"], 2) if a.get("start_price") else None,
                days=a.get("days", {}),
                equity=a["equity"], trades=a["trades"], log=self.log[-50:],
                updated_at=int(time.time()),
                brain=checklist(a, PRESETS[self.preset], self.ind, self.bars, self.ctx) if self.ind else None,
                params=PRESETS[self.preset],
            )


def checklist(acct, p, ind, bars, ctx, i=-1):
    """The bot's reasoning at bar i, as rows the dashboard can show: (rule, ok, detail)."""
    if not bars:
        return dict(decision="Waiting for data", rows=[])
    i = i % len(bars)
    b = bars[i]
    rows = []
    reg = ind["regime"][i]
    if p.get("regime"):
        s200 = ctx.get("s200")
        close = ctx.get("close")
        detail = (f"Close ${close:,.2f} vs 200-day average ${s200:,.2f}" if s200 and close else "200-day average")
        rows.append(("Market trend: NVDA above its 200-day average", reg is not False,
                     detail if reg is not None else "no daily data yet, allowed"))
    if p.get("earnings_guard"):
        nxt = ctx.get("earnings")
        today = ind["earnings"][i]
        rows.append(("No earnings report today", not today,
                     "Report tonight: no new buys, sell before the close" if today
                     else (f"Next report {nxt}" if nxt else "next date unknown")))
    if p.get("day_loss"):
        lim = acct.get("day_start")
        blocked = bool(lim) and acct.get("day") == ind["day"][i] and acct["cash"] < lim * (1 - p["day_loss"])
        rows.append((f"Daily loss under {p['day_loss']:.0%}", not blocked, "Trading paused for today" if blocked else "OK"))
    if acct["shares"] > 0:
        lo = ind["lo10"][i]
        rows.append(("Holding position", True,
                     f"{acct['shares']:.2f} sh from ${acct['entry']:,.2f}. Sells if price hits the stop "
                     f"${acct['stop']:,.2f} or closes under ${lo:,.2f}" if lo else "holding"))
        decision = "HOLDING: trend intact" if all(r[1] for r in rows) else "PULLING OUT"
    else:
        gates = list(rows)
        hi = ind["hi20"][i]
        need = hi is not None and b["c"] > hi
        rows.append((f"Breakout: close above the {p.get('hi_n', 20)}-bar high", need,
                     f"Needs a close above ${hi:,.2f}, last ${b['c']:,.2f}" if hi else "not enough history"))
        rv = ind["rvol"][i]
        if p.get("vol_target") and rv:
            size = min(1.0, p["vol_target"] / rv)
            rows.append(("Position size from volatility", True,
                         f"NVDA moving at {rv:.0%} a year, would invest {size:.0%} of the account"))
        if all(r[1] for r in rows):
            decision = "READY TO BUY"
        elif any(not r[1] for r in gates):
            decision = "STAYING OUT: " + next(r[2] for r in gates if not r[1])
        else:
            decision = "WAITING for a breakout"
    return dict(decision=decision, rows=[dict(rule=r, ok=o, detail=d) for r, o, d in rows])


def fmt(t):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


# ---------------------------------------------------------------- web server

_bt_cache = {}


def backtest_report(symbol, rng, interval="1d"):
    key = (symbol, rng, interval)
    if key in _bt_cache and time.time() - _bt_cache[key][0] < 600:
        return _bt_cache[key][1]
    bars, _ = fetch_bars(symbol, interval, rng)
    per_year = bars_per_year(symbol, interval)
    out = dict(symbol=symbol, range=rng, interval=interval,
               start=bars[200]["t"], end=bars[-1]["t"], runs={})
    ctx = own_context(bars) if interval == "1d" else safe_context(symbol)
    for name in PRESETS:
        acct, m = backtest(bars, name, per_year=per_year, ctx=ctx)
        out["runs"][name] = dict(metrics=m, equity=acct["equity"], trades=acct["trades"])
    eq, m = buy_and_hold(bars, per_year=per_year)
    out["runs"]["hold"] = dict(metrics=m, equity=eq, trades=[])
    _bt_cache[key] = (time.time(), out)
    return out


REPLAY_RANGE = {"5m": "1mo", "1h": "6mo", "1d": "2y"}


def replay_report(symbol, interval, preset):
    """Everything the dashboard needs to animate the bot bar by bar."""
    bars, _ = fetch_bars(symbol, interval, REPLAY_RANGE[interval])
    p = PRESETS[preset]
    ctx = own_context(bars) if interval == "1d" else safe_context(symbol)
    regime, earn = context_series(bars, ctx)
    ind = indicators(bars, p, regime, earn)
    acct = new_account()
    stops, held = [], []
    for i in range(len(bars)):
        if i >= 200:
            step(acct, bars, ind, i, p)
        held.append(acct["shares"] > 0)
        stops.append(round(acct["stop"], 4) if acct["shares"] > 0 and acct["stop"] else None)
    r = lambda xs: [None if x is None else round(x, 4) for x in xs]
    return dict(symbol=symbol, interval=interval, preset=preset, start=200, params=p,
                bars=bars, fast=r(ind["fast"]), slow=r(ind["slow"]), trend=r(ind["trend"]),
                hi=r(ind["hi20"]), lo=r(ind["lo10"]), regime=regime, earnings=earn, rvol=r(ind["rvol"]),
                stops=stops, held=held, trades=acct["trades"], equity=acct["equity"])


def own_context(bars):
    """Regime from the bars themselves (daily data that already spans years)."""
    s200 = sma([b["c"] for b in bars], 200)
    return dict(NO_CONTEXT, above={datetime.fromtimestamp(b["t"], ET).date().isoformat(): (s is not None and b["c"] > s)
                                   for b, s in zip(bars, s200)})


def safe_context(symbol):
    try:
        return daily_context(symbol)
    except Exception:
        return NO_CONTEXT


def make_handler(bot):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            # big payloads got cut off on some Windows setups, gzip keeps them small
            if "gzip" in self.headers.get("Accept-Encoding", ""):
                data = gzip.compress(data)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/":
                    with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                        self._send(200, f.read(), "text/html; charset=utf-8")
                elif u.path == "/api/live":
                    self._send(200, bot.snapshot())
                elif u.path == "/api/backtest":
                    sym = q.get("symbol", [bot.symbol])[0].upper()
                    iv = q.get("interval", ["1d"])[0]
                    iv = iv if iv in RANGES else "1d"
                    rng = q.get("range", [RANGES[iv][-1]])[0]
                    rng = rng if rng in RANGES[iv] else RANGES[iv][-1]
                    self._send(200, backtest_report(sym, rng, iv))
                elif u.path == "/api/rules":
                    self._send(200, dict(rules=RULES, reality=REALITY, live=bot.preset))
                elif u.path in ("/api/sim", "/api/montecarlo"):
                    import sim  # lazy: sim imports this module
                    scen = q.get("scenario", ["bull"])[0]
                    preset = q.get("strategy", [bot.preset])[0]
                    weeks = min(max(int(q.get("weeks", ["4"])[0]), 1), 8)
                    if scen not in sim.SCENARIOS or preset not in PRESETS:
                        self._send(400, {"error": "bad scenario or strategy"})
                    elif u.path == "/api/sim":
                        seed = int(q.get("seed", ["1"])[0])
                        bars = sim.generate(bot.symbol, weeks * 5, scen, seed)
                        out = sim.run_live(bars, preset, bot.symbol)
                        out.update(scenario=scen, seed=seed, weeks=weeks)
                        self._send(200, out)
                    else:
                        self._send(200, sim.monte_carlo(scen, preset, 200, weeks * 5, bot.symbol))
                elif u.path == "/api/replay":
                    sym = q.get("symbol", [bot.symbol])[0].upper()
                    iv = q.get("interval", ["5m"])[0]
                    preset = q.get("strategy", ["smart"])[0]
                    if iv not in REPLAY_RANGE or preset not in PRESETS:
                        self._send(400, {"error": "bad interval or strategy"})
                    else:
                        self._send(200, replay_report(sym, iv, preset))
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            if urlparse(self.path).path == "/api/reset":
                bot.reset()
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, *a):
            pass

    return H


STRATEGIES = ["swing", "smart", "plain", "breakout", "pullback"]  # the ones the dashboard offers
SIM_WEEKS, SIM_SEEDS = (2, 4, 6, 8), (1, 2, 3, 4, 5)


def _slim(obj):
    """Round bar prices so pre-built files stay small."""
    if isinstance(obj, dict) and "bars" in obj:
        obj = dict(obj, bars=[dict(t=b["t"], o=round(b["o"], 3), h=round(b["h"], 3), l=round(b["l"], 3),
                                   c=round(b["c"], 3), v=b["v"]) for b in obj["bars"]])
    return obj


def _save_json(out, name, data):
    with open(os.path.join(out, name), "w") as f:
        json.dump(_slim(data), f, separators=(",", ":"))


def _build_sim(task):
    """One simulator file plus, for seed 1, the matching Monte Carlo summary. Runs in a worker process."""
    import sim
    symbol, out, sc, wk, st, seed = task
    bars = sim.generate(symbol, wk * 5, sc, seed)
    _save_json(out, f"sim_{sc}_{wk}_{seed}_{st}.json",
               dict(sim.fast_replay(bars, st, symbol), scenario=sc, seed=seed, weeks=wk))
    if seed == SIM_SEEDS[0]:
        _save_json(out, f"mc_{sc}_{wk}_{st}.json", sim.monte_carlo(sc, st, 200, wk * 5, symbol))


def build_static(symbol, out):
    """Pre-compute everything the Simulator and Backtest tabs ask the local server for, as files,
    so the hosted (GitHub Pages) dashboard works without a server. Names match staticUrl() in dashboard.html."""
    import sim
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    for iv, rngs in RANGES.items():
        for rng in rngs:
            _save_json(out, f"backtest_{iv}_{rng}.json", backtest_report(symbol, rng, iv))
    for iv in REPLAY_RANGE:
        for st in STRATEGIES:
            _save_json(out, f"replay_{iv}_{st}.json", replay_report(symbol, iv, st))
    tasks = [(symbol, out, sc, wk, st, seed) for sc in sim.SCENARIOS for wk in SIM_WEEKS
             for st in STRATEGIES for seed in SIM_SEEDS]
    with ProcessPoolExecutor() as pool:  # ~400 simulations: use every CPU core
        list(pool.map(_build_sim, tasks, chunksize=4))
    with open(os.path.join(out, "built.json"), "w") as f:
        json.dump(dict(built_at=int(time.time()), seeds=list(SIM_SEEDS)), f)
    print(f"{fmt(time.time())} static files built in {time.time() - t0:.0f}s")


def run_once(symbol, strategy):
    """One pass for a scheduled job (GitHub Actions): catch up on any finished bars, then publish
    the dashboard as static files in docs/."""
    bot = LiveBot(symbol, strategy)
    bot.tick()
    docs = os.path.join(HERE, "docs")
    os.makedirs(docs, exist_ok=True)
    snap = bot.snapshot()
    snap["updated_at"] = int(time.time())
    with open(os.path.join(docs, "live.json"), "w") as f:
        json.dump(snap, f, separators=(",", ":"))
    with open(os.path.join(docs, "rules.json"), "w") as f:
        json.dump(dict(rules=RULES, reality=REALITY, live=strategy), f)
    with open(os.path.join(HERE, "dashboard.html"), "rb") as src, open(os.path.join(docs, "index.html"), "wb") as dst:
        dst.write(src.read())
    print(f"{fmt(time.time())} tick ok: {len(bot.bars)} bars, equity ${snap['equity_now']:,.2f}, {snap['brain']['decision']}")


def main():
    if sys.stdout is None:  # started with pythonw, in the background
        log = open(os.path.join(HERE, "bot.log"), "a", buffering=1, encoding="utf-8")
        sys.stdout = sys.stderr = log
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NVDA")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--strategy", default="swing", choices=list(PRESETS), help="strategy the live bot trades")
    ap.add_argument("--backtest", action="store_true", help="print a backtest table and exit")
    ap.add_argument("--build-static", action="store_true",
                    help="pre-build Simulator and Backtest data into docs/static for the hosted dashboard")
    ap.add_argument("--tick", action="store_true",
                    help="cloud mode: process new bars once, write docs/ for GitHub Pages, and exit")
    args = ap.parse_args()

    if args.tick:
        run_once(args.symbol.upper(), args.strategy)
        return
    if args.build_static:
        build_static(args.symbol.upper(), os.path.join(HERE, "docs", "static"))
        return

    if args.backtest:
        rep = backtest_report(args.symbol.upper(), "10y")
        print(f"{args.symbol} daily, {fmt(rep['start'])[:10]} to {fmt(rep['end'])[:10]}, "
              f"${START_CASH:,.0f} start\n")
        cols = ["total_return", "cagr", "max_drawdown", "sharpe", "trades", "win_rate",
                "profit_factor", "exposure"]
        print(f"{'':8}" + "".join(f"{c:>14}" for c in cols))
        for name, r in rep["runs"].items():
            print(f"{name:8}" + "".join(f"{str(r['metrics'].get(c)):>14}" for c in cols))
        return

    bot = LiveBot(args.symbol.upper(), args.strategy)
    threading.Thread(target=bot.run_forever, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(bot))
    print(f"{fmt(time.time())} Paper trading {bot.symbol} ({bot.preset}). Dashboard: http://localhost:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
