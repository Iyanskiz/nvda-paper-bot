"""Penny-stock data: the universe (every listed stock priced $1-$5) and intraday bars as numpy arrays."""
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

import bot

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def penny_universe(lo=1.0, hi=5.0):
    """Listed NYSE/Nasdaq/AMEX stocks priced $lo-$hi (OTC 'pink sheets' are excluded: too easy to manipulate)."""
    req = urllib.request.Request("https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true",
                                 headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        rows = json.load(r)["data"]["rows"]
    num = lambda s: float(str(s).replace("$", "").replace(",", "") or 0)
    out = []
    for r in rows:
        sym = r["symbol"].strip()
        if sym.isalpha() and lo <= num(r["lastsale"]) < hi:
            out.append(dict(symbol=sym, name=r["name"][:50], sector=r["sector"] or "", price=num(r["lastsale"])))
    return out


def fetch_arrays(sym, interval="5m", rng="60d"):
    """t, o, h, l, c, v arrays (regular session only for intraday)."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?interval={interval}&range={rng}&includePrePost=false")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        res = json.load(r)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    t = np.array(res.get("timestamp") or [], dtype=np.int64)
    cols = [np.array([np.nan if x is None else x for x in q[k]], dtype=float) for k in ("open", "high", "low", "close", "volume")]
    ok = ~np.isnan(cols[3])
    if interval != "1d":
        mins = np.array([(datetime.fromtimestamp(x, bot.ET).hour * 60 + datetime.fromtimestamp(x, bot.ET).minute) for x in t])
        ok &= (mins >= 570) & (mins < 960)
    return dict(t=t[ok], o=cols[0][ok], h=cols[1][ok], l=cols[2][ok], c=cols[3][ok], v=np.nan_to_num(cols[4][ok]))


def load_intraday(symbols, interval="5m", rng="60d", tag=None):
    tag = tag or datetime.now(bot.ET).date().isoformat()
    path = os.path.join(CACHE, f"penny_{interval}_{rng}_{len(symbols)}_{tag}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["data"].item()

    def get(sym):
        for k in range(3):
            try:
                return sym, fetch_arrays(sym, interval, rng)
            except Exception:
                time.sleep(1 + k)
        return sym, None

    with ThreadPoolExecutor(12) as ex:
        data = {s: d for s, d in ex.map(get, symbols) if d is not None and len(d["t"])}
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, data=np.array(data, dtype=object))
    return data


def by_day(d):
    """Split one stock's intraday arrays into {date: slice}."""
    days = np.array([datetime.fromtimestamp(x, bot.ET).date().isoformat() for x in d["t"]])
    out, start = {}, 0
    for i in range(1, len(days) + 1):
        if i == len(days) or days[i] != days[start]:
            out[days[start]] = slice(start, i)
            start = i
    return out
