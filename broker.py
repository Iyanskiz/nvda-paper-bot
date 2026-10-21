"""Optional brokerage link: mirror the bot's portfolio into an Alpaca account.

The bot keeps trading its own $10k paper book exactly as before. When keys are set, every check during market hours
also makes the Alpaca account hold the same stocks in the same proportions (scaled to that account's size).

Setup (GitHub repo > Settings > Secrets and variables > Actions > New repository secret):
    ALPACA_KEY_ID, ALPACA_SECRET_KEY   from alpaca.markets (Paper Trading dashboard > API keys)
Optional repository variables (same page, "Variables" tab):
    ALPACA_MODE      "paper" (default) or "live". Live trades real money from the funded Alpaca account.
    ALPACA_MAX_EXPOSURE   share of the account to invest, default 1.0 (no borrowing at the broker).
Alpaca accounts are funded from a bank by ACH inside Alpaca's own app; the bot never sees bank details.
"""
import json
import os
import time
import urllib.error
import urllib.request

ENDPOINTS = {"paper": "https://paper-api.alpaca.markets", "live": "https://api.alpaca.markets"}
MIN_ORDER = 5.0       # dollars: skip tiny adjustments
DRIFT = 0.03          # only resize a position when it is 3%+ of the account off its target


def configured():
    return bool(os.environ.get("ALPACA_KEY_ID") and os.environ.get("ALPACA_SECRET_KEY"))


def mode():
    return "live" if (os.environ.get("ALPACA_MODE") or "paper").strip().lower() == "live" else "paper"


def _call(method, path, body=None):
    req = urllib.request.Request(ENDPOINTS[mode()] + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"],
                                          "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"],
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:200]
        raise RuntimeError(f"Alpaca {method} {path}: HTTP {e.code} {msg}") from None


def status():
    """Account summary for the dashboard (no secrets)."""
    if not configured():
        return dict(connected=False, mode=None)
    a = _call("GET", "/v2/account")
    pos = _call("GET", "/v2/positions") or []
    return dict(connected=True, mode=mode(), equity=float(a["equity"]), cash=float(a["cash"]),
                buying_power=float(a["buying_power"]), status=a.get("status"),
                positions={p["symbol"]: dict(qty=float(p["qty"]), value=float(p["market_value"]),
                                             pnl=float(p["unrealized_pl"])) for p in pos})


def sync(book, prices, log=print):
    """Make the Alpaca account hold the bot's stocks in the bot's proportions. Sells first, then buys.
    Returns a status dict for the dashboard; never raises."""
    out = dict(connected=configured(), mode=mode() if configured() else None, t=int(time.time()), orders=[], error=None)
    if not configured():
        return out
    try:
        clock = _call("GET", "/v2/clock")
        acct = _call("GET", "/v2/account")
        eq = float(acct["equity"])
        held = {p["symbol"]: p for p in (_call("GET", "/v2/positions") or [])}
        busy = {o["symbol"] for o in (_call("GET", "/v2/orders?status=open") or [])}  # don't double-send while one fills
        bot_eq = book["cash"] + sum(h["shares"] * prices.get(s, h["last"]) for s, h in book["pos"].items())
        gross = sum(h["shares"] * prices.get(s, h["last"]) for s, h in book["pos"].items())
        cap = float(os.environ.get("ALPACA_MAX_EXPOSURE") or "1.0")
        scale = min(1.0, cap * bot_eq / gross) if gross > 0 else 1.0   # the bot may borrow; the broker copy doesn't have to
        target = {s: h["shares"] * prices.get(s, h["last"]) / bot_eq * scale * eq for s, h in book["pos"].items()} if bot_eq > 0 else {}
        out.update(equity=eq, cash=float(acct["cash"]), market_open=bool(clock.get("is_open")))
        if not clock.get("is_open"):
            out["note"] = "Market closed: the brokerage copy updates at the next open."
            return out
        orders = []
        for sym, p in held.items():   # sell what the bot no longer owns, trim what is too big
            val, tgt = float(p["market_value"]), target.get(sym, 0.0)
            if tgt == 0:
                orders.append(dict(symbol=sym, qty=p["qty"], side="sell", type="market", time_in_force="day"))
            elif val - tgt > max(MIN_ORDER, DRIFT * eq):
                orders.append(dict(symbol=sym, notional=round(val - tgt, 2), side="sell", type="market", time_in_force="day"))
        for sym, tgt in target.items():  # buy new holdings, top up ones that are too small
            val = float(held[sym]["market_value"]) if sym in held else 0.0
            if tgt - val > (MIN_ORDER if sym not in held else max(MIN_ORDER, DRIFT * eq)):
                orders.append(dict(symbol=sym, notional=round(tgt - val, 2), side="buy", type="market", time_in_force="day"))
        for o in [o for o in orders if o["symbol"] not in busy]:
            try:
                try:
                    r = _call("POST", "/v2/orders", o)
                except RuntimeError:
                    px = prices.get(o["symbol"])
                    if "notional" not in o or not px or int(float(o["notional"]) / px) < 1:
                        raise
                    o = dict(o, qty=str(int(float(o.pop("notional")) / px)))  # not fractionable: whole shares
                    r = _call("POST", "/v2/orders", o)
                out["orders"].append(dict(symbol=o["symbol"], side=o["side"], amount=o.get("notional"), qty=o.get("qty"),
                                          id=r.get("id"), status=r.get("status")))
                log(f"Brokerage ({mode()}): {o['side'].upper()} {o['symbol']} "
                    + (f"${float(o['notional']):,.2f}" if "notional" in o else f"{o['qty']} shares"))
            except Exception as e:
                out["orders"].append(dict(symbol=o["symbol"], side=o["side"], error=str(e)[:160]))
                log(f"Brokerage order failed for {o['symbol']}: {str(e)[:160]}")
    except Exception as e:
        out["error"] = str(e)[:200]
    return out
