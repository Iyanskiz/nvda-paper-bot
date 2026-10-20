"""Company news for the bot: recent headlines (Yahoo, Nasdaq, Google News) and official SEC filings (EDGAR),
turned into a verdict per stock.

What the research says (Chan 2003; Tetlock et al. 2008; Loughran & McDonald 2011; Heston & Sinha 2017; event
studies on offerings, restatements, going-concern warnings, bankruptcy, enforcement): for a weeks-long momentum
strategy, news is mainly a DOWNSIDE filter. Bad news is priced in slowly and keeps drifting; good news is priced in
within hours. So:
  veto     (tier 1)  bankruptcy, delisting notice, restatement, going-concern / late filing, SEC/DOJ fraud action,
                     FDA rejection: never buy, sell a holding at the next run.
  caution  (tier 2)  share offering, guidance cut, short-seller report, abrupt CEO/CFO exit: no new buys; a holding
                     is sold only if the price confirms (falls 5%+ vs the S&P 500 that day).
  positive           only a tiebreaker between similar candidates, never a reason to buy.
A failed fetch is 'unknown', never bad news.
"""
import email.utils
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
SEC_UA = "paper-bot research 221913144+Iyanskiz@users.noreply.github.com"  # SEC asks for a contact in the UA
HEADLINE_TTL = 15 * 60      # news servers refresh every 5-10 min; re-fetching every minute adds nothing
FILING_TTL = 6 * 3600

# ---------------------------------------------------------------- event keywords (phrases, lower case)
TIER1 = {
    "bankruptcy": ["chapter 11", "chapter 7", "files for bankruptcy", "filed for bankruptcy", "bankruptcy protection",
                   "receivership", "debtor-in-possession", "restructuring support agreement"],
    "delisting notice": ["notice of delisting", "delisting notice", "deficiency notice", "minimum bid price",
                         "continued listing standard", "failure to satisfy a continued listing", "to be delisted"],
    "restatement": ["non-reliance", "should no longer be relied upon", "restatement", "will restate", "to restate",
                    "accounting irregularities", "special committee investigation"],
    "going concern": ["going concern", "substantial doubt", "notification of late filing", "auditor resigned",
                      "auditor resigns"],
    "SEC/DOJ action": ["sec investigation", "sec subpoena", "wells notice", "doj probe", "indicted", "fraud charges",
                       "securities fraud", "sec charges", "charged by the sec", "enforcement action"],
    "FDA rejection": ["complete response letter", "fda rejects", "clinical hold", "refuse to file",
                      "did not meet primary endpoint", "failed primary endpoint", "advisory committee voted against"],
}
TIER2 = {
    "share offering": ["prices offering", "priced offering", "pricing of public offering", "proposed public offering",
                       "announces offering", "registered direct", "at-the-market", "secondary offering",
                       "convertible notes offering", "convertible senior notes offering", "public offering of common stock"],
    "guidance cut": ["lowers guidance", "cuts guidance", "cuts outlook", "lowers outlook", "lowers forecast",
                     "cuts forecast", "withdraws guidance", "profit warning", "sees revenue below", "below expectations"],
    "short-seller report": ["short seller report", "short-seller report", "short report", "hindenburg",
                            "muddy waters", "citron research", "grizzly research", "spruce point", "kerrisdale"],
    "abrupt exit": ["resigns effective immediately", "steps down effective immediately", "terminated for cause",
                    "disagreement with the board"],
}
POSITIVE = {
    "beat and raise": ["raises guidance", "raises outlook", "raises full-year", "boosts forecast", "beats estimates",
                       "tops estimates", "record revenue"],
    "upgrade / win": ["upgraded to buy", "upgrades to buy", "upgraded to outperform", "wins contract", "awarded contract"],
}
NEGATORS = re.compile(r"\b(no|not|without|denies|deny|dismissed|resolves|resolved|ends|ended)\b")


def _get(url, accept="*/*", ua=UA, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": accept, "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _rfc822(s):
    try:
        return int(email.utils.parsedate_to_datetime(s).timestamp())
    except Exception:
        return None


# ---------------------------------------------------------------- headlines

def _yahoo_search(sym):
    u = (f"https://query1.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(sym)}"
         "&quotesCount=0&newsCount=20&enableFuzzyQuery=false")
    out = []
    for a in json.loads(_get(u, "application/json")).get("news", []):
        rel = a.get("relatedTickers") or []
        if rel and sym not in rel:
            continue
        out.append(dict(title=a.get("title", ""), published=a.get("providerPublishTime"),
                        source=a.get("publisher", "Yahoo"), url=a.get("link", "")))
    return out


def _yahoo_rss(sym):
    u = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(sym)}&region=US&lang=en-US"
    root = ET.fromstring(_get(u, "application/rss+xml,application/xml"))
    return [dict(title=(it.findtext("title") or "").strip(), published=_rfc822(it.findtext("pubDate") or ""),
                 source="Yahoo Finance", url=(it.findtext("link") or "").strip())
            for it in root.findall("./channel/item")]


_AGO = re.compile(r"(\d+)\s+(minute|min|hour|day)s?\s+ago", re.I)
_UNIT = {"minute": 60, "min": 60, "hour": 3600, "day": 86400}


def _nasdaq(sym, limit=20):
    u = (f"https://api.nasdaq.com/api/news/topic/articlebysymbol?q={urllib.parse.quote(sym.lower())}|stocks"
         f"&offset=0&limit={limit}")
    data = json.loads(_get(u, "application/json")).get("data") or {}
    now, out, tag = time.time(), [], sym.lower() + "|stocks"
    for r in data.get("rows") or []:
        if r.get("primarysymbol") != sym.lower() and tag not in (r.get("related_symbols") or []):
            continue
        ts, m = None, _AGO.search(r.get("ago") or "")
        if m:
            ts = int(now - int(m.group(1)) * _UNIT[m.group(2).lower()])
        else:
            try:
                ts = int(datetime.strptime(r.get("created", ""), "%b %d, %Y").replace(hour=12, tzinfo=timezone.utc).timestamp())
            except ValueError:
                pass
        url = r.get("url") or ""
        out.append(dict(title=(r.get("title") or "").strip(), published=ts, source=r.get("publisher") or "Nasdaq",
                        url="https://www.nasdaq.com" + url if url.startswith("/") else url))
    return out


def _google(sym):
    q = urllib.parse.quote(f'"{sym}" stock when:7d')
    root = ET.fromstring(_get(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"))
    return [dict(title=(it.findtext("title") or "").rsplit(" - ", 1)[0].strip(),
                 published=_rfc822(it.findtext("pubDate") or ""), source=it.findtext("source") or "Google News",
                 url=it.findtext("link") or "") for it in root.findall("./channel/item")[:20]]


def fetch_headlines(sym, max_age_days=14, limit=30):
    """Newest-first, de-duplicated headlines. Never raises; returns (items, ok)."""
    items, ok = [], False
    for fn in (_yahoo_search, _yahoo_rss, _nasdaq):
        try:
            items += fn(sym)
            ok = True
        except Exception:
            pass
    cutoff = time.time() - max_age_days * 86400
    if sum(1 for i in items if (i["published"] or 0) >= cutoff) < 5:
        try:
            items += _google(sym)
            ok = True
        except Exception:
            pass
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["published"] or 0, reverse=True):
        key = re.sub(r"[^a-z0-9]", "", it["title"].lower())[:80]
        if not key or key in seen or (it["published"] and it["published"] < cutoff):
            continue
        seen.add(key)
        out.append(it)
    return out[:limit], ok


# ---------------------------------------------------------------- SEC EDGAR filings (official, structured)

def _cik_map():
    path = os.path.join(CACHE, "sec_tickers.json")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 7 * 86400:
        return json.load(open(path))
    data = json.loads(_get("https://www.sec.gov/files/company_tickers.json", "application/json", SEC_UA, 20))
    m = {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}
    os.makedirs(CACHE, exist_ok=True)
    json.dump(m, open(path, "w"))
    return m


def sec_events(sym, cik_map, days=90):
    """Red-flag filings in the last `days`: 8-K items 1.03 (bankruptcy), 3.01 (delisting), 4.02 (non-reliance),
    NT 10-K/10-Q (late filing) = tier 1; S-1/S-3/424B* offerings = tier 2. Returns (events, ok)."""
    cik = cik_map.get(sym.upper())
    if not cik:
        return [], False
    try:
        d = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", "application/json", SEC_UA, 15))
    except Exception:
        return [], False
    r = d.get("filings", {}).get("recent", {})
    cutoff = datetime.now(timezone.utc).date().toordinal() - days
    out = []
    for form, date, items, acc in zip(r.get("form", []), r.get("filingDate", []), r.get("items", []),
                                      r.get("accessionNumber", [])):
        try:
            if datetime.fromisoformat(date).date().toordinal() < cutoff:
                break  # newest first
        except ValueError:
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"
        items = items or ""
        if form == "8-K":
            for code, label in (("1.03", "bankruptcy"), ("3.01", "delisting notice"), ("4.02", "restatement")):
                if code in items:
                    out.append(dict(tier=1, event=label, form=f"8-K item {code}", date=date, url=url))
        elif form in ("NT 10-K", "NT 10-Q"):
            out.append(dict(tier=1, event="going concern", form=form + " (late filing)", date=date, url=url))
        elif form in ("S-1", "S-3", "S-1/A", "S-3/A") or form.startswith("424B"):
            if (datetime.now(timezone.utc).date().toordinal() - datetime.fromisoformat(date).date().toordinal()) <= 14:
                out.append(dict(tier=2, event="share offering", form=form, date=date, url=url))
    return out, True


# ---------------------------------------------------------------- scoring

def _match(title, groups):
    t = title.lower()
    hits = []
    for label, phrases in groups.items():
        for ph in phrases:
            i = t.find(ph)
            if i >= 0 and not NEGATORS.search(t[max(0, i - 25):i]):
                hits.append(label)
                break
    return hits


def score(sym, headlines, filings, days_fresh=10):
    """Turn headlines + filings into a verdict: veto / caution / positive / neutral (or unknown if nothing loaded)."""
    fresh = time.time() - days_fresh * 86400
    items, t1, t2, pos = [], set(), set(), set()
    for h in headlines:
        f1, f2, fp = _match(h["title"], TIER1), _match(h["title"], TIER2), _match(h["title"], POSITIVE)
        recent = (h["published"] or 0) >= fresh
        if recent:
            t1.update(f1); t2.update(f2); pos.update(fp)
        tone = "negative" if (f1 or f2) else "positive" if fp else "neutral"
        items.append(dict(title=h["title"], published=h["published"], source=h["source"], url=h["url"],
                          tone=tone, flags=f1 + f2 + fp))
    for e in filings:
        (t1 if e["tier"] == 1 else t2).add(e["event"])
    if t1:
        verdict, summary = "veto", "Serious red flag: " + ", ".join(sorted(t1)) + ". Won't buy; sells if held."
    elif t2:
        verdict, summary = "caution", "Warning: " + ", ".join(sorted(t2)) + ". No new buys; sells only if the price drops 5%+ vs the market."
    elif pos:
        verdict, summary = "positive", "Good news: " + ", ".join(sorted(pos)) + " (only used as a tiebreaker)."
    else:
        n = sum(1 for h in headlines if (h["published"] or 0) >= fresh)
        verdict, summary = "neutral", f"{n} recent headline{'s' if n != 1 else ''}, nothing that changes the plan."
    return dict(verdict=verdict, summary=summary, red_flags=sorted(t1), warnings=sorted(t2), good=sorted(pos),
                filings=filings, items=items[:12], t=int(time.time()))


# ---------------------------------------------------------------- cached batch lookups

def _load_cache():
    path = os.path.join(CACHE, "news_cache.json")
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _save_cache(c):
    os.makedirs(CACHE, exist_ok=True)
    json.dump(c, open(os.path.join(CACHE, "news_cache.json"), "w"))


def check(symbols, threads=6):
    """News verdicts for these symbols, cached 15 min (headlines) / 6 h (filings). Never raises."""
    cache = _load_cache()
    now = time.time()
    try:
        ciks = _cik_map()
    except Exception:
        ciks = {}

    def one(sym):
        c = cache.get(sym, {})
        heads, h_ok = (c.get("heads"), True) if now - c.get("heads_t", 0) < HEADLINE_TTL else fetch_headlines(sym)
        if now - c.get("files_t", 0) < FILING_TTL:
            files, f_ok = c.get("files", []), True
        else:
            files, f_ok = sec_events(sym, ciks)
        entry = dict(c)
        if h_ok:
            entry.update(heads=heads, heads_t=c.get("heads_t") if heads is c.get("heads") else now)
        if f_ok:
            entry.update(files=files, files_t=c.get("files_t") if files is c.get("files") else now)
        res = score(sym, entry.get("heads") or [], entry.get("files") or [])
        if not (h_ok or f_ok):
            res.update(verdict="unknown", summary="Couldn't load news right now; trading on price rules only.")
        return sym, entry, res

    out = {}
    with ThreadPoolExecutor(threads) as ex:
        for sym, entry, res in ex.map(one, symbols):
            cache[sym] = entry
            out[sym] = res
    _save_cache(cache)
    return out


if __name__ == "__main__":
    import sys
    for s, r in check(sys.argv[1:] or ["AAPL", "ATRC", "MPC"]).items():
        print(s, r["verdict"], "|", r["summary"], "|", [i["title"][:70] for i in r["items"][:2]])
