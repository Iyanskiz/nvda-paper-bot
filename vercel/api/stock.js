// Stock lookup for the dashboard's search: any ticker worldwide, via Yahoo Finance (browsers can't call Yahoo
// directly because of CORS). GET /api/stock?q=apple  -> matching tickers
//                                GET /api/stock?s=AAPL&range=6mo -> quote, price history and recent headlines
const UA = { "User-Agent": "Mozilla/5.0 (iyanstockbot)" };
const RANGES = { "1d": "5m", "5d": "30m", "1mo": "1d", "6mo": "1d", "1y": "1d", "5y": "1wk" };

async function yahoo(url) {
  const r = await fetch(url, { headers: UA, signal: AbortSignal.timeout(8000) });
  if (!r.ok) throw new Error(`Yahoo ${r.status}`);
  return r.json();
}

// Yahoo's per-ticker RSS feed: the fallback when the search API returns no headlines
async function rssNews(sym) {
  const r = await fetch(`https://feeds.finance.yahoo.com/rss/2.0/headline?s=${encodeURIComponent(sym)}&region=US&lang=en-US`, { headers: UA, signal: AbortSignal.timeout(5000) });
  const xml = await r.text();
  const tag = (block, name) => { const m = block.match(new RegExp(`<${name}>(?:<!\\[CDATA\\[)?([\\s\\S]*?)(?:\\]\\]>)?</${name}>`)); return m ? m[1].trim() : ""; };
  const unesc = t => t.replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">");
  return [...xml.matchAll(/<item>([\s\S]*?)<\/item>/g)].slice(0, 6).map(m => ({
    title: unesc(tag(m[1], "title")), publisher: "Yahoo Finance", link: tag(m[1], "link"),
    t: Math.floor(Date.parse(tag(m[1], "pubDate")) / 1000) || null })).filter(n => n.title && /^https?:/.test(n.link));
}

// What traders are saying about any stock (StockTwits). Earnings dates and analyst targets come from the bot itself:
// Nasdaq refuses requests from Vercel's servers, and a hung upstream must never stall the stock page.
async function crowd(sym) {
  const r = await fetch(`https://api.stocktwits.com/api/2/streams/symbol/${encodeURIComponent(sym)}.json`,
                        { headers: { ...UA, Accept: "application/json" }, signal: AbortSignal.timeout(4000) }).catch(() => null);
  const st = r && r.ok ? await r.json().catch(() => null) : null;
  const msgs = (st && st.messages) || [];
  if (!msgs.length) return {};
  const tag = m => ((m.entities || {}).sentiment || {}).basic;
  const bull = msgs.filter(m => tag(m) === "Bullish").length, bear = msgs.filter(m => tag(m) === "Bearish").length;
  return { buzz: { bull, bear, n: msgs.length, watchers: (st.symbol || {}).watchlist_count,
    mood: bull + bear >= 5 ? (bull / (bull + bear) >= .7 ? "bullish" : bull / (bull + bear) <= .4 ? "bearish" : "mixed") : null } };
}

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  const { q, s } = req.query;
  const range = RANGES[req.query.range] ? req.query.range : "6mo";
  try {
    if (q !== undefined) {
      const text = String(q).slice(0, 40);
      const d = await yahoo(`https://query2.finance.yahoo.com/v1/finance/search?q=${encodeURIComponent(text)}&quotesCount=10&newsCount=0`);
      const quotes = (d.quotes || []).filter(x => x.symbol && ["EQUITY", "ETF"].includes(x.quoteType)).map(x => ({
        sym: x.symbol, name: x.shortname || x.longname || "", exchange: x.exchDisp || x.exchange || "", type: x.quoteType }));
      res.setHeader("Cache-Control", "s-maxage=300");
      return res.status(200).json({ quotes });
    }
    const sym = String(s || "").toUpperCase();
    if (!/^[A-Z0-9.\-^=]{1,15}$/.test(sym)) return res.status(400).json({ error: "bad symbol" });
    const [chart, search, extra] = await Promise.all([
      yahoo(`https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(sym)}?range=${range}&interval=${RANGES[range]}`),
      yahoo(`https://query2.finance.yahoo.com/v1/finance/search?q=${encodeURIComponent(sym)}&quotesCount=1&newsCount=6`).catch(() => ({})),
      req.query.crowd === "0" ? Promise.resolve({}) : crowd(sym).catch(() => ({})),
    ]);
    const r = chart.chart && chart.chart.result && chart.chart.result[0];
    if (!r) return res.status(404).json({ error: "not found" });
    const m = r.meta, ts = r.timestamp || [], cl = (r.indicators.quote[0] || {}).close || [];
    const points = ts.map((t, i) => [t, cl[i]]).filter(p => p[1] != null).map(p => [p[0], Math.round(p[1] * 10000) / 10000]);
    let news = (search.news || []).map(n => ({ title: n.title, publisher: n.publisher, link: n.link, t: n.providerPublishTime }));
    if (!news.length) news = await rssNews(sym).catch(() => []);
    res.setHeader("Cache-Control", "s-maxage=60");
    return res.status(200).json({
      sym, name: m.longName || m.shortName || sym, currency: m.currency, exchange: m.fullExchangeName || m.exchangeName,
      price: m.regularMarketPrice, prev: m.chartPreviousClose ?? m.previousClose, high52: m.fiftyTwoWeekHigh, low52: m.fiftyTwoWeekLow,
      dayHigh: m.regularMarketDayHigh, dayLow: m.regularMarketDayLow, volume: m.regularMarketVolume, range, points, news,
      crowd: extra,
    });
  } catch (e) {
    return res.status(502).json({ error: String(e.message || e).slice(0, 120) });
  }
}
