// Live bot data for the dashboard, always the newest copy.
// The bot force-pushes scanner.json to the `data` branch every 30 seconds. raw.githubusercontent.com serves branch URLs
// from a CDN that can lag by minutes, and the GitHub API allows 60 requests an hour per visitor. Instead we ask git itself
// which commit the branch points at (the smart-HTTP ref list: fresh and not rate-limited like the API) and then fetch the
// file at that exact commit, whose raw URL can never be stale. Vercel's edge caches the answer for 10 seconds, so every
// viewer shares one upstream fetch.
const REPO = "Iyanskiz/nvda-paper-bot";
const FILES = new Set(["scanner.json", "universe.json"]);

async function dataSha() {
  const r = await fetch(`https://github.com/${REPO}.git/info/refs?service=git-upload-pack`, { headers: { "User-Agent": "git/2.45" }, signal: AbortSignal.timeout(5000) });
  if (!r.ok) throw new Error(`refs ${r.status}`);
  const m = (await r.text()).match(/([0-9a-f]{40}) refs\/heads\/data/);
  if (!m) throw new Error("no data branch");
  return m[1];
}

export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  const file = FILES.has(req.query.f) ? req.query.f : "scanner.json";
  try {
    const sha = await dataSha();
    const r = await fetch(`https://raw.githubusercontent.com/${REPO}/${sha}/${file}`, { signal: AbortSignal.timeout(8000) });
    if (!r.ok) throw new Error(`raw ${r.status}`);
    res.setHeader("Cache-Control", "s-maxage=10, stale-while-revalidate=30");
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.setHeader("X-Data-Commit", sha);
    return res.status(200).send(await r.text());
  } catch (e) {
    return res.status(502).json({ error: String(e.message || e).slice(0, 120) });
  }
}
