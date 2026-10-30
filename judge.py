"""The AI judge: a machine-learning model that reads 17 signals for every stock the scanner passes and predicts which
ones will do best over the next month. It's a LightGBM model (gradient-boosted decision trees) trained on what actually
happened to thousands of stocks after each day in the past.

Honesty rule: the judge is always tested walk-forward. On any day of a backtest it has only been trained on months
that were already over, so it can't peek at the future it's being graded on.

    python judge.py build     walk-forward judge scores for the training datasets (research/judge_<ds>.pkl) + report
    python judge.py fit       train the live model on all history -> judge_model.txt (the bot loads this)
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "judge_model.txt")
H = 21            # it predicts the next 21 trading days (about a month)
STEP = 5          # one training snapshot a week
REFIT = 63        # walk-forward: retrain every quarter on everything that's finished by then

FEATURES = ["score", "trend", "r2", "mom12", "mom6", "ret21", "ret5", "high52", "dist_ma", "vol20", "vol_ratio",
            "gap", "dvol", "vol_surge", "beta", "spy_vol", "spy_ret21"]
NAMES = {"score": "trend score", "trend": "trend pace", "r2": "trend smoothness", "mom12": "12-month momentum",
         "mom6": "6-month gain", "ret21": "last month's gain", "ret5": "last week's gain", "high52": "closeness to 52w high",
         "dist_ma": "height above its 100-day average", "vol20": "how wild it swings", "vol_ratio": "swings picking up",
         "gap": "biggest recent jump", "dvol": "dollars traded", "vol_surge": "volume surge", "beta": "moves with the market",
         "spy_vol": "market calm or wild", "spy_ret21": "market's last month"}
MARKET = {"spy_vol", "spy_ret21"}


def candidates(P, ind, p):
    """Same filters as scanner.ranking, for every day at once."""
    C = P["C"]
    ok = (ind["score"] > 0) & (C > ind["ma"]) & (ind["gap"] < p["max_gap"]) & (C >= p["near_high"] * ind["hi"]) \
        & (ind["dvol"] >= p["min_dollar_vol"])
    return ok.drop(columns=["SPY"], errors="ignore")


def raw_features(P, ind):
    C, V = P["C"], P["V"]
    r = C.pct_change(fill_method=None)
    spy = r["SPY"]
    v20 = r.rolling(20).std() * math.sqrt(252)
    cov = r.rolling(60).cov(spy)
    return dict(score=ind["score"], trend=ind["trend"], r2=ind["r2"], mom12=ind["mom12"], mom6=ind["mom6"],
                ret21=C / C.shift(21) - 1, ret5=C / C.shift(5) - 1, high52=ind["high52"], dist_ma=C / ind["ma"] - 1,
                vol20=v20, vol_ratio=v20 / (r.rolling(60).std() * math.sqrt(252)), gap=ind["gap"],
                dvol=np.log(ind["dvol"].clip(lower=1)), vol_surge=V.rolling(5).mean() / V.rolling(60).mean(),
                beta=cov.div(spy.rolling(60).var(), axis=0),
                spy_vol=ind["spy_vol"], spy_ret21=C["SPY"] / C["SPY"].shift(21) - 1)


def table(P, ind, p, rows=None):
    """Long table: one row per (day, candidate stock). Stock signals become 0-1 ranks within that day, so the judge
    compares stocks to each other instead of memorizing price levels from one era."""
    ok = candidates(P, ind, p)
    if rows is not None:
        ok = ok.iloc[rows]
    idx = ok.stack()
    idx = idx[idx].index                       # (date, sym) pairs that pass the filters
    F = raw_features(P, ind)
    cols = {}
    for k in FEATURES:
        f = F[k]
        if k in MARKET:
            cols[k] = f.reindex(idx.get_level_values(0)).to_numpy()
        else:
            s = f.reindex(index=ok.index, columns=ok.columns).where(ok).stack()
            cols[k] = s.groupby(level=0).rank(pct=True).reindex(idx).to_numpy()
    X = pd.DataFrame(cols, index=idx)
    return X


def target(P, ok_idx):
    """What actually happened: buy at the next open, hold H days. Ranked 0-1 within the day (1 = best of that day)."""
    C, O = P["C"], P["O"]
    fwd = (C.shift(-H) / O.shift(-1) - 1).stack(future_stack=True).reindex(ok_idx)
    return fwd.groupby(level=0).rank(pct=True), fwd


def _lgb():
    import lightgbm as lgb
    return lgb


MODE = dict(kind="top", top_k=None)   # kind: "rank" (regress on 0-1 rank), "top" (was it a top-10% stock?),
                                         # "ret" (clipped return), "lambda" (learning-to-rank by day); top_k: only judge the
                                         # top-k by trend score (re-ranking where the bot actually picks)


def _label(y, fwd):
    k = MODE["kind"]
    if k == "top":
        return (y >= 0.9).astype(float)
    if k == "ret":
        return fwd.clip(-0.5, 1.0)
    if k == "lambda":
        return np.floor(y.clip(0, 0.999) * 5)
    return y


def train(X, y, seed=7):
    lgb = _lgb()
    if MODE["kind"] == "lambda":
        g = X.groupby(level=0, sort=False).size().to_numpy()
        params = dict(objective="lambdarank", learning_rate=0.05, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
                      lambdarank_truncation_level=20, eval_at=[7], label_gain=[0, 1, 3, 7, 15], lambda_l2=10.0,
                      verbose=-1, seed=seed, num_threads=4)
        return lgb.train(params, lgb.Dataset(X[FEATURES].to_numpy(), y.to_numpy().astype(int), group=g), num_boost_round=300)
    params = dict(objective="binary" if MODE["kind"] == "top" else "regression", learning_rate=0.05, num_leaves=15, min_data_in_leaf=300, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=seed, num_threads=4)
    return lgb.train(params, lgb.Dataset(X[FEATURES].to_numpy(), y.to_numpy()), num_boost_round=300)


def walk_forward(P, ind, p, min_days=150):
    """Judge scores for every candidate on every day, each from a model trained only on finished history.
    Days before the judge has enough history keep the plain trend-score order (so they match the old rules)."""
    X = table(P, ind, p)
    if MODE["top_k"]:
        sc = ind["score"].stack().reindex(X.index)
        X = X[sc.groupby(level=0).rank(ascending=False, method="first").le(MODE["top_k"]).to_numpy()]
    y, fwd = target(P, X.index)
    y = _label(y, fwd)
    dates = P["C"].index
    pos = pd.Series(np.arange(len(dates)), index=dates)
    di = pos.reindex(X.index.get_level_values(0)).to_numpy()
    labeled = y.notna().to_numpy()
    out = pd.Series(np.nan, index=X.index)
    first = int(np.nanmin(di[np.isfinite(X["mom12"].to_numpy())])) if len(di) else 0
    t0 = first + min_days
    fits = 0
    for t in range(t0, len(dates), REFIT):
        tr = (di % STEP == 0) & (di + H <= t) & labeled
        if tr.sum() < 5000:
            continue
        m = train(X[tr], y[tr])
        te = (di >= t) & (di < t + REFIT)
        if te.any():
            out[te] = m.predict(X[te][FEATURES].to_numpy())
        fits += 1
    J = out.unstack().reindex(index=dates, columns=P["C"].columns)
    J = J.rank(axis=1, pct=True)
    base = ind["score"].where(candidates(P, ind, p).reindex(columns=P["C"].columns, fill_value=False)).rank(axis=1, pct=True)
    ready = J.notna().any(axis=1)
    J[~ready] = base[~ready]
    print(f"judge: {fits} walk-forward fits, ready from {dates[ready.to_numpy().argmax()] if ready.any() else '-'}", flush=True)
    return J, X, y


def report(P, ind, p, J):
    """Is the judge actually better at picking? Compare the top 7 by trend score with the top 7 by judge."""
    ok = candidates(P, ind, p).reindex(columns=P["C"].columns, fill_value=False)
    fwd = P["C"].shift(-H) / P["O"].shift(-1) - 1
    base = ind["score"].where(ok)
    blend = 0.5 * base.rank(axis=1, pct=True) + 0.5 * J.where(ok)
    rows = []
    days = [t for t in range(len(P["C"])) if t % STEP == 0 and J.iloc[t].notna().sum() > 20
            and not J.iloc[t].equals(base.iloc[t].rank(pct=True)) and t + H < len(P["C"])]
    for name, S in (("trend score (now)", base), ("judge", J.where(ok)), ("50/50 blend", blend)):
        top, ic = [], []
        for t in days:
            s, f = S.iloc[t].dropna(), fwd.iloc[t]
            f = f[s.index].dropna()
            s = s[f.index]
            if len(s) < 20:
                continue
            top.append(f[s.nlargest(7).index].mean())
            ic.append(s.rank().corr(f.rank()))
        rows.append((name, np.mean(top) * 100, np.mean(ic), len(top)))
    for name, t7, ic, n in rows:
        print(f"  {name:20} top-7 next-month avg {t7:+5.2f}%   rank correlation {ic:+.3f}   ({n} test days)")
    return rows


def build():
    sys.path.insert(0, HERE)
    import scanner
    sys.path.insert(0, os.path.join(HERE, "research"))
    import train as T
    for name in T.DS:
        P = T.panel(name)
        ind = scanner.indicators(P, scanner.PARAMS)
        J, X, y = walk_forward(P, ind, scanner.PARAMS)
        J.to_pickle(os.path.join(HERE, "research", f"judge_{name}.pkl"))
        print(f"== {name}")
        report(P, ind, scanner.PARAMS, J)


def fit():
    """Train the live judge on every finished month in both training datasets and save it."""
    sys.path.insert(0, HERE)
    import scanner
    sys.path.insert(0, os.path.join(HERE, "research"))
    import train as T
    Xs, ys = [], []
    for name in T.DS:
        P = T.panel(name)
        ind = scanner.indicators(P, scanner.PARAMS)
        X = table(P, ind, scanner.PARAMS)
        y, fwd = target(P, X.index)
        y = _label(y, fwd)
        dates = P["C"].index
        di = pd.Series(np.arange(len(dates)), index=dates).reindex(X.index.get_level_values(0)).to_numpy()
        keep = (di % STEP == 0) & y.notna().to_numpy()
        Xs.append(X[keep]), ys.append(y[keep])
    X, y = pd.concat(Xs), pd.concat(ys)
    m = train(X, y)
    m.save_model(MODEL)
    imp = dict(zip(FEATURES, m.feature_importance("gain")))
    tot = sum(imp.values())
    print(f"saved {MODEL}: trained on {len(X):,} stock-days")
    for k, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        print(f"  {NAMES[k]:32} {v / tot:5.1%}")


_LIVE = {}


def live_scores(P, ind, p, t):
    """Live: the judge's verdict for every candidate on day t. Returns {sym: 0-1 score} plus why for each (top reasons),
    or None if the model file or lightgbm isn't available (the bot then falls back to the plain trend ranking)."""
    try:
        lgb = _lgb()
        if "m" not in _LIVE:
            _LIVE["m"] = lgb.Booster(model_file=MODEL)
    except Exception as e:
        print(f"judge unavailable: {e}")
        return None
    X = table(P, ind, p, rows=[t])
    if X.empty:
        return {}
    X = X.droplevel(0)
    m = _LIVE["m"]
    pred = pd.Series(m.predict(X[FEATURES].to_numpy()), index=X.index)
    contrib = m.predict(X[FEATURES].to_numpy(), pred_contrib=True)[:, :-1]
    score = pred.rank(pct=True)
    out = {}
    for i, s in enumerate(X.index):
        c = contrib[i]
        order = np.argsort(c)
        pros = [NAMES[FEATURES[j]] for j in order[::-1][:2] if c[j] > 0]
        cons = [NAMES[FEATURES[j]] for j in order[:2] if c[j] < 0]
        out[s] = dict(score=round(float(score[s]), 3), pros=pros, cons=cons)
    return out


if __name__ == "__main__":
    {"build": build, "fit": fit}[sys.argv[1]]()
