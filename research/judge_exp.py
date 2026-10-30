"""Which way of training the judge actually picks better stocks? (walk-forward, so no peeking)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scanner, judge, train as T
VARIANTS = {"top10%": dict(kind="top", top_k=None), "ret": dict(kind="ret", top_k=None),
            "lambda": dict(kind="lambda", top_k=None), "top10% rerank top40": dict(kind="top", top_k=40),
            "ret rerank top40": dict(kind="ret", top_k=40)}
if __name__ == "__main__":
    pick = sys.argv[1:] or list(VARIANTS)
    for name in os.environ.get("JDS", ",".join(T.DS)).split(","):
        P = T.panel(name); ind = scanner.indicators(P, scanner.PARAMS)
        for v in pick:
            judge.MODE.update(VARIANTS[v]); print(f"== {name} / {v}", flush=True)
            J, _, _ = judge.walk_forward(P, ind, scanner.PARAMS)
            judge.report(P, ind, scanner.PARAMS, J)
            J.to_pickle(os.path.join(T.HERE, f"judge_{name}_{v.replace(' ', '_').replace('%', '')}.pkl"))
