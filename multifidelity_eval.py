"""Equal-budget comparison of stage-2 search rules, all on one DiskANN graph built with d.

    rerank     D on the top-Q of d                         (paper's baseline)
    full       D on every unseen neighbor                  (paper's method)
    d{k}       D on the k unseen neighbors best under d    (the variant; k >= Deg is `full`)
    rand{k}    D on k random unseen neighbors              (control: is it d, or just a smaller branching factor?)

Scores come from exact q x corpus matmuls; a D-call is counted when the search reads a D score.
Stage 1 is exact top-L under d (the paper sets L=30000 so that DiskANN returns this).

    python multifidelity_eval.py hotpotqa     # run
    python multifidelity_eval.py --report hotpotqa  # re-print tables
    python multifidelity_eval.py --check fiqa # match against greedy_search_quota in bi_metric_search.py
"""
import ast
import bisect
import heapq
import os
import sys
import time

import numpy as np
import torch
from numba import njit, prange

from embed_fast import OUT, load_corpus, load_queries, DATA

BUDGETS = [100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
METHODS = ["rerank", "full", "d1", "d2", "d4", "d8", "d16", "d32", "rand4", "rand8", "rand16"]
K = 10
ROOT = os.path.dirname(OUT.rstrip("/")) or "."


def load_emb(dataset, name):
    parts = f"{OUT}/{dataset}_{name}_parts"
    files = sorted(f for f in os.listdir(parts) if f.endswith(".npy") and "tmp" not in f)
    return np.concatenate([np.load(f"{parts}/{f}") for f in files])


def load_graph(dataset, emb):
    """diskannpy index with the repo's parameters (alpha 1.2, l_build 125, degree 64), as a -1 padded array."""
    d = f"{ROOT}/indices"
    prefix = f"{dataset}-bge-micro-1.2-125-64"
    if not os.path.exists(f"{d}/{prefix}"):
        import diskannpy
        os.makedirs(d, exist_ok=True)
        diskannpy.build_memory_index(emb.astype(np.float32), alpha=1.2, complexity=125, graph_degree=64,
                                     distance_metric="cosine", index_directory=d, num_threads=0,
                                     use_pq_build=False, use_opq=False, index_prefix=prefix)
    raw = np.fromfile(f"{d}/{prefix}", dtype=np.uint32)[6:]  # 24 byte header
    return parse_graph(raw, len(emb))


@njit
def parse_graph(raw, n):
    i = 0
    maxdeg = 0
    for _ in range(n):
        maxdeg = max(maxdeg, raw[i])
        i += raw[i] + 1
    g = np.full((n, maxdeg), -1, dtype=np.int32)
    i = 0
    for v in range(n):
        for j in range(raw[i]):
            g[v, j] = raw[i + 1 + j]
        i += raw[i] + 1
    assert i == len(raw)
    return g


@njit
def stage2(graph, dd, dD, seeds, quota, keep, rand, out, stats, rng=0):
    """dd/dD: distances of this query to every passage. keep=-1: full. Writes top-K ids to out."""
    n, deg = graph.shape
    if rand:
        np.random.seed(rng)  # per (query, budget, method), so results do not depend on thread scheduling
    seen = np.zeros(n, dtype=np.uint8)
    heap = [(np.float64(0), np.int64(0)) for _ in range(0)]
    top = [(np.float64(0), np.int64(0)) for _ in range(0)]  # max-heap of the best K
    used = dcalls = expansions = 0

    def_batch = np.empty(deg, dtype=np.int64)
    key = np.empty(deg, dtype=np.float64)
    ns = min(len(seeds), quota)
    for i in range(ns):
        s = seeds[i]
        seen[s] = 1
        heap.append((np.float64(dD[s]), np.int64(s)))
    heapq.heapify(heap)
    used = ns
    for h in heap:
        heapq.heappush(top, (-h[0], -h[1]))
        if len(top) > K:
            heapq.heappop(top)

    while used < quota and len(heap) > 0:
        p = heapq.heappop(heap)[1]
        expansions += 1
        m = 0
        for j in range(deg):
            x = graph[p, j]
            if x < 0:
                break
            if not seen[x]:
                dup = False
                for t in range(m):
                    if def_batch[t] == x:
                        dup = True
                if not dup:
                    def_batch[m] = x
                    m += 1
        if keep >= 0 and m > keep:
            if rand:
                for t in range(m):
                    key[t] = np.random.random()
            else:
                dcalls += m
                for t in range(m):
                    key[t] = dd[def_batch[t]]
            order = np.argsort(key[:m])
            def_batch[:m] = def_batch[:m][order]
            m = keep
        m = min(m, quota - used)
        used += m
        for t in range(m):
            x = def_batch[t]
            seen[x] = 1
            e = (np.float64(dD[x]), np.int64(x))
            heapq.heappush(heap, e)
            heapq.heappush(top, (-e[0], -e[1]))
            if len(top) > K:
                heapq.heappop(top)

    top.sort()
    r = len(top)
    for i in range(r):
        out[r - 1 - i] = -top[i][1]
    stats[0] = used
    stats[1] = dcalls
    stats[2] = expansions


@njit(parallel=True)
def run_batch(graph, dd, dD, first, budgets, keeps, rands, out, stats, seed0):
    nq = len(dd)
    for job in prange(nq * len(budgets) * len(keeps)):
        q = job // (len(budgets) * len(keeps))
        b = (job // len(keeps)) % len(budgets)
        m = job % len(keeps)
        quota = budgets[b]
        # keep == -2 encodes rerank: seeds fill the whole quota
        ns = quota if keeps[m] == -2 else max(100, quota // 2)
        stage2(graph, dd[q], dD[q], first[q, :ns], quota, keeps[m], rands[m], out[q, b, m], stats[q, b, m],
               seed0 + job)


def method_code(name):
    if name in ("rerank", "full"):
        return (-2 if name == "rerank" else -1), False
    r = name.startswith("rand")
    return int(name[4 if r else 1:]), r


def distances(emb, q, dev, chunk=20_000):
    """1 - cos, float32, in chunks so that float16 storage never means float16 arithmetic."""
    q = torch.from_numpy(q.astype(np.float32)).to(dev)
    q = torch.nn.functional.normalize(q, dim=1)
    out = torch.empty((len(q), len(emb)), dtype=torch.float32)
    for lo in range(0, len(emb), chunk):
        e = emb[lo:lo + chunk].to(dev, non_blocking=True).float()
        e = torch.nn.functional.normalize(e, dim=1)
        out[:, lo:lo + chunk] = (1 - q @ e.T).cpu()
    return out


def load_all(dataset):
    ed = load_emb(dataset, "bge-micro")
    eD = load_emb(dataset, "mistral")
    qd = np.load(f"{OUT}/{dataset}_test_bge-micro_query_embeddings.npy")
    qD = np.load(f"{OUT}/{dataset}_test_mistral_query_embeddings.npy")
    assert len(ed) == len(eD) and len(qd) == len(qD)
    return load_graph(dataset, ed), torch.from_numpy(ed), torch.from_numpy(eD), qd, qD


def main(dataset):
    graph, ed, eD, qd, qD = load_all(dataset)
    print(dataset, graph.shape, "mean degree", (graph >= 0).sum(1).mean(), "queries", len(qd), flush=True)
    budgets = np.array([b for b in BUDGETS if b <= len(ed) // 5])
    keeps, rands = map(np.array, zip(*map(method_code, METHODS)))
    out = np.full((len(qd), len(budgets), len(METHODS), K), -1, dtype=np.int32)
    stats = np.zeros((len(qd), len(budgets), len(METHODS), 3), dtype=np.int32)
    exact = np.zeros((len(qd), K), dtype=np.int32)
    exact_d = np.zeros((len(qd), K), dtype=np.int32)
    dev = "cuda"
    ed = ed.to(dev)
    if torch.cuda.mem_get_info()[0] > eD.numel() * 2 + 12e9:
        eD = eD.to(dev)
    bs = 64
    for lo in range(0, len(qd), bs):
        t = time.time()
        dd = distances(ed, qd[lo:lo + bs], dev)
        dD = distances(eD, qD[lo:lo + bs], dev)
        first = torch.topk(dd, int(budgets.max()), largest=False).indices.numpy().astype(np.int64)
        exact[lo:lo + bs] = torch.topk(dD, K, largest=False).indices.numpy()
        exact_d[lo:lo + bs] = first[:, :K]
        run_batch(graph, dd.numpy(), dD.numpy(), first, budgets, keeps, rands, out[lo:lo + bs], stats[lo:lo + bs],
                  lo * len(budgets) * len(METHODS))
        print(lo, f"{time.time() - t:.1f}s", flush=True)
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    np.savez(f"{ROOT}/results/multifidelity_{dataset}.npz", out=out, stats=stats, exact=exact, exact_d=exact_d,
             budgets=budgets, methods=METHODS)


def report(dataset):
    """Per method and budget: mean NDCG@10, Recall@10 vs exact D, paired difference to `full`, search cost."""
    import pandas as pd
    z = np.load(f"{ROOT}/results/multifidelity_{dataset}.npz")
    qids, pids = load_queries(dataset)[0], load_corpus(dataset)[0]
    qrels = {}
    with open(f"{DATA}/{dataset}/qrels/test.tsv") as f:
        next(f)
        for line in f:
            q, p, r = line.split("\t")
            qrels.setdefault(q, {})[p] = int(r)
    disc = 1 / np.log2(np.arange(2, K + 2))

    def ndcg(i, ids):
        rel = qrels[qids[i]]
        ideal = sorted(rel.values(), reverse=True)[:K]
        return sum(disc[j] * rel.get(pids[p], 0) for j, p in enumerate(ids) if p >= 0) / (disc[:len(ideal)] @ ideal)

    out, stats, exact, methods = z["out"], z["stats"], z["exact"], [str(m) for m in z["methods"]]
    keep = [i for i in range(len(qids)) if sum(qrels[qids[i]].values()) > 0]
    n = len(keep)
    print(f"{dataset}: {n} queries; exact D NDCG {np.mean([ndcg(i, exact[i]) for i in keep]):.4f}, "
          f"exact d NDCG {np.mean([ndcg(i, z['exact_d'][i]) for i in keep]):.4f}")
    boot = np.random.default_rng(0).integers(0, n, (2000, n))
    rows = []
    for b, Q in enumerate(z["budgets"]):
        val = {m: np.array([[ndcg(i, out[i, b, j]), len(set(out[i, b, j]) & set(exact[i])) / K] for i in keep])
               for j, m in enumerate(methods)}
        for j, m in enumerate(methods):
            r = dict(Q=Q, method=m, ndcg=val[m][:, 0].mean(), recall=val[m][:, 1].mean(),
                     used=stats[keep, b, j, 0].mean(), min_used=stats[keep, b, j, 0].min(),
                     d_calls=stats[keep, b, j, 1].mean(), expansions=stats[keep, b, j, 2].mean())
            for c, name in enumerate(("ndcg", "recall")):
                diff = (val[m] - val["full"])[:, c]
                r[f"d_{name}"] = diff.mean()
                r[f"d_{name}_lo"], r[f"d_{name}_hi"] = np.percentile(diff[boot].mean(1), [2.5, 97.5])
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{ROOT}/results/multifidelity_{dataset}.csv", index=False)
    for c in ("ndcg", "recall", "used", "expansions", "d_calls"):
        print(f"\n{c}")
        print(df.pivot(index="Q", columns="method", values=c)[methods].round(4 if c in ("ndcg", "recall") else 0).to_string())
    for c in ("ndcg", "recall"):
        print(f"\n{c} minus full, * = 95% paired bootstrap interval excludes 0")
        df["cell"] = [f"{a:+.4f}{'*' if lo > 0 or hi < 0 else ' '}"
                      for a, lo, hi in zip(df[f"d_{c}"], df[f"d_{c}_lo"], df[f"d_{c}_hi"])]
        print(df.pivot(index="Q", columns="method", values="cell")[[m for m in methods if m != "full"]].to_string())


def check(dataset, nq=20):
    """Same top-10 and same D-call count as the repo's greedy_search_quota, for full and filtered search."""
    graph, ed, eD, qd, qD = load_all(dataset)
    src = open("bi_metric_search.py").read()
    fn = next(x for x in ast.parse(src).body if getattr(x, "name", "") == "greedy_search_quota")
    dd = distances(ed, qd[:nq], "cuda")
    first = torch.topk(dd, 4000, largest=False).indices.numpy().astype(np.int64)
    dd = dd.numpy()
    dD = distances(eD, qD[:nq], "cuda").numpy()
    calls = {"cos": 0, "expensive": 0}

    def dist_all(qid, pids, metric):
        calls[metric] += len(pids)
        return [float((dd if metric == "cos" else dD)[qid, p]) for p in pids]

    ns = dict(graph=[[int(x) for x in r if x >= 0] for r in graph], num_nodes=len(graph), heapq=heapq, bisect=bisect,
              time=time, dist_all=dist_all, set=dict.fromkeys)  # dict.fromkeys: order-preserving dedup
    exec(compile(ast.Module([fn], []), "repo", "exec"), ns)
    for q in range(nq):
        for quota in (100, 150, 500, 2000):
            for keep in (-1, 1, 4, 16, 64, 1000):
                calls.update(cos=0, expensive=0)
                seeds = first[q, :max(100, quota // 2)]
                ref = ns["greedy_search_quota"](q, K, quota, [int(s) for s in seeds], "expensive",
                                                None if keep < 0 else keep)
                out = np.full(K, -1, dtype=np.int32)
                st = np.zeros(3, dtype=np.int32)
                stage2(graph, dd[q], dD[q], seeds, quota, keep, False, out, st)
                assert list(out) == ref, (q, quota, keep, list(out), ref)
                assert (st[0], st[1]) == (calls["expensive"], calls["cos"]), (q, quota, keep, st, calls)
    print("check ok")


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        check(sys.argv[2])
    elif sys.argv[1] == "--report":
        report(sys.argv[2])
    else:
        main(sys.argv[1])
        report(sys.argv[1])
