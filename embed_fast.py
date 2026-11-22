"""Faster stand-in for preprocess_embeddings.py (same models, prompts, "title text" format).

Writes float16, L2-normalised embeddings. Several processes on different GPUs can run the same command:
corpus chunks are claimed through lock files holding the owner pid. Stale locks are reclaimed.

CUDA_VISIBLE_DEVICES=1 python embed_fast.py hotpotqa mistral [vllm]
"""
import faulthandler
import json
import os
import sys
import time

import numpy as np

DATA = os.environ.get("BIMETRIC_DATASETS", "./datasets")
OUT = os.environ.get("BIMETRIC_EMBEDDINGS", "./embedding_data")
CHUNK = 100_000

MODELS = {"bge-micro": "TaylorAI/bge-micro-v2", "mistral": "Salesforce/SFR-Embedding-Mistral"}
# copied from preprocess_embeddings.py
TASKS = {
    "fiqa": "Given a financial question, retrieve user replies that best answer the question",
    "hotpotqa": "Given a multi-hop question, retrieve documents that can help answer the question",
    "nq": "Given a question, retrieve Wikipedia passages that answer the question",
    "dbpedia-entity": "Given a query, retrieve relevant entity descriptions from DBPedia",
}


def load_corpus(dataset):
    with open(f"{DATA}/{dataset}/corpus.jsonl") as f:
        rows = [json.loads(line) for line in f]
    return [r["_id"] for r in rows], [r.get("title", "") + " " + r["text"] for r in rows]


def load_queries(dataset, split="test"):
    with open(f"{DATA}/{dataset}/qrels/{split}.tsv") as f:
        next(f)
        keep = {line.split("\t")[0] for line in f}
    with open(f"{DATA}/{dataset}/queries.jsonl") as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r["_id"] in keep]
    return [r["_id"] for r in rows], [r["text"] for r in rows]


def main(dataset, name, engine="st"):
    os.makedirs(OUT, exist_ok=True)
    if engine == "vllm":  # ~1.5x faster than st for mistral, same embeddings (cos > 0.99999 on 6000 HotpotQA passages)
        from transformers import AutoTokenizer
        from vllm import LLM
        tok = AutoTokenizer.from_pretrained(MODELS[name])
        llm = LLM(MODELS[name], runner="pooling", dtype="float16", max_model_len=512, gpu_memory_utilization=0.8,
                  max_num_seqs=512, max_num_batched_tokens=32768, enable_prefix_caching=False)

        def encode(texts):
            ids = tok(list(texts), truncation=True, max_length=512)["input_ids"]
            res = llm.encode([{"prompt_token_ids": i} for i in ids], pooling_task="embed", use_tqdm=False)
            e = np.stack([r.outputs.data.float().cpu().numpy() for r in res])
            return (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float16)
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODELS[name], model_kwargs={"torch_dtype": torch.float16})
        # ponytail: 512-token cap (repo default is 4096 for mistral); only a tiny share of passages is longer
        model.max_seq_length = 512
        bs = 64 if name == "mistral" else 512

        def encode(texts):
            return model.encode(texts, batch_size=bs, normalize_embeddings=True,
                                show_progress_bar=False).astype(np.float16)

    if not os.path.exists(f"{OUT}/{dataset}_test_{name}_query_embeddings.npy"):
        q = load_queries(dataset)[1]
        if name == "mistral":
            q = [f"Instruct: {TASKS[dataset]}\nQuery: {x}" for x in q]
        np.save(f"{OUT}/{dataset}_test_{name}_query_embeddings.npy", encode(q))

    texts = load_corpus(dataset)[1]
    parts = f"{OUT}/{dataset}_{name}_parts"
    os.makedirs(parts, exist_ok=True)
    for lo in range(0, len(texts), CHUNK):
        out = f"{parts}/{lo:09d}.npy"
        try:
            # ponytail: single host only, a lock without .npy whose pid is gone is stale
            if os.path.exists(out + ".lock") and not os.path.exists(out):
                pid = open(out + ".lock").read().strip()
                if pid and not os.path.exists(f"/proc/{pid}"):
                    os.unlink(out + ".lock")
            fd = os.open(out + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
        except FileExistsError:
            continue
        # a chunk takes about 15 min; after 40, print where every thread is stuck and exit
        faulthandler.dump_traceback_later(2400, exit=True)
        t = time.time()
        e = encode(texts[lo:lo + CHUNK])
        assert np.isfinite(e).all()
        np.save(out + ".tmp.npy", e)
        os.rename(out + ".tmp.npy", out)
        faulthandler.cancel_dump_traceback_later()
        print(f"{out} {engine} {len(e) / (time.time() - t):.0f} passages/s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
