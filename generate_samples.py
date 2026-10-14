"""
Generate HumanEval completions with a local Ollama model.

  ollama pull qwen2.5-coder:1.5b-base
  python generate_samples.py --problems HumanEval.jsonl --n 20 --out samples.jsonl

Resumable: rerun the same command and it only generates what's missing.
Uses only the standard library.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# Standard HumanEval stop sequences: the function body ends at the next top-level statement.
STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert ", '\n"""', "\n```"]


def truncate(text: str) -> str:
    cut = len(text)
    for s in STOPS:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].rstrip() + "\n"


def generate(host, model, prompt, temperature, max_tokens, seed):
    body = json.dumps({
        "model": model, "prompt": prompt, "raw": True, "stream": False,
        "options": {"temperature": temperature, "top_p": 0.95, "num_predict": max_tokens,
                    "seed": seed, "stop": STOPS},
    }).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
        # done_reason is "length" when the model hit max_tokens mid-answer
        return d["response"], d.get("done_reason"), d.get("eval_count")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True)
    ap.add_argument("--out", default="samples.jsonl")
    ap.add_argument("--model", default="qwen2.5-coder:1.5b-base")
    ap.add_argument("--n", type=int, default=20, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=None, help="only first N problems (testing)")
    a = ap.parse_args()

    problems = [json.loads(l) for l in open(a.problems)][: a.limit]

    done = Counter()
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip():
                done[json.loads(line)["task_id"]] += 1

    jobs = [(p, i) for p in problems for i in range(done[p["task_id"]], a.n)]
    total = len(jobs)
    if not total:
        print(f"Nothing to do: {a.out} already has {a.n} samples for every problem.")
        return
    print(f"{total} generations to do with {a.model} ({sum(done.values())} already in {a.out})")

    try:
        urllib.request.urlopen(f"{a.host}/api/tags", timeout=5)
    except Exception:
        sys.exit("Can't reach Ollama at localhost:11434. Is the Ollama app running?")

    lock, out = threading.Lock(), open(a.out, "a")
    state = {"done": 0, "errors": 0, "t0": time.time()}

    def work(job):
        p, i = job
        seed = zlib.crc32(f"{p['task_id']}#{i}".encode()) % (2 ** 31)  # reproducible
        try:
            raw, done_reason, n_tokens = generate(a.host, a.model, p["prompt"], a.temperature,
                                                  a.max_tokens, seed)
            completion = truncate(raw)
        except Exception as e:
            with lock:
                state["errors"] += 1
                print(f"\n  error on {p['task_id']}#{i}: {e}")
            return
        with lock:
            out.write(json.dumps({"task_id": p["task_id"], "sample_idx": i, "completion": completion,
                                  "done_reason": done_reason, "n_tokens": n_tokens,
                                  "workers": a.workers, "seed": seed}) + "\n")
            out.flush()
            state["done"] += 1
            d = state["done"]
            rate = d / (time.time() - state["t0"])
            eta = (total - d) / rate / 60 if rate else 0
            print(f"\r  {d}/{total}  {rate:.2f}/s  ~{eta:.0f} min left  ", end="", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, jobs))
    out.close()
    print(f"\nDone. {state['done']} written, {state['errors']} errors. "
          f"Rerun the same command to retry any errors.")


if __name__ == "__main__":
    main()
