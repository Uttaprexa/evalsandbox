"""
Are NameError failures caused by the standard HumanEval stop rule (cut at the next
top-level `def`), which deletes helper functions the model wrote below its answer?

For every NameError sample: regenerate with the SAME seed but without the "\\ndef " stop,
check the new completion starts with the original one (same code, just longer), and
re-run it in the sandbox.

  python rerun_helpers.py --problems HumanEval.jsonl --samples samples.jsonl --results results.jsonl

Needs Ollama (same model) and Docker running. Writes helper_rerun.json.
"""
import argparse
import json
import re
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from evalkit.runner import DockerBackend, Limits, Sample, execute_many
from run_eval import load_problems

# Same stops as generate_samples.py, minus "\ndef " so helper functions survive.
STOPS_KEEP_DEFS = ["\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert ", '\n"""', "\n```"]


def truncate(text):
    cut = len(text)
    for s in STOPS_KEEP_DEFS:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].rstrip() + "\n"


def generate(host, model, prompt, seed, temperature, max_tokens):
    body = json.dumps({
        "model": model, "prompt": prompt, "raw": True, "stream": False,
        "options": {"temperature": temperature, "top_p": 0.95, "num_predict": max_tokens,
                    "seed": seed, "stop": STOPS_KEEP_DEFS},
    }).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="HumanEval.jsonl")
    ap.add_argument("--samples", default="samples.jsonl")
    ap.add_argument("--results", default="results.jsonl")
    ap.add_argument("--model", default="qwen2.5-coder:1.5b-base")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--host", default="http://localhost:11434")
    a = ap.parse_args()

    probs = load_problems(a.problems)
    raw = {}
    counter = defaultdict(int)
    for line in open(a.samples):
        s = json.loads(line)
        raw[(s["task_id"], counter[s["task_id"]])] = s
        counter[s["task_id"]] += 1

    targets = []
    for r in map(json.loads, open(a.results)):
        if r["status"] == "FAIL_EXCEPTION" and r["detail"].startswith("NameError"):
            m = re.search(r"name '(\w+)'", r["detail"])
            targets.append((r["problem_id"], r["sample_id"], m.group(1) if m else None))
    print(f"{len(targets)} NameError samples to regenerate")

    def regen(t):
        pid, sid, name = t
        s = raw[(pid, sid)]
        text = truncate(generate(a.host, a.model, probs[pid]["prompt"], s["seed"],
                                 a.temperature, a.max_tokens))
        return t, s["completion"], text

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        regenerated = list(ex.map(regen, targets))

    rows, samples = [], []
    for (pid, sid, name), old, new in regenerated:
        same_prefix = new.startswith(old.rstrip())
        defines = bool(name) and re.search(rf"^def {re.escape(name)}\s*\(", new, re.M) is not None
        rows.append({"problem_id": pid, "sample_id": sid, "missing": name,
                     "same_prefix": same_prefix, "defines_missing": defines})
        p = probs[pid]
        samples.append(Sample(pid, sid, p["prompt"] + new, p["test"], p["entry_point"], p["prelude"]))

    results = execute_many(samples, Limits(), DockerBackend(), concurrency=8)
    for row, res in zip(rows, results):
        row["new_status"] = res.status

    controlled = [r for r in rows if r["same_prefix"]]
    helper = [r for r in controlled if r["defines_missing"]]
    flipped = [r for r in helper if r["new_status"] == "PASS"]
    summary = {
        "namerror_samples": len(rows),
        "same_code_prefix": len(controlled),
        "model_defined_missing_name_later": len(helper),
        "of_those_now_pass": len(flipped),
        "pass@1_gain_if_helpers_kept": round(len(flipped) / sum(counter.values()), 5),
        "new_status_of_helper_cases": dict(Counter(r["new_status"] for r in helper)),
        "missing_names_in_helper_cases": dict(Counter(r["missing"] for r in helper).most_common(15)),
        "missing_names_not_defined": dict(Counter(r["missing"] for r in controlled
                                                  if not r["defines_missing"]).most_common(15)),
    }
    json.dump({"summary": summary, "rows": rows}, open("helper_rerun.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
