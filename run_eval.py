"""
Usage:
  python run_eval.py canonical --problems HumanEval.jsonl          # sanity: all must PASS
  python run_eval.py eval  --problems HumanEval.jsonl --samples samples.jsonl --k 1 10
  python run_eval.py study --problems HumanEval.jsonl --samples samples.jsonl
Add --local to use the UNSAFE dev backend (no Docker).
samples.jsonl lines: {"task_id": "...", "completion": "..."}
"""
import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict

from evalkit import attribution
from evalkit.passk import counts_by_problem, score
from evalkit.runner import DockerBackend, Limits, LocalBackend, Sample, execute_many


def load_problems(path):
    probs = {}
    for line in open(path):
        p = json.loads(line)
        if "test_list" in p:  # MBPP
            entry = re.search(r"assert\s+\(?\s*(\w+)\s*\(", p["test_list"][0]).group(1)
            probs[str(p["task_id"])] = {"prompt": "", "test": "\n".join(p["test_list"]),
                                        "entry_point": entry, "canonical": p["code"],
                                        "prelude": p.get("test_setup_code", "")}
        else:  # HumanEval
            probs[p["task_id"]] = {"prompt": p["prompt"], "test": p["test"],
                                   "entry_point": p["entry_point"],
                                   "canonical": p["prompt"] + p["canonical_solution"],
                                   "prelude": p["prompt"]}
    return probs


def load_samples(path, probs):
    counter, out = defaultdict(int), []
    for line in open(path):
        s = json.loads(line)
        pid = str(s["task_id"])
        p = probs[pid]
        out.append(Sample(pid, counter[pid], p["prompt"] + s["completion"], p["test"],
                          p["entry_point"], p["prelude"]))
        counter[pid] += 1
    return out


def canonical_samples(probs):
    return [Sample(pid, 0, p["canonical"], p["test"], p["entry_point"], p["prelude"])
            for pid, p in probs.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["canonical", "eval", "study"])
    ap.add_argument("--problems", required=True)
    ap.add_argument("--samples")
    ap.add_argument("--k", type=int, nargs="+", default=[1])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--cpu", type=float, default=3.0)
    ap.add_argument("--wall", type=float, default=10.0)
    ap.add_argument("--mem", type=int, default=256)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--local", action="store_true")
    a = ap.parse_args()

    backend = LocalBackend() if a.local else DockerBackend()
    lim = Limits(cpu_seconds=a.cpu, wall_seconds=a.wall, memory_mb=a.mem)
    probs = load_problems(a.problems)

    if a.cmd == "canonical":
        res = execute_many(canonical_samples(probs), lim, backend, a.concurrency)
        bad = [r for r in res if not r.passed]
        print(f"{len(res) - len(bad)}/{len(res)} canonical solutions pass")
        for r in bad:
            print(f"  {r.problem_id}: {r.status} {r.detail[:120]}")
        return

    samples = load_samples(a.samples, probs)
    res = execute_many(samples, lim, backend, a.concurrency)
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(asdict(r)) + "\n")

    if a.cmd == "eval":
        for k in a.k:
            s = score(counts_by_problem(res), k)
            print(f"pass@{k} = {s['mean']:.4f}  95% CI [{s['lo']:.4f}, {s['hi']:.4f}]")
        print(json.dumps(attribution.score_views(res, a.k[0]), indent=2))
        return

    report = {
        "score_views": attribution.score_views(res, 1),
        "counterfactual": {k: v for k, v in attribution.counterfactual_rerun(
            samples, res, lim, backend=backend).items() if k != "rows"},
        "naive_vs_oracle": attribution.naive_disagreement(samples, lim, backend, a.concurrency),
        "limit_sensitivity": attribution.limit_sensitivity(samples, lim, backend=backend,
                                                           concurrency=a.concurrency),
        "flakes_on_canonical": attribution.flake_study(canonical_samples(probs), lim,
                                                       backend=backend),
    }
    with open("study_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report["score_views"], indent=2))
    print("full report -> study_report.json")


if __name__ == "__main__":
    main()
