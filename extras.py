"""
Extra analyses from an existing results.jsonl. No containers or model needed.

  python extras.py --results results.jsonl

Writes analysis/extras.md plus three figures.
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evalkit.passk import pass_at_k


def q(xs, p):
    return float(np.quantile(xs, p)) if len(xs) else float("nan")


def exc_type(detail):
    head = (detail or "").split(":", 1)[0].strip()
    return head if head.isidentifier() else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.jsonl")
    ap.add_argument("--out-dir", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(a.results) if l.strip()]
    rows = [r for r in rows if r.get("mode", "oracle") == "oracle"]
    passed = [r for r in rows if r["status"] == "PASS"]
    lim = rows[0]["limits"]
    md = ["# Extra analyses", ""]

    # 1. Headroom: how close do correct solutions come to the limits?
    cpu = [r["cpu_time"] for r in passed if r.get("cpu_time") is not None]
    rss = [r["max_rss_kb"] / 1024 for r in passed if r.get("max_rss_kb")]
    floor = min(cpu) if cpu else 0.0  # ~ Python interpreter startup inside the container
    md += ["## 1. Resource headroom of correct solutions", "",
           f"Across {len(passed)} passing samples (CPU time includes Python startup, "
           f"~{floor:.3f} s at minimum):", "",
           f"- CPU time: median {q(cpu, .5):.3f} s, 99th percentile {q(cpu, .99):.3f} s, "
           f"max {max(cpu):.3f} s",
           f"- CPU limit {lim['cpu_seconds']:.1f} s, so the slowest correct solution used "
           f"{100 * max(cpu) / lim['cpu_seconds']:.1f}% of it "
           f"(**{lim['cpu_seconds'] / max(cpu):.0f}× headroom**)",
           f"- Peak memory: median {q(rss, .5):.1f} MB, max {max(rss):.1f} MB of "
           f"{lim['memory_mb']} MB (**{lim['memory_mb'] / max(rss):.0f}× headroom**)",
           f"- Single CPU-time readings are noisy under load: compare the max ({max(cpu):.2f} s) "
           "with the limit-sensitivity runs before concluding anything from one outlier.", ""]

    # 2. Isolation overhead: total per-sample time vs time spent running code
    wall = [r["host_wall"] for r in rows if r.get("host_wall")]
    pairs = [(r["host_wall"], r["cpu_time"]) for r in rows
             if r.get("host_wall") and r.get("cpu_time") is not None]
    overhead = [w - c for w, c in pairs]
    md += ["## 2. Cost of isolation", "",
           f"- End-to-end time per sample (create, start, run, delete container): "
           f"median {q(wall, .5) * 1000:.0f} ms, 95th percentile {q(wall, .95) * 1000:.0f} ms",
           f"- CPU time actually spent in candidate code: median {q(cpu, .5) * 1000:.0f} ms",
           f"- So the median sample spends ~{q(overhead, .5) * 1000:.0f} ms on isolation, about "
           f"**{q(wall, .5) / max(q(cpu, .5), 1e-9):.0f}×** the cost of running the code itself.",
           "- Measured with 8 containers in parallel on a MacBook Air (Docker Desktop VM), "
           "so treat it as an upper-end figure.", ""]

    # 3. What kind of exceptions does the model hit?
    ex = Counter(exc_type(r["detail"]) for r in rows if r["status"] == "FAIL_EXCEPTION")
    total_ex = sum(ex.values())
    md += ["## 3. Exception types (FAIL_EXCEPTION)", "",
           "| exception | count | share |", "|---|---|---|"]
    for name, n in ex.most_common(10):
        md.append(f"| {name} | {n} | {100 * n / total_ex:.1f}% |")
    md.append("")
    top = ex.most_common(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([t for t, _ in top], [n for _, n in top], color="#c62828")
    ax.set_xlabel("failed samples")
    ax.set_title("Exceptions raised by model-generated code")
    fig.tight_layout()
    fig.savefig(f"{a.out_dir}/exception_types.png", dpi=150)
    plt.close(fig)

    # 4. pass@k curve and per-problem difficulty
    n, c = defaultdict(int), defaultdict(int)
    for r in rows:
        n[r["problem_id"]] += 1
        c[r["problem_id"]] += r["status"] == "PASS"
    kmax = min(n.values())
    ks = list(range(1, kmax + 1))
    curve = [np.mean([pass_at_k(n[p], c[p], k) for p in n]) for k in ks]
    rates = [c[p] / n[p] for p in n]
    never = sum(c[p] == 0 for p in n)
    always = sum(c[p] == n[p] for p in n)
    md += ["## 4. pass@k curve and problem difficulty", "",
           "- " + ", ".join(f"pass@{k} {curve[k - 1]:.3f}" for k in (1, 2, 5, 10, kmax) if k <= kmax),
           f"- {never} problems ({100 * never / len(n):.0f}%) were never solved in {kmax} tries; "
           f"{always} ({100 * always / len(n):.0f}%) were solved every time; "
           f"{len(n) - never - always} were solved sometimes.",
           "- Hardest problems (0 passes): "
           + ", ".join(sorted((p for p in n if c[p] == 0), key=lambda s: int(s.split("/")[-1]))[:15]),
           ""]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, curve, marker="o", color="#1565c0")
    ax.set_xlabel("k (attempts)")
    ax.set_ylabel("pass@k")
    ax.set_ylim(0, 1)
    ax.set_title("More attempts help a lot")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{a.out_dir}/passk_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rates, bins=np.linspace(0, 1, 11), color="#5e35b1", edgecolor="white")
    ax.set_xlabel(f"fraction of {kmax} samples that passed")
    ax.set_ylabel("problems")
    ax.set_title("Per-problem solve rate")
    fig.tight_layout()
    fig.savefig(f"{a.out_dir}/problem_difficulty.png", dpi=150)
    plt.close(fig)

    md += ["## Figures", "", "![passk_curve](passk_curve.png)",
           "![problem_difficulty](problem_difficulty.png)",
           "![exception_types](exception_types.png)"]
    with open(f"{a.out_dir}/extras.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
