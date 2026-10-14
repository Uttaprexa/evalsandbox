"""
Turn eval outputs into plots and a findings summary.

  pip install matplotlib
  python analyze.py --samples samples.jsonl --results results.jsonl --report study_report.json

Writes analysis/*.png and analysis/findings.md. --report is optional.
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evalkit.passk import counts_by_problem, score
from evalkit.taxonomy import category


@dataclass
class Row:
    problem_id: str
    sample_id: int
    status: str
    category: str
    done_reason: str | None
    n_tokens: int | None

    @property
    def passed(self):
        return self.status == "PASS"


def load(samples_path, results_path):
    # sample_id = position of the line within its task in samples.jsonl (same rule as run_eval.py)
    meta, counter = {}, defaultdict(int)
    for line in open(samples_path):
        if not line.strip():
            continue
        s = json.loads(line)
        pid = str(s["task_id"])
        meta[(pid, counter[pid])] = (s.get("done_reason"), s.get("n_tokens"))
        counter[pid] += 1
    rows = []
    for line in open(results_path):
        r = json.loads(line)
        if r.get("mode", "oracle") != "oracle":
            continue
        dr, nt = meta.get((r["problem_id"], r["sample_id"]), (None, None))
        rows.append(Row(r["problem_id"], r["sample_id"], r["status"], r["category"], dr, nt))
    return rows


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def plot_status(rows, path):
    counts = Counter(r.status for r in rows)
    order = sorted(counts, key=lambda s: (-counts[s], s))
    colors = {"pass": "#2e7d32", "logic": "#c62828", "resource": "#ef6c00",
              "integrity": "#6a1b9a", "infra": "#455a64"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(order, [counts[s] for s in order], color=[colors[category(s).value] for s in order])
    ax.set_ylabel("samples")
    ax.set_title("Outcome of every sample, by cause")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_truncation(rows, path):
    fails = [r for r in rows if not r.passed]
    by = defaultdict(lambda: [0, 0])
    for r in fails:
        by[r.status][int(r.done_reason == "length")] += 1
    order = sorted(by, key=lambda s: -sum(by[s]))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(order, [by[s][0] for s in order], label="model finished its answer", color="#90a4ae")
    ax.bar(order, [by[s][1] for s in order], bottom=[by[s][0] for s in order],
           label="cut off at token limit", color="#d84315")
    ax.set_ylabel("failed samples")
    ax.set_title("Failures split by whether generation was truncated")
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_line(xs, ys, xlabel, ylabel, title, path, logx=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", color="#1565c0")
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--out-dir", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows = load(a.samples, a.results)
    n = len(rows)
    fails = [r for r in rows if not r.passed]
    trunc = [r for r in rows if r.done_reason == "length"]
    trunc_fail = [r for r in fails if r.done_reason == "length"]
    syntax = [r for r in fails if r.status == "FAIL_SYNTAX"]
    syntax_trunc = [r for r in syntax if r.done_reason == "length"]
    cats = Counter(r.category for r in rows)

    strict = score(counts_by_problem(rows), 1)
    no_trunc = score(counts_by_problem(rows, include=lambda r: r.done_reason != "length"), 1)
    ks = [k for k in (1, 10) if all(nn >= k for nn, _ in counts_by_problem(rows).values())]

    md = ["# Findings", "", f"{n} samples across {len(counts_by_problem(rows))} problems.", "",
          "## Headline scores", ""]
    for k in ks:
        s = score(counts_by_problem(rows), k)
        md.append(f"- pass@{k} = **{s['mean']:.3f}** (95% CI {s['lo']:.3f} to {s['hi']:.3f})")
    md += ["", "## Where failures come from", "",
           "| category | samples | share |", "|---|---|---|"]
    for c in ("pass", "logic", "resource", "integrity", "infra"):
        md.append(f"| {c} | {cats[c]} | {pct(cats[c], n):.1f}% |")

    md += ["", "## Generation-side misattribution (truncation)", "",
           f"- {len(trunc)} samples ({pct(len(trunc), n):.1f}%) hit the token limit before finishing.",
           f"- {len(trunc_fail)} of {len(fails)} failures ({pct(len(trunc_fail), len(fails)):.1f}%) "
           "were truncated generations, not finished wrong answers.",
           f"- {len(syntax_trunc)} of {len(syntax)} syntax errors "
           f"({pct(len(syntax_trunc), len(syntax)):.1f}%) were caused by truncation.",
           f"- pass@1 counting everything: {strict['mean']:.3f}; excluding truncated samples: "
           f"{no_trunc['mean']:.3f} (difference {100 * (no_trunc['mean'] - strict['mean']):+.1f} points).",
           f"- {sum(r.passed for r in trunc)} truncated samples still passed "
           "(the cut-off part was dead code or comments)."]
    if any(r.done_reason is None for r in rows):
        md.append(f"- Note: {sum(r.done_reason is None for r in rows)} samples have no done_reason "
                  "recorded (generated by an older script).")

    plot_status(rows, f"{a.out_dir}/status_breakdown.png")
    plot_truncation(rows, f"{a.out_dir}/truncation.png")
    figs = ["status_breakdown.png", "truncation.png"]

    if a.report and os.path.exists(a.report):
        rep = json.load(open(a.report))
        sv = rep.get("score_views", {})
        if sv:
            md += ["", "## Execution-side noise band", "",
                   f"- strict pass@1 {sv['strict']['mean']:.3f}, verdict-only "
                   f"{sv['verdict_only']['mean']:.3f}, lenient {sv['lenient']['mean']:.3f}."]
        cf = rep.get("counterfactual")
        if cf:
            md += ["", "## Counterfactual re-execution", "",
                   f"- {cf['flipped_to_pass']} of {cf['ambiguous_failures']} resource/infra failures "
                   f"({100 * cf['flip_rate']:.1f}%) passed when re-run with relaxed limits.",
                   f"- By status: {json.dumps(cf['flip_rate_by_status'])}",
                   f"- Limit multiplier needed: {json.dumps(cf['flip_factor_hist'])}"]
        nv = rep.get("naive_vs_oracle")
        if nv:
            md += ["", "## Naive exit-code harness vs. oracle", "",
                   f"- naive pass@1 {nv['naive_pass@1']:.3f} vs oracle {nv['oracle_pass@1']:.3f}.",
                   f"- naive false-pass rate {100 * nv['naive_false_pass_rate']:.2f}% of samples; "
                   f"naive 'fails' that were really limit hits: "
                   f"{100 * nv['naive_ambiguous_fail_rate']:.2f}%.",
                   "", "| naive / oracle | count |", "|---|---|"]
            for key, v in sorted(nv["confusion"].items(), key=lambda kv: -kv[1]):
                md.append(f"| {key} | {v} |")
            if nv.get("false_pass_examples"):
                md += ["", "False-pass examples (inspect these by hand): "
                       + ", ".join(f"{p}#{i} ({st})" for p, i, st in nv["false_pass_examples"][:10])]
        ls = rep.get("limit_sensitivity")
        if ls:
            xs, ys = [p["factor"] for p in ls], [p["pass@1"] for p in ls]
            plot_line(xs, ys, "limit multiplier (CPU, wall, memory)", "pass@1",
                      "Score depends on the resource budget", f"{a.out_dir}/limit_sensitivity.png", logx=True)
            figs.append("limit_sensitivity.png")
            md += ["", "## Limit sensitivity", ""] + \
                  [f"- {x}x limits: pass@1 {y:.3f}" for x, y in zip(xs, ys)]
        fl = rep.get("flakes_on_canonical")
        if fl:
            xs, ys = [p["concurrency"] for p in fl], [100 * p["flake_rate"] for p in fl]
            plot_line(xs, ys, "containers running at once", "% of canonical solutions flaky",
                      "Infrastructure noise vs. host load", f"{a.out_dir}/flake_rate.png", logx=True)
            figs.append("flake_rate.png")
            md += ["", "## Flakiness of known-correct solutions", ""] + \
                  [f"- concurrency {p['concurrency']}: {p['flaky_samples']} flaky, "
                   f"non-pass rate {100 * p['non_pass_rate']:.2f}%" for p in fl]

    md += ["", "## Figures", ""] + [f"![{f}]({f})" for f in figs]
    with open(f"{a.out_dir}/findings.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:40]))
    print(f"\nWrote {a.out_dir}/findings.md and {len(figs)} figures.")


if __name__ == "__main__":
    main()
