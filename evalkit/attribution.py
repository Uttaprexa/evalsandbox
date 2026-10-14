"""Tools for the research question: how often does a "failure" in a coding benchmark
actually mean the model's code is wrong?

Four experiments:
  1. counterfactual_rerun  - re-run resource/infra failures under relaxed limits; any that
                             flip to PASS were mislabeled by the strict run.
  2. limit_sensitivity     - pass@k as a function of the limit budget (a curve, not a number).
  3. flake_study           - same samples, repeated, at rising host concurrency; measures
                             noise from CPU contention alone.
  4. naive_disagreement    - our oracle vs a typical exit-code harness on identical inputs;
                             quantifies false passes (gaming) and false fails (misattribution).
Plus calibrate_limits: per-problem budgets derived from the canonical solution's cost.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace

from .passk import counts_by_problem, score
from .runner import Limits, Sample, execute, execute_many
from .taxonomy import Category

AMBIGUOUS = {Category.RESOURCE.value, Category.INFRA.value}


def score_views(results, k: int = 1) -> dict:
    """Same results, three scoring policies. The spread is the benchmark's noise band."""
    return {
        # every non-pass counts against the model (what most leaderboards do)
        "strict": score(counts_by_problem(results), k),
        # only samples with an unambiguous verdict count
        "verdict_only": score(counts_by_problem(results, include=lambda r: r.category not in AMBIGUOUS), k),
        # upper bound: give the model the benefit of the doubt on ambiguous failures
        "lenient": score(counts_by_problem(results, is_pass=lambda r: r.passed or r.category in AMBIGUOUS), k),
        "status_counts": dict(Counter(r.status for r in results)),
    }


def counterfactual_rerun(samples, results, base: Limits, factors=(2, 4, 8), backend=None,
                         concurrency: int = 4) -> dict:
    by_key = {(s.problem_id, s.sample_id): s for s in samples}
    todo = [r for r in results if r.category in AMBIGUOUS]
    rows = []
    for r in todo:
        s, row = by_key[(r.problem_id, r.sample_id)], {"key": (r.problem_id, r.sample_id),
                                                       "original": r.status, "trail": []}
        for f in factors:
            rr = execute(s, base.scaled(f), backend)
            row["trail"].append((f, rr.status))
            if rr.passed:
                row["flipped_at"] = f
                break
        rows.append(row)
    flipped = [x for x in rows if "flipped_at" in x]
    by_status = defaultdict(lambda: [0, 0])
    for x in rows:
        by_status[x["original"]][0] += 1
        by_status[x["original"]][1] += int("flipped_at" in x)
    return {
        "ambiguous_failures": len(rows),
        "flipped_to_pass": len(flipped),
        "flip_rate": len(flipped) / len(rows) if rows else 0.0,
        "flip_rate_by_status": {k: {"n": n, "flipped": f} for k, (n, f) in by_status.items()},
        "flip_factor_hist": dict(Counter(x["flipped_at"] for x in flipped)),
        "rows": rows,
    }


def limit_sensitivity(samples, base: Limits, factors=(0.25, 0.5, 1, 2, 4), ks=(1,),
                      backend=None, concurrency: int = 8) -> list[dict]:
    curve = []
    for f in factors:
        res = execute_many(samples, base.scaled(f), backend, concurrency)
        point = {"factor": f, "status_counts": dict(Counter(r.status for r in res))}
        for k in ks:
            point[f"pass@{k}"] = score(counts_by_problem(res), k)["mean"]
        curve.append(point)
    return curve


def flake_study(samples, base: Limits, concurrency_levels=(1, 4, 16), repeats: int = 5,
                backend=None) -> list[dict]:
    """Run ideally-deterministic samples (e.g. canonical solutions) repeatedly.
    Any sample whose outcome differs across repeats is pure infrastructure noise."""
    out = []
    for c in concurrency_levels:
        outcomes = defaultdict(list)
        for _ in range(repeats):
            for r in execute_many(samples, base, backend, concurrency=c):
                outcomes[(r.problem_id, r.sample_id)].append(r.status)
        flaky = {k: v for k, v in outcomes.items() if len(set(v)) > 1}
        all_runs = [st for v in outcomes.values() for st in v]
        out.append({
            "concurrency": c,
            "flaky_samples": len(flaky),
            "flake_rate": len(flaky) / len(outcomes) if outcomes else 0.0,
            "non_pass_rate": sum(st != "PASS" for st in all_runs) / len(all_runs) if all_runs else 0.0,
            "examples": dict(list(flaky.items())[:10]),
        })
    return out


def naive_disagreement(samples, base: Limits, backend=None, concurrency: int = 8) -> dict:
    oracle = execute_many(samples, base, backend, concurrency, mode="oracle")
    naive = execute_many(samples, base, backend, concurrency, mode="naive")
    confusion = Counter()
    false_pass, false_fail = [], []
    for o, n in zip(oracle, naive):
        confusion[(n.status, o.status)] += 1
        if n.passed and not o.passed:
            false_pass.append((o.problem_id, o.sample_id, o.status))
        if not n.passed and o.category in AMBIGUOUS:
            false_fail.append((o.problem_id, o.sample_id, o.status))
    total = len(samples) or 1
    return {
        "naive_false_pass_rate": len(false_pass) / total,   # gaming the naive harness caught
        "naive_ambiguous_fail_rate": len(false_fail) / total,  # naive says "wrong", really a limit
        "confusion": {f"naive={a} | oracle={b}": v for (a, b), v in confusion.items()},
        "false_pass_examples": false_pass[:20],
        "naive_pass@1": score(counts_by_problem(naive), 1)["mean"],
        "oracle_pass@1": score(counts_by_problem(oracle), 1)["mean"],
    }


def calibrate_limits(canonical: list[Sample], base: Limits, multiplier: float = 10.0,
                     floor_s: float = 1.0, backend=None, repeats: int = 3) -> dict[str, Limits]:
    """Per-problem CPU budget = multiplier x the canonical solution's worst observed CPU time.
    A global timeout is either too tight for heavy problems or too loose for light ones."""
    worst = defaultdict(float)
    for _ in range(repeats):
        for r in execute_many(canonical, base, backend, concurrency=1):
            if not r.passed:
                raise RuntimeError(f"canonical solution fails: {r.problem_id} {r.status} {r.detail}")
            worst[r.problem_id] = max(worst[r.problem_id], r.cpu_time or 0.0)
    return {pid: replace(base, cpu_seconds=max(floor_s, t * multiplier),
                         wall_seconds=max(base.wall_seconds, t * multiplier * 3))
            for pid, t in worst.items()}
