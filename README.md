# evalsandbox

A sandbox for running untrusted, model-generated code in HumanEval/MBPP-style benchmarks, built around one question: **when a benchmark says a sample failed, was the model actually wrong?**

## What makes it different

Most harnesses concatenate `program + tests`, run it, and treat exit code 0 as a pass. That has two failure modes in opposite directions:

- **False passes.** `sys.exit(0)` before the tests run, `os._exit(0)` inside the function, or returning an object whose `__eq__` always returns `True` all score as correct. These are real reward-hacking patterns.
- **False fails.** A correct-but-slow solution, a CPU-contention hiccup, or a harness bug all look identical to a wrong answer.

This project fixes both with an **out-of-process oracle**:

```
host (evalkit/runner.py)
  └─ docker run --network none --read-only --memory --pids-limit --cap-drop ALL ...
       └─ harness.py (trusted parent, drops to uid 1000)
            │  runs the TEST code; candidate function is a proxy
            │        JSON over pipes (plain data only)
            └─ candidate process (uid 65534, RLIMIT_CPU, RLIMIT_FSIZE=0)
                 only executes the model's code and returns values
```

Assertions run in a different process, under a different uid, from the model's code. Return values cross the boundary as plain JSON with exact-type checks, so custom `__eq__` objects are rejected. The candidate can't kill the judge (different uid), can't write the result line (different stdout, unknown per-run nonce), and can't exit its way to a pass. Harness faults subclass `BaseException`, so a test's own `except Exception:` can't swallow a timeout into a pass. PASS is emitted only if the test completed, called the candidate at least once, and no fault occurred. Everything else fails closed.

Every outcome gets a cause (see `evalkit/taxonomy.py`):

| Category | Statuses | Whose fault |
|---|---|---|
| logic | FAIL_ASSERTION, FAIL_EXCEPTION, FAIL_SYNTAX, FAIL_LOAD, FAIL_MISSING_ENTRY, FAIL_UNSERIALIZABLE, FAIL_TEST_ERROR | model |
| resource | TIMEOUT_CPU, TIMEOUT_WALL, OOM, OOM_SUSPECTED | ambiguous |
| integrity | CHILD_EXITED, CHILD_CRASH, PROTOCOL_VIOLATION | model gaming / breaking harness |
| infra | TEST_BROKEN, HARNESS_LOST, HOST_TIMEOUT, INFRA_ERROR, RESULT_FORGERY | us |

CPU timeouts (infinite loop, via `RLIMIT_CPU` → SIGXCPU) are distinguished from wall timeouts (sleeping or blocked), and OOM is read from the cgroup's `memory.events` rather than guessed from exit code 137.

## Quickstart

```bash
docker build -t evalsandbox:latest sandbox/
pip install numpy pytest
pytest tests/                                  # red-team suite against the real sandbox
python run_eval.py canonical --problems HumanEval.jsonl      # every canonical must PASS
python run_eval.py eval  --problems HumanEval.jsonl --samples samples.jsonl --k 1 10
python run_eval.py study --problems HumanEval.jsonl --samples samples.jsonl
```

`--local` / `SANDBOX_BACKEND=local` uses a dev backend with no container. It is for developing the harness only; never run model output with it.

## The research part (`evalkit/attribution.py`)

1. **Three scoring views.** `strict` (every non-pass counts against the model), `verdict_only` (drop ambiguous samples), and `lenient` (give the benefit of the doubt). The spread is the benchmark's noise band; report it alongside pass@k.
2. **Counterfactual re-execution.** Re-run every resource/infra failure at 2×, 4×, 8× limits. The fraction that flips to PASS is the misattribution rate of the strict run, broken down by status.
3. **Limit-sensitivity curve.** pass@k as a function of the limit multiplier (0.25× to 4×). A model whose score moves a lot is being measured partly on speed.
4. **Flake study.** Run canonical solutions repeatedly at concurrency 1, 4, 16. Any sample that changes outcome is pure infrastructure noise; plot flake rate vs. concurrency.
5. **Naive vs. oracle.** Same samples through a typical exit-code harness (`mode="naive"`, same container) and through the oracle. Reports naive false-pass rate and how many naive "fails" were really limit hits.
6. **Calibrated limits.** `calibrate_limits` sets per-problem budgets from the canonical solution's measured CPU time instead of one global timeout.

## Roadmap

1. **Week 1: core.** Build the image, make `pytest tests/` green under Docker, run `canonical` on HumanEval (164/164) and MBPP. Fix any TEST_BROKEN items; those are harness/dataset mismatches.
2. **Week 2: real samples.** Generate n=20 to 200 samples per problem from one or two small open models (or use published sample dumps). Run `eval` and get pass@1/pass@10 with bootstrap CIs. Validate against the official `human-eval` harness: on honest samples, pass rates should match, and any disagreements are findings.
3. **Week 3: experiments.** Run `study`. Make plots: limit-sensitivity curve, flake rate vs. concurrency, confusion matrix naive vs. oracle, flip rate by status.
4. **Week 4: write-up.** A short blog post or report with the headline numbers (for example "X% of timeout failures were correct-but-slow; naive harness over-reports pass@1 by Y points on adversarial samples").

Stretch goals: gVisor (`--runtime=runsc`) as a second isolation layer with an overhead comparison, a warm container pool with a measured cost to isolation, a seccomp profile tighter than Docker's default, and an "adversarial completion" set (prompting a model to pass tests by any means) to stress-test the integrity checks.

## Known limitations (be upfront about these in interviews)

- Arguments and return values must be plain data (int, float, str, bool, None, list, tuple, dict, set, bytes). Generators, numpy arrays, and in-place mutation of arguments aren't supported by the oracle, so those surface as FAIL_UNSERIALIZABLE and should be audited.
- `CHILD_CRASH`/`CHILD_EXITED` after a privilege drop can't be killed by the parent; cleanup relies on container teardown (the harness is PID 1).
- Container-per-sample costs roughly 0.3 to 1 s of overhead, so concurrency matters, and concurrency itself is a noise source. That trade-off is what the flake study measures.
- Docker shares the host kernel. For truly hostile code, add gVisor or a microVM (Firecracker).
