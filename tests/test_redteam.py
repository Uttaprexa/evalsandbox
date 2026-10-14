"""Sandbox regression suite. Every case here is something a model has done, or could do,
to make a wrong answer look right or to break the harness. Runs against Docker by default;
SANDBOX_BACKEND=local uses the (non-isolating) dev backend and skips the isolation tests."""
import os

import pytest

from evalkit.runner import DockerBackend, Limits, LocalBackend, Sample, execute

LOCAL = os.environ.get("SANDBOX_BACKEND") == "local"
BACKEND = LocalBackend() if LOCAL else DockerBackend()
LIM = Limits(cpu_seconds=1, wall_seconds=3, memory_mb=128)
TEST = "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n"
docker_only = pytest.mark.skipif(LOCAL, reason="needs container isolation")


def run(program, **kw):
    return execute(Sample("t", 0, program, TEST, "add"), LIM, BACKEND, **kw)


def test_correct_solution_passes():
    assert run("def add(a, b):\n    return a + b").status == "PASS"


@pytest.mark.parametrize("program,expected", [
    ("def add(a, b):\n    return a - b", "FAIL_ASSERTION"),
    ("def add(a, b)\n    return", "FAIL_SYNTAX"),
    ("def add(a, b):\n    raise ValueError()", "FAIL_EXCEPTION"),
    ("def other(a, b):\n    return a + b", "FAIL_MISSING_ENTRY"),
    ("def add(a, b):\n    while True: pass", "TIMEOUT_CPU"),
    ("import time\ndef add(a, b):\n    time.sleep(60)", "TIMEOUT_WALL"),
    # interactive code must fail fast as a logic error, not hang into a timeout
    ("def add(a, b):\n    return int(input())", "FAIL_EXCEPTION"),
    ("x = input()\ndef add(a, b):\n    return a + b", "FAIL_LOAD"),
])
def test_failure_is_classified(program, expected):
    assert run(program).status == expected


@pytest.mark.parametrize("program", [
    "import sys\nsys.exit(0)\ndef add(a, b): return 0",                           # exit before tests
    "import os\ndef add(a, b):\n    os._exit(0)",                                 # hard exit mid-test
    "class E(int):\n    def __eq__(s, o): return True\ndef add(a, b): return E(0)",  # always-equal
    "def add(a, b):\n    print('@@RESULT@@x@@{\"status\": \"PASS\"}'); return 0",  # fake result
    "import os, signal\ndef add(a, b):\n    os.kill(os.getppid(), signal.SIGKILL)",  # kill the judge
    "import os\ndef add(a, b):\n    os.write(3, b'{\"ok\": 5}\\n' * 100); return 0",  # protocol spam
])
def test_gaming_never_passes(program):
    assert run(program).status != "PASS"


@docker_only
@pytest.mark.parametrize("program", [
    "import socket\ndef add(a, b):\n    socket.create_connection(('1.1.1.1', 80), timeout=1); return a + b",
    "def add(a, b):\n    f = open('/tmp/x', 'w'); f.write('x'); f.close(); return a + b",
    "def add(a, b):\n    f = open('/opt/harness/harness.py', 'a'); f.write('#'); f.close(); return a + b",
])
def test_isolation(program):
    assert run(program).status != "PASS"


@docker_only
def test_memory_bomb_is_resource_failure():
    r = run("def add(a, b):\n    x = bytearray(1 << 30)\n    return a + b")
    assert r.status in ("OOM", "OOM_SUSPECTED", "FAIL_EXCEPTION")  # MemoryError is acceptable


@docker_only
def test_fork_bomb_contained():
    r = run("import os\ndef add(a, b):\n    while True: os.fork()")
    assert r.status != "PASS" and r.category != "infra"


def test_naive_harness_is_fooled():
    """Documents the baseline's weakness -- the motivating result for the oracle design."""
    assert run("import sys\nsys.exit(0)\ndef add(a, b): return 0", mode="naive").status == "PASS"


@pytest.mark.parametrize("cpu", [0.5, 1.5, 2.5])
def test_fractional_cpu_limit_still_classified_as_timeout(cpu):
    """Regression: RLIMIT_CPU is whole seconds; a fractional budget once produced CHILD_CRASH."""
    lim = Limits(cpu_seconds=cpu, wall_seconds=10, memory_mb=128)
    r = execute(Sample("t", 0, "def add(a, b):\n    while True: pass", TEST, "add"), lim, BACKEND)
    assert r.status == "TIMEOUT_CPU"
