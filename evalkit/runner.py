"""Host side: one fresh container per sample, structured result back."""
from __future__ import annotations

import json
import secrets
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .taxonomy import category

IMAGE = "evalsandbox:latest"
HARNESS = Path(__file__).resolve().parent.parent / "sandbox" / "harness.py"


@dataclass(frozen=True)
class Limits:
    cpu_seconds: float = 3.0
    wall_seconds: float = 10.0
    memory_mb: int = 256
    pids: int = 32
    cpus: float = 1.0

    def scaled(self, f: float) -> "Limits":
        return replace(self, cpu_seconds=self.cpu_seconds * f,
                       wall_seconds=self.wall_seconds * f,
                       memory_mb=int(self.memory_mb * f))


@dataclass
class Sample:
    problem_id: str
    sample_id: int
    program: str            # full candidate source (prompt + completion)
    test: str               # trusted test code (HumanEval `check` or MBPP asserts)
    entry_point: str
    prelude: str = ""       # trusted helpers the test needs (e.g. HumanEval prompt)


@dataclass
class ExecResult:
    problem_id: str
    sample_id: int
    status: str
    category: str
    detail: str = ""
    cpu_time: float | None = None
    max_rss_kb: int | None = None
    host_wall: float = 0.0
    limits: dict = field(default_factory=dict)
    mode: str = "oracle"

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


def _payload(s: Sample, lim: Limits, nonce: str, mode: str) -> str:
    return json.dumps({
        "nonce": nonce, "mode": mode, "program": s.program, "test": s.test,
        "entry_point": s.entry_point, "prelude": s.prelude,
        "cpu_seconds": lim.cpu_seconds, "wall_seconds": lim.wall_seconds,
    })


def _parse(stdout: str, nonce: str) -> tuple[str, dict]:
    marker = f"@@RESULT@@{nonce}@@"
    hits = [l[len(marker):] for l in stdout.splitlines() if l.startswith(marker)]
    stray = [l for l in stdout.splitlines() if l.startswith("@@RESULT@@") and not l.startswith(marker)]
    if len(hits) > 1 or stray:
        return "RESULT_FORGERY", {"detail": f"{len(hits)} valid, {len(stray)} stray result lines"}
    if not hits:
        return "HARNESS_LOST", {}
    rec = json.loads(hits[0])
    return rec.pop("status"), rec


class DockerBackend:
    def __init__(self, image: str = IMAGE, grace: float = 5.0):
        self.image, self.grace = image, grace

    def create_args(self, name: str, lim: Limits) -> list[str]:
        return [
            "docker", "create", "-i", "--name", name,
            "--network", "none",                        # no network at all
            "--read-only",                              # immutable rootfs
            "--tmpfs", "/tmp:rw,size=8m,mode=700",      # root-only scratch; candidate can't write
            f"--memory={lim.memory_mb}m", f"--memory-swap={lim.memory_mb}m",  # no swap
            f"--cpus={lim.cpus}", f"--pids-limit={lim.pids}",  # fork bombs stop here
            "--cap-drop=ALL", "--cap-add=SETUID", "--cap-add=SETGID",
            "--security-opt=no-new-privileges",
            "--ipc=none", "--ulimit", "nofile=128:128",
            self.image,
        ]

    def run(self, s: Sample, lim: Limits, mode: str = "oracle") -> tuple[str, dict]:
        name, nonce = f"sbx-{uuid.uuid4().hex[:16]}", secrets.token_hex(16)
        try:
            c = subprocess.run(self.create_args(name, lim), capture_output=True, text=True, timeout=60)
            if c.returncode != 0:
                return "INFRA_ERROR", {"detail": c.stderr[-500:]}
            try:
                p = subprocess.run(["docker", "start", "-a", "-i", name],
                                   input=_payload(s, lim, nonce, mode), capture_output=True,
                                   text=True, timeout=lim.wall_seconds + self.grace)
                status, rec = _parse(p.stdout, nonce)
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "kill", name], capture_output=True)
                status, rec = "HOST_TIMEOUT", {}
            if status in ("HARNESS_LOST", "HOST_TIMEOUT") and self._oom_killed(name):
                status = "OOM"   # cgroup killed the whole container, harness included
            return status, rec
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    @staticmethod
    def _oom_killed(name: str) -> bool:
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.OOMKilled}}", name],
                           capture_output=True, text=True)
        return r.stdout.strip() == "true"


class LocalBackend:
    """DEV ONLY. Runs the harness as a plain subprocess: no network/fs/memory isolation.
    Uid separation only works if you run as root. Never point this at real model output."""

    def run(self, s: Sample, lim: Limits, mode: str = "oracle") -> tuple[str, dict]:
        nonce = secrets.token_hex(16)
        try:
            p = subprocess.run([sys.executable, str(HARNESS)], input=_payload(s, lim, nonce, mode),
                               capture_output=True, text=True, timeout=lim.wall_seconds + 5)
        except subprocess.TimeoutExpired:
            return "HOST_TIMEOUT", {}
        return _parse(p.stdout, nonce)


def execute(s: Sample, lim: Limits = Limits(), backend=None, mode: str = "oracle") -> ExecResult:
    backend = backend or DockerBackend()
    t0 = time.monotonic()
    try:
        status, rec = backend.run(s, lim, mode)
    except Exception as e:  # never let one sample crash a whole eval
        status, rec = "INFRA_ERROR", {"detail": f"{type(e).__name__}: {e}"}
    return ExecResult(
        problem_id=s.problem_id, sample_id=s.sample_id, status=status,
        category=category(status).value, detail=rec.get("detail", ""),
        cpu_time=rec.get("cpu_time"), max_rss_kb=rec.get("max_rss_kb"),
        host_wall=round(time.monotonic() - t0, 3), limits=asdict(lim), mode=mode,
    )


def execute_many(samples, lim: Limits = Limits(), backend=None, concurrency: int = 8,
                 mode: str = "oracle") -> list[ExecResult]:
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(lambda s: execute(s, lim, backend, mode), samples))
