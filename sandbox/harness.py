"""
Trusted harness. Runs INSIDE the container as the only process the host talks to.

Key design: the candidate (untrusted) code and the test assertions (trusted) run in
DIFFERENT processes under DIFFERENT uids. The candidate process only exposes its
entry-point function over a JSON pipe; the test code runs here, in the parent, and
calls a proxy. Consequences:
  * sys.exit(0) / os._exit(0) in candidate code can never look like a pass.
  * Returned objects are serialized to plain data, so tricks like a custom class
    whose __eq__ always returns True are rejected (exact-type checks in enc()).
  * The candidate cannot print a fake result line: it never touches this process's
    stdout and doesn't know the per-run nonce.
  * Every failure is classified with a cause instead of collapsing to "failed".

Fail-closed: PASS is emitted only if the test ran to completion, made >= 1 call to
the candidate, and no harness-level fault occurred.
"""
import builtins
import json
import os
import resource
import select
import signal
import subprocess
import sys
import threading
import time

CANDIDATE_UID = 65534   # nobody
PARENT_UID = 1000       # harness drops to this after spawning the child
MAX_LINE = 1 << 20      # 1 MiB per protocol message
STDERR_CAP = 4096

# ---------------------------------------------------------------------------
# Codec shared by parent and child. Exact type checks are deliberate: subclasses
# (e.g. `class X(int): __eq__ = lambda *a: True`) are refused.
# ---------------------------------------------------------------------------
CODEC = r'''
import math
def enc(v, d=0):
    if d > 64: raise TypeError("nesting too deep")
    t = type(v)
    if v is None or t is bool or t is int or t is str: return v
    if t is float:
        if math.isnan(v): return {"__f": "nan"}
        if math.isinf(v): return {"__f": "inf" if v > 0 else "-inf"}
        return v
    if t is list: return [enc(x, d + 1) for x in v]
    if t is tuple: return {"__t": [enc(x, d + 1) for x in v]}
    if t is set or t is frozenset:
        return {"__s": [enc(x, d + 1) for x in v], "fz": t is frozenset}
    if t is dict: return {"__d": [[enc(k, d + 1), enc(x, d + 1)] for k, x in v.items()]}
    if t is bytes: return {"__b": v.hex()}
    raise TypeError(t.__name__)

def dec(v):
    if isinstance(v, list): return [dec(x) for x in v]
    if isinstance(v, dict):
        if "__t" in v: return tuple(dec(x) for x in v["__t"])
        if "__s" in v:
            items = [dec(x) for x in v["__s"]]
            return frozenset(items) if v.get("fz") else set(items)
        if "__d" in v: return {dec(k): dec(x) for k, x in v["__d"]}
        if "__f" in v: return float(v["__f"])
        if "__b" in v: return bytes.fromhex(v["__b"])
        raise ValueError("untagged dict")
    return v
'''

CHILD_MAIN = r'''
import json, os, sys
_proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)                      # candidate prints go to stderr, never the protocol
sys.stdout = sys.stderr
_in = os.fdopen(os.dup(0), "r")    # private copy of the request pipe
_null = os.open(os.devnull, os.O_RDONLY)
os.dup2(_null, 0); os.close(_null) # candidate stdin is empty: input() raises EOFError
sys.stdin = open(os.devnull)       # instead of hanging on (or eating) protocol messages

def _send(o):
    _proto.write(json.dumps(o) + "\n"); _proto.flush()

def _safe_repr(e):
    try: return repr(e)[:500]
    except BaseException: return "<unrepresentable>"

_hdr = json.loads(_in.readline())
_ns = {"__name__": "__main__"}
try:
    _code = compile(_hdr["program"], "<candidate>", "exec")
except (SyntaxError, ValueError) as e:
    _send({"load": "syntax", "msg": _safe_repr(e)}); os._exit(0)
try:
    exec(_code, _ns)
except BaseException as e:
    _send({"load": "error", "type": type(e).__name__, "msg": _safe_repr(e)}); os._exit(0)
_fn = _ns.get(_hdr["entry_point"])
if not callable(_fn):
    _send({"load": "missing"}); os._exit(0)
_send({"load": "ok"})

for _line in _in:
    _req = json.loads(_line)
    try:
        _out = _fn(*dec(_req["a"]), **dec(_req["k"]))
    except BaseException as e:          # includes SystemExit raised by candidate
        _send({"exc": type(e).__name__, "msg": _safe_repr(e)}); continue
    try:
        _send({"ok": enc(_out)})
    except BaseException:
        _send({"unser": type(_out).__name__})
'''

exec(CODEC)  # defines enc/dec in this module


# Harness-level faults subclass BaseException so a test's `except Exception:` can't
# swallow them and accidentally turn a timeout into a pass.
class HarnessFault(BaseException):
    status = "HARNESS_FAULT"

class WallTimeout(HarnessFault):
    status = "TIMEOUT_WALL"

class ChildDied(HarnessFault):
    status = "CHILD_DIED"

class ProtocolViolation(HarnessFault):
    status = "PROTOCOL_VIOLATION"

class Unserializable(HarnessFault):
    status = "FAIL_UNSERIALIZABLE"

class CandidateRaised(Exception):
    """Exception raised inside candidate code, re-raised here for the test to see."""


def _reraise_type(name):
    base = getattr(builtins, str(name), None)
    if (isinstance(base, type) and issubclass(base, Exception)
            and base is not AssertionError):
        try:
            return type("Candidate" + base.__name__, (CandidateRaised, base), {})
        except TypeError:
            pass
    return CandidateRaised


class Channel:
    def __init__(self, proc, deadline):
        self.p, self.deadline, self.buf = proc, deadline, b""
        self.calls, self.fault = 0, None

    def send(self, obj):
        try:
            self.p.stdin.write((json.dumps(obj) + "\n").encode())
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            raise self._fault(ChildDied("stdin closed"))

    def recv(self):
        fd = self.p.stdout.fileno()
        while b"\n" not in self.buf:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise self._fault(WallTimeout())
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise self._fault(WallTimeout())
            chunk = os.read(fd, 65536)
            if not chunk:
                raise self._fault(ChildDied("eof"))
            self.buf += chunk
            if len(self.buf) > MAX_LINE:
                raise self._fault(ProtocolViolation("oversized message"))
        line, self.buf = self.buf.split(b"\n", 1)
        try:
            return json.loads(line)
        except ValueError:
            raise self._fault(ProtocolViolation("malformed message"))

    def _fault(self, f):
        # remember the first fault even if the test code catches it with a bare except
        if self.fault is None:
            self.fault = f
        return f

    def proxy(self):
        def candidate(*args, **kwargs):
            self.calls += 1
            self.send({"a": enc(list(args)), "k": enc(kwargs)})
            resp = self.recv()
            if "ok" in resp:
                try:
                    return dec(resp["ok"])
                except Exception:
                    raise self._fault(ProtocolViolation("undecodable value"))
            if "exc" in resp:
                raise _reraise_type(resp["exc"])(f'{resp["exc"]}: {resp.get("msg", "")}')
            if "unser" in resp:
                raise self._fault(Unserializable(str(resp["unser"])[:100]))
            raise self._fault(ProtocolViolation("unexpected message"))
        return candidate


def read_oom_kills():
    for path, key in (("/sys/fs/cgroup/memory.events", "oom_kill"),
                      ("/sys/fs/cgroup/memory/memory.oom_control", "oom_kill")):
        try:
            with open(path) as f:
                for line in f:
                    k, _, v = line.partition(" ")
                    if k == key:
                        return int(v)
        except OSError:
            continue
    return None


def spawn_child(payload, child_src, drop_privs):
    cpu = max(1, int(payload["cpu_seconds"]))

    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()

    kw = dict(user=CANDIDATE_UID, group=CANDIDATE_UID, extra_groups=[]) if drop_privs else {}
    return subprocess.Popen(
        [sys.executable, "-I", "-c", child_src],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0, close_fds=True, cwd="/", preexec_fn=limits,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        **kw,
    )


def drain(stream, sink):
    total = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            return
        if total < STDERR_CAP:
            sink.append(chunk[: STDERR_CAP - total])
        total += len(chunk)


def classify_death(proc, oom_before, cpu_limit):
    try:
        rc = proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        return "CHILD_CRASH"
    oom_after = read_oom_kills()
    if oom_before is not None and oom_after is not None and oom_after > oom_before:
        return "OOM"
    if rc in (-signal.SIGXCPU, -signal.SIGKILL):
        used = resource.getrusage(resource.RUSAGE_CHILDREN)
        if used.ru_utime + used.ru_stime >= cpu_limit * 0.9:
            return "TIMEOUT_CPU"
    if rc == -signal.SIGKILL:
        return "OOM_SUSPECTED"   # killed with no cgroup counter available
    if rc is not None and rc >= 0:
        return "CHILD_EXITED"    # candidate called exit()/os._exit() mid-test
    return "CHILD_CRASH"


def finish(proc):
    """Close the pipe so a healthy child exits. We deliberately do NOT kill: after the
    privilege drop parent and child have different uids, so neither can signal the
    other. A hung child dies when this process (PID 1 of the container) exits and the
    container is torn down."""
    try:
        proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass


def run_oracle(payload, drop_privs):
    oom_before = read_oom_kills()
    deadline = time.monotonic() + payload["wall_seconds"]
    proc = spawn_child(payload, CODEC + CHILD_MAIN, drop_privs)
    if drop_privs:
        os.setgroups([]); os.setgid(PARENT_UID); os.setuid(PARENT_UID)
    err = []
    threading.Thread(target=drain, args=(proc.stderr, err), daemon=True).start()
    ch = Channel(proc, deadline)
    status, detail = None, ""
    try:
        ch.send({"program": payload["program"], "entry_point": payload["entry_point"]})
        hello = ch.recv()
        load = hello.get("load")
        if load == "syntax":
            status, detail = "FAIL_SYNTAX", hello.get("msg", "")
        elif load == "error":
            status, detail = "FAIL_LOAD", f'{hello.get("type")}: {hello.get("msg", "")}'
        elif load == "missing":
            status = "FAIL_MISSING_ENTRY"
        elif load != "ok":
            status = "PROTOCOL_VIOLATION"
        else:
            status, detail = run_tests(payload, ch)
    except HarnessFault as f:
        status, detail = f.status, str(f)
    if status == "CHILD_DIED":
        status = classify_death(proc, oom_before, payload["cpu_seconds"])
    finish(proc)
    use = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "status": status, "detail": detail[:1000], "calls": ch.calls,
        "cpu_time": round(use.ru_utime + use.ru_stime, 4),
        "max_rss_kb": use.ru_maxrss,
        "stderr_tail": b"".join(err).decode("utf-8", "replace")[-2000:],
    }


def exec_prelude(src, ns):
    """Trusted helpers the test needs (HumanEval's prompt defines e.g. `poly` for /32).
    Prompts end in an unfinished `def`, so give the stub a body if needed."""
    for attempt in (src, src.rstrip() + "\n    pass\n"):
        try:
            exec(compile(attempt, "<prelude>", "exec"), ns)
            return
        except SyntaxError:
            continue
    raise ValueError("prelude does not compile")


def run_tests(payload, ch):
    ns = {"__name__": "__test__"}
    try:
        if payload.get("prelude"):
            exec_prelude(payload["prelude"], ns)
        candidate = ch.proxy()
        ns[payload["entry_point"]] = candidate
        ns["candidate"] = candidate
        exec(payload["test"], ns)
        if callable(ns.get("check")):
            ns["check"](candidate)
    except HarnessFault:
        raise
    except CandidateRaised as e:
        return "FAIL_EXCEPTION", str(e)
    except AssertionError as e:
        return ("FAIL_ASSERTION" if ch.calls else "TEST_BROKEN"), str(e)
    except Exception as e:
        # test code blew up on the candidate's output (e.g. len(None)) -> logic error,
        # unless the candidate was never called, which means the benchmark is broken
        return ("FAIL_TEST_ERROR" if ch.calls else "TEST_BROKEN"), f"{type(e).__name__}: {e}"
    if ch.fault is not None:          # a fault was swallowed by the test's own try/except
        raise ch.fault
    if ch.calls == 0:
        return "TEST_BROKEN", "test never called the candidate"
    return "PASS", ""


def run_naive(payload, drop_privs):
    """Baseline that mimics a typical harness: concat program+tests, exit code 0 = pass."""
    src = payload["program"] + "\n\n" + payload.get("test", "") + \
        f"\n\nif 'check' in globals(): check({payload['entry_point']})\n"
    proc = spawn_child(payload, src, drop_privs)
    if drop_privs:
        os.setgroups([]); os.setgid(PARENT_UID); os.setuid(PARENT_UID)
    try:
        proc.communicate(timeout=payload["wall_seconds"])
        status = "PASS" if proc.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        status = "FAIL"   # child is reaped by container teardown
    return {"status": status, "detail": f"rc={proc.returncode}", "calls": None}


def main():
    payload = json.loads(sys.stdin.read())
    nonce = payload.pop("nonce")
    drop_privs = os.getuid() == 0
    t0 = time.monotonic()
    try:
        fn = run_naive if payload.get("mode") == "naive" else run_oracle
        rec = fn(payload, drop_privs)
    except Exception as e:
        rec = {"status": "HARNESS_ERROR", "detail": f"{type(e).__name__}: {e}"}
    rec["harness_wall"] = round(time.monotonic() - t0, 4)
    sys.__stdout__.write(f"@@RESULT@@{nonce}@@{json.dumps(rec)}\n")
    sys.__stdout__.flush()


if __name__ == "__main__":
    main()
