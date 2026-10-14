"""Every status the system can emit, grouped by who is to blame.

The point of the project: a benchmark score should only count LOGIC failures against the
model. RESOURCE failures are ambiguous (slow-but-correct vs. genuinely broken), INFRA
failures are our fault, and INTEGRITY failures are the model gaming the harness.
"""
from enum import Enum


class Category(str, Enum):
    PASS = "pass"
    LOGIC = "logic"          # model's code is wrong
    RESOURCE = "resource"    # hit a limit; may or may not be the model's fault
    INTEGRITY = "integrity"  # code tried to exit early / fake results / break protocol
    INFRA = "infra"          # sandbox or benchmark problem, not the model


STATUS_CATEGORY = {
    "PASS": Category.PASS,
    "FAIL_ASSERTION": Category.LOGIC,
    "FAIL_EXCEPTION": Category.LOGIC,
    "FAIL_TEST_ERROR": Category.LOGIC,
    "FAIL_SYNTAX": Category.LOGIC,
    "FAIL_LOAD": Category.LOGIC,
    "FAIL_MISSING_ENTRY": Category.LOGIC,
    "FAIL_UNSERIALIZABLE": Category.LOGIC,
    "TIMEOUT_CPU": Category.RESOURCE,
    "TIMEOUT_WALL": Category.RESOURCE,
    "OOM": Category.RESOURCE,
    "OOM_SUSPECTED": Category.RESOURCE,
    "CHILD_EXITED": Category.INTEGRITY,
    "CHILD_CRASH": Category.INTEGRITY,
    "PROTOCOL_VIOLATION": Category.INTEGRITY,
    "TEST_BROKEN": Category.INFRA,
    "HARNESS_ERROR": Category.INFRA,
    "HARNESS_LOST": Category.INFRA,
    "HOST_TIMEOUT": Category.INFRA,
    "INFRA_ERROR": Category.INFRA,
    "RESULT_FORGERY": Category.INFRA,  # >1 result line or bad nonce: investigate!
    # naive baseline only
    "FAIL": Category.LOGIC,
}


def category(status: str) -> Category:
    return STATUS_CATEGORY.get(status, Category.INFRA)
