"""Checks shared by the tool verifiers. Pure functions of their arguments."""

from contract import CASES


def result(valid, message, *, score=None, **details):
    return {
        "valid": valid,
        "score": float(valid) if score is None else score,
        "message": message,
        "details": details,
    }


def case_problem(case):
    if not 1 <= case <= CASES:
        return f"case must be 1 to {CASES}, not {case}"
    return None


def merges_problem(merges):
    """What is wrong with a case's merge rules, or None: ranks must count 1, 2, ... in order."""
    for position, rule in enumerate(merges, start=1):
        if rule.rank != position:
            return f"merge rule {position} has rank {rule.rank}"
        if not rule.left or not rule.right:
            return f"merge rule {position} has an empty symbol"
    return None
