"""merge_once's checks. Without the case's rules they check what any merge round must keep."""

import itertools

from .bpe import join
from .checks import case_problem, result


def verify_input(value):
    problem = case_problem(value.case)
    if problem:
        return result(False, problem)
    for symbol in value.symbols:
        if symbol == "" or " " in symbol:
            return result(False, f"every symbol must be non-empty and hold no space, not {symbol!r}")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    given = input_value.symbols
    if "".join(output_value.symbols) != "".join(given):
        return result(False, "merging changed the text the symbols spell")
    if output_value.done != (output_value.merged is None):
        return result(False, "done must be true exactly when no merge was applied")
    if output_value.merged is None:
        if output_value.symbols != given:
            return result(False, "no merge was applied, yet the symbols changed")
        return result(True, "no adjacent pair has a rule")
    rule = output_value.merged
    if (rule.left, rule.right) not in set(itertools.pairwise(given)):
        return result(False, f"the merged pair {rule.left!r} {rule.right!r} is not adjacent in the symbols")
    if output_value.symbols != join(given, rule.left, rule.right):
        return result(False, "the pair was not joined at every occurrence, left to right without overlap")
    return result(True, f"merge rule {rule.rank} applied", symbols=len(output_value.symbols))
