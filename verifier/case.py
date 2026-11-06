"""read_case's and full_case's checks: the case asked for, with well-formed rules and vocabulary."""

from contract import WORD_START

from .checks import case_problem, merges_problem, result


def verify_input(value):
    problem = case_problem(value.case)
    if problem:
        return result(False, problem)
    return result(True, "input is valid")


def verify_read_output(input_value, output_value, workspace):
    if output_value.case != input_value.case:
        return result(False, f"asked for case {input_value.case}, got case {output_value.case}")
    problem = merges_problem(output_value.merges)
    if problem:
        return result(False, problem)
    if WORD_START in output_value.text:
        return result(False, "the text holds the word-start mark")
    return result(True, "the case is well formed", merges=len(output_value.merges))


def verify_full_output(input_value, output_value, workspace):
    checked = verify_read_output(input_value, output_value, workspace)
    if not checked["valid"]:
        return checked
    ids = {}
    for entry in output_value.vocab:
        if entry.token in ids:
            return result(False, f"the vocabulary holds {entry.token!r} twice")
        ids[entry.token] = entry.id
    if len(set(ids.values())) != len(ids):
        return result(False, "two tokens share an id")
    if ids.get(output_value.unk_token) != output_value.unk_id:
        return result(False, "the unk token is not in the vocabulary under its id")
    for rule in output_value.merges:
        for token in (rule.left, rule.right, rule.left + rule.right):
            if token not in ids:
                return result(False, f"merge rule {rule.rank} uses {token!r}, which the vocabulary does not hold")
    return result(True, "the case is well formed", merges=len(output_value.merges), vocab=len(ids))
