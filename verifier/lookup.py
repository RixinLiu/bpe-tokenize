"""lookup_ids's checks: one id or null per token."""

from .checks import case_problem, result


def verify_input(value):
    problem = case_problem(value.case)
    if problem:
        return result(False, problem)
    if "" in value.tokens:
        return result(False, "a token must not be empty")
    return result(True, "input is valid")


def verify_output(input_value, output_value, workspace):
    if len(output_value.ids) != len(input_value.tokens):
        return result(False, f"{len(input_value.tokens)} tokens but {len(output_value.ids)} ids")
    seen = {}
    for token, token_id in zip(input_value.tokens, output_value.ids):
        if token_id is not None and token_id < 0:
            return result(False, f"{token!r} has a negative id")
        if seen.get(token, token_id) != token_id:
            return result(False, f"{token!r} got two different ids")
        seen[token] = token_id
    return result(True, "every token has one id or none", tokens=len(output_value.ids))
