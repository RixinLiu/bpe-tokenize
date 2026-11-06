"""The tokenize-case task's check: the submitted tokens and ids are the case's, recomputed from scratch."""

from .bpe import encode
from .checks import result


def first_difference(submitted, expected):
    """Where two token lists first differ, as a message."""
    for position, (got, want) in enumerate(zip(submitted, expected)):
        if got != want:
            return f"token {position} is {got!r}, expected {want!r}"
    return f"{len(submitted)} tokens submitted, expected {len(expected)}"


def verify_task(arguments, output, reference, workspace):
    """``reference``: the contract FullCaseOutput of the case asked for, pinned before the agent ran."""
    if reference is None or reference.case != arguments.case:
        return result(False, f"the pinned reference is not case {arguments.case}")
    expected_tokens, expected_ids = encode(reference)
    tokens, ids = output.tokens, output.ids
    # The share of positions whose token and id are both right.
    longest = max(len(tokens), len(ids), len(expected_tokens))
    submitted = zip(tokens, ids)
    matching = sum(1 for got, want in zip(submitted, zip(expected_tokens, expected_ids)) if got == want)
    score = 1.0 if longest == 0 else matching / longest
    if len(tokens) != len(ids):
        return result(False, f"{len(tokens)} tokens but {len(ids)} ids", score=0.0)
    vocab = {entry.token: entry.id for entry in reference.vocab}
    for position, (token, token_id) in enumerate(zip(tokens, ids)):
        if token not in vocab:
            return result(
                False,
                f"token {position} {token!r} is not in the vocabulary: a symbol it does not hold is written as"
                f" {reference.unk_token!r}",
                score=score,
            )
        if vocab[token] != token_id:
            return result(False, f"token {position} {token!r} has id {vocab[token]}, not {token_id}", score=score)
    if tokens != expected_tokens or ids != expected_ids:
        return result(
            False, first_difference(tokens, expected_tokens), score=score, expected=expected_tokens, submitted=tokens
        )
    return result(True, f"all {len(tokens)} tokens and ids match", tokens=len(tokens))
