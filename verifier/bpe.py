"""The verifier's own byte-pair encoder, written apart from the plugin's so a bug in one shows against the other.

A word's symbols merge one rule at a time: the rule of least rank among the adjacent pairs joins every
occurrence of its pair, left to right without overlap, until no adjacent pair has a rule.
"""

import itertools

from contract import WORD_START


def join(symbols, left, right):
    """Every non-overlapping occurrence of (left, right) joined, scanning left to right.

    A joined symbol is left + right, never equal to left, so it never starts another occurrence.
    """
    out = []
    for symbol in symbols:
        if out and symbol == right and out[-1] == left:
            out[-1] = left + right
        else:
            out.append(symbol)
    return out


def encode_symbols(text, merges):
    """The text's final symbols, word after word."""
    rank_of = {}
    rule_of = {}
    for rule in merges:
        rank_of[(rule.left, rule.right)] = rule.rank
        rule_of[rule.rank] = (rule.left, rule.right)
    symbols = []
    for word in text.split(" "):
        if word == "":
            continue
        pieces = [WORD_START] + list(word)
        while True:
            ranks = [rank_of[pair] for pair in itertools.pairwise(pieces) if pair in rank_of]
            if not ranks:
                break
            left, right = rule_of[min(ranks)]
            pieces = join(pieces, left, right)
        symbols.extend(pieces)
    return symbols


def encode(case):
    """``case``: a contract FullCaseOutput. Its text's tokens and ids, the unk token for an unknown symbol."""
    ids = {entry.token: entry.id for entry in case.vocab}
    tokens = [symbol if symbol in ids else case.unk_token for symbol in encode_symbols(case.text, case.merges)]
    return tokens, [ids.get(token, case.unk_id) for token in tokens]
