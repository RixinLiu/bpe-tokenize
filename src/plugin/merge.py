"""merge_once: one merge round on one word's symbols, under a case's rules."""

from __future__ import annotations

from pathlib import Path

from contract import Merge, MergeInput, MergeOutput
from plugin import cases
from plugin.bpe import join_pair, least_ranked_pair


def merge_once(given: MergeInput, workspace: Path) -> MergeOutput:
    for symbol in given.symbols:
        if not symbol or " " in symbol:
            raise ValueError(f"every symbol must be non-empty and hold no space, not {symbol!r}")
    rules = cases.full_case(given.case).merges
    ranks = {(rule.left, rule.right): rule.rank for rule in rules}
    pair = least_ranked_pair(given.symbols, ranks)
    if pair is None:
        return MergeOutput(merged=None, symbols=list(given.symbols), done=True)
    merged = Merge(rank=ranks[pair], left=pair[0], right=pair[1])
    return MergeOutput(merged=merged, symbols=join_pair(given.symbols, pair), done=False)
