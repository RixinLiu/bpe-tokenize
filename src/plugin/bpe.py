"""Byte-pair encoding as this plugin defines it: training a case's merge rules, and one merge round."""

from __future__ import annotations

import itertools
from collections import Counter

from contract import WORD_START

Pair = tuple[str, str]


def word_symbols(word: str) -> list[str]:
    """A word's symbols before any merge: the word-start mark, then each character."""
    return [WORD_START, *word]


def words(text: str) -> list[str]:
    """The text's words: maximal runs of characters other than the space."""
    return [word for word in text.split(" ") if word]


def least_ranked_pair(symbols: list[str], ranks: dict[Pair, int]) -> Pair | None:
    """The adjacent pair with the smallest rank, or None when no adjacent pair has a rule."""
    best: tuple[int, Pair] | None = None
    for pair in itertools.pairwise(symbols):
        rank = ranks.get(pair)
        if rank is not None and (best is None or rank < best[0]):
            best = (rank, pair)
    return None if best is None else best[1]


def join_pair(symbols: list[str], pair: Pair) -> list[str]:
    """Every occurrence of ``pair`` joined into one symbol, scanning left to right without overlap."""
    joined: list[str] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            joined.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            joined.append(symbols[i])
            i += 1
    return joined


def encode_word(symbols: list[str], ranks: dict[Pair, int]) -> list[str]:
    """Merge rounds until no adjacent pair has a rule."""
    while (pair := least_ranked_pair(symbols, ranks)) is not None:
        symbols = join_pair(symbols, pair)
    return symbols


def train(corpus: str, merges: int) -> list[Pair]:
    """Up to ``merges`` rules learned from ``corpus``: each round joins the most frequent adjacent pair,
    ties going to the smallest pair in string order, and stops early once no word has two symbols."""
    counts = Counter(words(corpus))
    split = {word: word_symbols(word) for word in counts}
    rules: list[Pair] = []
    for _ in range(merges):
        pairs: Counter[Pair] = Counter()
        for word, symbols in split.items():
            for pair in itertools.pairwise(symbols):
                pairs[pair] += counts[word]
        if not pairs:
            break
        best = min(pairs, key=lambda pair: (-pairs[pair], pair))
        rules.append(best)
        split = {word: join_pair(symbols, best) for word, symbols in split.items()}
    return rules
