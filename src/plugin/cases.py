"""The built-in cases: each trains its merge rules on its own corpus, so a case is the same on every run."""

from __future__ import annotations

from dataclasses import dataclass

from contract import CASES, WORD_START, FullCaseOutput, Merge, VocabEntry
from plugin.bpe import train, words

UNK_TOKEN = "<unk>"
UNK_ID = 0

ARTICLE = (
    "the cat sat on the mat the rat ate the hat that cat sat on the rat "
    "the cat and the rat ate at the mat"
)
TOKENIZERS = (
    "a tokenizer splits text into tokens and a model reads the tokens as ids "
    "the merges of byte pair encoding join the most frequent pairs first so common words become single tokens "
    "while rare words stay split into smaller pieces the order of the merges matters because an earlier merge "
    "can take a symbol that a later merge would need tokens tokens merges merges words words"
)


@dataclass(frozen=True)
class CaseSpec:
    corpus: str  # what the merge rules are trained on
    merges: int  # how many merge rules to train, at most
    text: str  # what the task tokenizes
    tests: str  # what the case is there to test


SPECS = {
    1: CaseSpec(
        corpus="low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3,
        merges=10,
        text="lowest newer wider low",
        tests="the textbook corpus: words seen and unseen in training",
    ),
    2: CaseSpec(
        corpus="aaa aaa aaa aaaa banana banana bandana",
        merges=6,
        text="aaaaa banana aa aaa",
        tests="runs of one letter, where a pair overlaps itself and is joined left to right",
    ),
    3: CaseSpec(
        corpus=ARTICLE,
        merges=12,
        text="the café cat ate 42 hats!",
        tests="characters the vocabulary does not hold, each its own unk token",
    ),
    4: CaseSpec(
        corpus=ARTICLE,
        merges=12,
        text="  the   mat sat  on the hat  ",
        tests="leading, trailing and repeated spaces, which make no token",
    ),
    5: CaseSpec(
        corpus=TOKENIZERS,
        merges=40,
        text="the merger of rarer tokens is lending common pieces to words",
        tests="a longer text under 40 rules, where taking the longest token or merging by position goes wrong",
    ),
    6: CaseSpec(
        corpus="the the the the then them there these theme",
        merges=12,
        text="The THE the theme thee",
        tests="upper case the vocabulary does not hold, and whole words that become one token",
    ),
}


def full_case(number: int) -> FullCaseOutput:
    """Built-in case ``number``."""
    if number not in SPECS:
        raise ValueError(f"case must be 1 to {CASES}, not {number}")
    return build_case(number, SPECS[number])


def build_case(number: int, spec: CaseSpec) -> FullCaseOutput:
    """A case from ``spec``: its text, its trained merge rules and the vocabulary they make."""
    rules = train(spec.corpus, spec.merges)
    # The vocabulary: the unk token, the corpus's characters and the word-start mark in string order,
    # then each merge's joined symbol in rank order; ids are positions.
    tokens = [UNK_TOKEN, *sorted({WORD_START, *"".join(words(spec.corpus))})]
    for left, right in rules:
        if left + right not in tokens:
            tokens.append(left + right)
    return FullCaseOutput(
        case=number,
        text=spec.text,
        merges=[Merge(rank=rank, left=left, right=right) for rank, (left, right) in enumerate(rules, start=1)],
        vocab=[VocabEntry(token=token, id=index) for index, token in enumerate(tokens)],
        unk_token=UNK_TOKEN,
        unk_id=UNK_ID,
    )
