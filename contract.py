"""Each tool's and task's typed input and output (see nanoagent.plugin.v0001.contract): declarations only."""

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from slimconfig import Output


class Tool(Enum):
    READ_CASE = "read_case"
    MERGE_ONCE = "merge_once"
    LOOKUP_IDS = "lookup_ids"
    FULL_CASE = "full_case"


class Task(Enum):
    TOKENIZE_CASE = "tokenize-case"


# How many built-in cases there are: a case is numbered 1 to CASES.
CASES = 6
# The symbol that marks the start of a word (U+2581, LOWER ONE EIGHTH BLOCK).
WORD_START = "▁"

CASE = "The case's number, 1 to 6"
SYMBOLS = "One word's symbols in order, each a non-empty string without spaces, e.g. ['▁', 'l', 'o', 'w']"


@dataclass
class Merge(Output):
    rank: Annotated[int, "The merge's priority: 1 is applied before 2, and so on"]
    left: Annotated[str, "The left symbol of the pair"]
    right: Annotated[str, "The right symbol of the pair"]


@dataclass
class VocabEntry(Output):
    token: Annotated[str, "A token of the vocabulary"]
    id: Annotated[int, "Its id"]


@dataclass
class CaseInput(Output):
    case: Annotated[int, CASE]


@dataclass
class CaseOutput(Output):
    case: Annotated[int, CASE]
    text: Annotated[str, "The text to tokenize"]
    merges: Annotated[list[Merge], "The merge rules, rank 1 first"]
    unk_token: Annotated[str, "The token written for a symbol the vocabulary does not hold"]
    unk_id: Annotated[int, "The unk token's id"]
    vocab_size: Annotated[int, "How many tokens the vocabulary holds, the unk token included"]


@dataclass
class MergeInput(Output):
    case: Annotated[int, CASE]
    symbols: Annotated[list[str], SYMBOLS]


@dataclass
class MergeOutput(Output):
    merged: Annotated[Merge | None, "The merge applied: the adjacent pair of least rank; null when no pair has a rule"]
    symbols: Annotated[
        list[str], "The symbols after every occurrence of that pair, left to right without overlap, is joined"
    ]
    done: Annotated[bool, "Whether no adjacent pair has a merge rule, so the symbols are final"]


@dataclass
class LookupInput(Output):
    case: Annotated[int, CASE]
    tokens: Annotated[list[str], "Tokens to look up, each a non-empty string"]


@dataclass
class LookupOutput(Output):
    ids: Annotated[list[int | None], "Each token's id in the vocabulary, in order; null for a token it does not hold"]


@dataclass
class FullCaseOutput(Output):
    case: Annotated[int, CASE]
    text: Annotated[str, "The text to tokenize"]
    merges: Annotated[list[Merge], "The merge rules, rank 1 first"]
    vocab: Annotated[list[VocabEntry], "Every token of the vocabulary with its id, the unk token included"]
    unk_token: Annotated[str, "The token written for a symbol the vocabulary does not hold"]
    unk_id: Annotated[int, "The unk token's id"]


@dataclass
class TokenizeOutput(Output):
    tokens: Annotated[list[str], "The text's tokens in order, the unk token for a symbol the vocabulary does not hold"]
    ids: Annotated[list[int], "Each token's id, in the same order"]


TOOLS = {
    Tool.READ_CASE: (CaseInput, CaseOutput),
    Tool.MERGE_ONCE: (MergeInput, MergeOutput),
    Tool.LOOKUP_IDS: (LookupInput, LookupOutput),
    Tool.FULL_CASE: (CaseInput, FullCaseOutput),
}
TASKS = {Task.TOKENIZE_CASE: (CaseInput, TokenizeOutput)}
