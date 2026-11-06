from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import pytest
from slimconfig import build_output, plain_output

import verifier
from contract import CASES, TASKS, TOOLS, WORD_START, FullCaseOutput, Task, TokenizeOutput, Tool
from plugin import cases
from plugin.bpe import encode_word, join_pair, word_symbols, words

TASK = Task.TOKENIZE_CASE.value


# The verifier gets contract instances, as NanoAgent's sandbox builds them from JSON.
def verify_task(case: int, output: dict, reference: FullCaseOutput | None, workspace: Path) -> dict:
    given, made = TASKS[Task.TOKENIZE_CASE]
    pinned = None if reference is None else build_output(plain_output(reference), FullCaseOutput)
    return verifier.verify_task(
        TASK, build_output({"case": case}, given), build_output(output, made), pinned, str(workspace)
    )


def verify_input(tool: Tool, value: dict) -> dict:
    return verifier.verify_input(tool.value, build_output(value, TOOLS[tool][0]))


def verify_output(tool: Tool, given: dict, made: dict) -> dict:
    given_class, made_class = TOOLS[tool]
    return verifier.verify_output(tool.value, build_output(given, given_class), build_output(made, made_class), ".")


def tokenize(case: FullCaseOutput, symbols_of=None) -> dict:
    """The case's answer, built with the plugin's own encoder (or ``symbols_of``, a wrong one)."""
    ranks = {(rule.left, rule.right): rule.rank for rule in case.merges}
    ids = {entry.token: entry.id for entry in case.vocab}
    symbols_of = symbols_of or (lambda word: encode_word(word_symbols(word), ranks))
    symbols = [symbol for word in words(case.text) for symbol in symbols_of(word)]
    tokens = [symbol if symbol in ids else case.unk_token for symbol in symbols]
    return {"tokens": tokens, "ids": [ids[token] for token in tokens]}


def greedy_longest_match(case: FullCaseOutput):
    """A common wrong tokenizer: at each position, take the longest vocabulary token that fits."""
    vocab = {entry.token for entry in case.vocab}

    def symbols_of(word: str) -> list[str]:
        text, out = WORD_START + word, []
        while text:
            length = next((n for n in range(len(text), 0, -1) if text[:n] in vocab), 1)
            out.append(text[:length])
            text = text[length:]
        return out

    return symbols_of


def one_round(case: FullCaseOutput):
    """Stopping after the first merge round of each word."""
    ranks = {(rule.left, rule.right): rule.rank for rule in case.merges}

    def symbols_of(word: str) -> list[str]:
        symbols = word_symbols(word)
        pairs = [pair for pair in itertools.pairwise(symbols) if pair in ranks]
        return join_pair(symbols, min(pairs, key=ranks.__getitem__)) if pairs else symbols

    return symbols_of


def rank_ignored(case: FullCaseOutput):
    """Joining pairs in the order they appear in the word, not by rank."""
    ranks = {(rule.left, rule.right): rule.rank for rule in case.merges}

    def symbols_of(word: str) -> list[str]:
        symbols = word_symbols(word)
        while pairs := [pair for pair in itertools.pairwise(symbols) if pair in ranks]:
            symbols = join_pair(symbols, pairs[0])
        return symbols

    return symbols_of


def mutants(case: FullCaseOutput) -> dict[str, dict]:
    """Wrong answers an agent could submit, each differing from the right one in one way."""
    right = tokenize(case)
    tokens, ids = right["tokens"], right["ids"]
    ranks = {(rule.left, rule.right): rule.rank for rule in case.merges}
    vocab = {entry.token: entry.id for entry in case.vocab}
    raw = [symbol for word in words(case.text) for symbol in encode_word(word_symbols(word), ranks)]
    found = {
        "no word-start mark": tokenize(case, lambda word: encode_word(list(word), ranks)),
        "greedy longest match": tokenize(case, greedy_longest_match(case)),
        "one merge round per word": tokenize(case, one_round(case)),
        "merges in order of position, not rank": tokenize(case, rank_ignored(case)),
        "unmerged characters": tokenize(case, word_symbols),
        "last token missing": {"tokens": tokens[:-1], "ids": ids[:-1]},
        "first token missing": {"tokens": tokens[1:], "ids": ids[1:]},
        "token added": {"tokens": [*tokens, tokens[-1]], "ids": [*ids, ids[-1]]},
        "nothing submitted": {"tokens": [], "ids": []},
        "one id wrong": {"tokens": tokens, "ids": [*ids[:-1], ids[-1] + 1]},
        "tokens reversed": {"tokens": tokens[::-1], "ids": ids[::-1]},
        "ids shorter than tokens": {"tokens": tokens, "ids": ids[:-1]},
        "unknown symbol kept raw": {"tokens": raw, "ids": ids},
        "spaces kept as tokens": {"tokens": [tokens[0], " ", *tokens[1:]], "ids": [ids[0], case.unk_id, *ids[1:]]},
        "unk token with a known id": {"tokens": [case.unk_token, *tokens[1:]], "ids": [vocab[tokens[0]], *ids[1:]]},
    }
    return {name: wrong for name, wrong in found.items() if wrong != right}


@pytest.mark.parametrize("number", range(1, CASES + 1))
def test_approves_the_right_tokens_of_every_case(number: int, tmp_path: Path) -> None:
    case = cases.full_case(number)
    verdict = verify_task(number, tokenize(case), case, tmp_path)
    assert verdict["valid"] is True and verdict["score"] == 1.0, verdict["message"]


@pytest.mark.parametrize("number", range(1, CASES + 1))
def test_refuses_every_wrong_answer(number: int, tmp_path: Path) -> None:
    case = cases.full_case(number)
    wrong = mutants(case)
    assert len(wrong) >= 10
    for name, output in wrong.items():
        verdict = verify_task(number, output, case, tmp_path)
        assert verdict["valid"] is False, f"case {number}: {name} was approved"
        assert 0.0 <= verdict["score"] < 1.0, f"case {number}: {name} scored {verdict['score']}"


def test_each_wrong_answer_says_what_is_wrong(tmp_path: Path) -> None:
    case = cases.full_case(3)
    right = tokenize(case)
    wrong = mutants(case)
    assert "not in the vocabulary" in verify_task(3, wrong["unknown symbol kept raw"], case, tmp_path)["message"]
    assert "has id" in verify_task(3, wrong["one id wrong"], case, tmp_path)["message"]
    assert "tokens but" in verify_task(3, wrong["ids shorter than tokens"], case, tmp_path)["message"]
    message = verify_task(3, wrong["last token missing"], case, tmp_path)["message"]
    assert message == f"{len(right['tokens']) - 1} tokens submitted, expected {len(right['tokens'])}"
    assert "expected" in verify_task(3, wrong["one merge round per word"], case, tmp_path)["message"]


def test_refuses_without_the_reference_of_the_case_asked_for(tmp_path: Path) -> None:
    case = cases.full_case(1)
    assert verify_task(1, tokenize(case), None, tmp_path)["valid"] is False
    assert verify_task(2, tokenize(case), case, tmp_path)["valid"] is False


def test_a_text_of_spaces_has_no_tokens(tmp_path: Path) -> None:
    case = cases.full_case(4)
    blank = replace(case, text="   ")
    assert verify_task(4, {"tokens": [], "ids": []}, blank, tmp_path)["valid"] is True
    assert verify_task(4, {"tokens": ["▁"], "ids": [case.unk_id]}, blank, tmp_path)["valid"] is False


@pytest.mark.parametrize(
    "output",
    [
        {"tokens": ["▁low"]},
        {"tokens": ["▁low"], "ids": ["16"]},
        {"tokens": None, "ids": []},
        {"tokens": ["▁low"], "ids": [16], "score": 1},
        {"tokens": "▁low", "ids": [16]},
    ],
)
def test_the_contract_refuses_a_malformed_output_before_the_verifier(output: dict) -> None:
    with pytest.raises(ValueError):
        build_output(output, TokenizeOutput)


def test_checks_what_the_types_cannot_say() -> None:
    assert verify_input(Tool.READ_CASE, {"case": 1})["valid"] is True
    assert verify_input(Tool.READ_CASE, {"case": 0})["valid"] is False
    assert verify_input(Tool.FULL_CASE, {"case": CASES + 1})["valid"] is False
    assert verify_input(Tool.MERGE_ONCE, {"case": 1, "symbols": ["▁", "l"]})["valid"] is True
    assert verify_input(Tool.MERGE_ONCE, {"case": 1, "symbols": ["▁", ""]})["valid"] is False
    assert verify_input(Tool.MERGE_ONCE, {"case": 1, "symbols": ["l o"]})["valid"] is False
    assert verify_input(Tool.LOOKUP_IDS, {"case": 1, "tokens": ["lo"]})["valid"] is True
    assert verify_input(Tool.LOOKUP_IDS, {"case": 1, "tokens": [""]})["valid"] is False


def test_checks_each_merge_round() -> None:
    given = {"case": 2, "symbols": ["▁", "a", "a", "a"]}
    rule = {"rank": 1, "left": "a", "right": "a"}
    joined = {"merged": rule, "symbols": ["▁", "aa", "a"], "done": False}
    assert verify_output(Tool.MERGE_ONCE, given, joined)["valid"] is True
    assert verify_output(Tool.MERGE_ONCE, given, {**joined, "symbols": ["▁", "a", "aa"]})["valid"] is False
    assert verify_output(Tool.MERGE_ONCE, given, {**joined, "symbols": ["▁", "aa", "b"]})["valid"] is False
    assert verify_output(Tool.MERGE_ONCE, given, {**joined, "done": True})["valid"] is False
    rule_absent = {**rule, "left": "b", "right": "b"}
    assert verify_output(Tool.MERGE_ONCE, given, {**joined, "merged": rule_absent})["valid"] is False
    final = {"case": 2, "symbols": ["▁aa", "a"]}
    assert verify_output(Tool.MERGE_ONCE, final, {"merged": None, "symbols": ["▁aa", "a"], "done": True})["valid"]
    assert not verify_output(Tool.MERGE_ONCE, final, {"merged": None, "symbols": ["▁aaa"], "done": True})["valid"]


def test_checks_the_cases_and_lookups() -> None:
    for number in range(1, CASES + 1):
        full = plain_output(cases.full_case(number))
        assert verify_output(Tool.FULL_CASE, {"case": number}, full)["valid"] is True
    full = plain_output(cases.full_case(1))
    assert verify_output(Tool.FULL_CASE, {"case": 2}, full)["valid"] is False
    assert verify_output(Tool.FULL_CASE, {"case": 1}, {**full, "vocab": full["vocab"][1:]})["valid"] is False
    assert verify_output(Tool.FULL_CASE, {"case": 1}, {**full, "merges": full["merges"][::-1]})["valid"] is False
    looked = {"case": 1, "tokens": ["lo", "lo", "é"]}
    assert verify_output(Tool.LOOKUP_IDS, looked, {"ids": [3, 3, None]})["valid"] is True
    assert verify_output(Tool.LOOKUP_IDS, looked, {"ids": [3, 4, None]})["valid"] is False
    assert verify_output(Tool.LOOKUP_IDS, looked, {"ids": [3, 3]})["valid"] is False
    assert verify_output(Tool.LOOKUP_IDS, looked, {"ids": [3, 3, -1]})["valid"] is False
