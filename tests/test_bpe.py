from __future__ import annotations

import random
from pathlib import Path

import pytest

import verifier.bpe
from contract import CASES, CaseInput, LookupInput, MergeInput
from plugin import cases, full_case, lookup_ids, merge_once, read_case
from plugin.bpe import encode_word, word_symbols, words
from plugin.cases import CaseSpec, build_case


def plugin_tokens(case) -> list[str]:
    """The case's tokens by the plugin's encoder, which the merge_once tool runs one round at a time."""
    ranks = {(rule.left, rule.right): rule.rank for rule in case.merges}
    vocab = {entry.token for entry in case.vocab}
    symbols = [symbol for word in words(case.text) for symbol in encode_word(word_symbols(word), ranks)]
    return [symbol if symbol in vocab else case.unk_token for symbol in symbols]


def random_spec(rng: random.Random) -> CaseSpec:
    letters = rng.choice(["ab", "abc", "abcde", "lowenst"])
    corpus = " ".join(
        "".join(rng.choice(letters) for _ in range(rng.randint(1, 7))) for _ in range(rng.randint(1, 30))
    )
    # The text may hold letters the corpus lacks, and runs of spaces.
    text = "".join(rng.choice(letters + "xz" + "  ") for _ in range(rng.randint(0, 40)))
    return CaseSpec(corpus=corpus, merges=rng.randint(0, 30), text=text, tests="random")


@pytest.mark.parametrize("seed", range(500))
def test_the_verifier_and_the_plugin_tokenize_alike(seed: int) -> None:
    case = build_case(0, random_spec(random.Random(seed)))
    assert verifier.bpe.encode(case)[0] == plugin_tokens(case)


@pytest.mark.parametrize("seed", range(100))
def test_merge_once_calls_reach_the_same_tokens(seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """What an agent that follows the instructions does: merge_once until done, word by word."""
    case = build_case(1, random_spec(random.Random(seed)))
    monkeypatch.setattr(cases, "full_case", lambda number: case)
    vocab_ids = {entry.token: entry.id for entry in case.vocab}
    tokens: list[str] = []
    for word in words(case.text):
        symbols, done = word_symbols(word), False
        while not done:
            step = merge_once(MergeInput(case=1, symbols=symbols), tmp_path)
            symbols, done = step.symbols, step.done
        tokens += symbols
    ids = lookup_ids(LookupInput(case=1, tokens=tokens), tmp_path).ids
    tokens = [token if token_id is not None else case.unk_token for token, token_id in zip(tokens, ids)]
    expected_tokens, expected_ids = verifier.bpe.encode(case)
    assert tokens == expected_tokens
    assert [vocab_ids[token] for token in tokens] == expected_ids


def test_cases_are_the_same_on_every_run() -> None:
    for number in range(1, CASES + 1):
        assert cases.full_case(number) == cases.full_case(number)
    with pytest.raises(ValueError):
        cases.full_case(CASES + 1)


def test_each_case_tests_what_it_says() -> None:
    def tokens(number: int) -> list[str]:
        return verifier.bpe.encode(cases.full_case(number))[0]

    assert tokens(2)[:3] == ["▁aa", "aa", "a"]  # aaaaa: a+a joins left to right, the last a stays
    assert tokens(3).count("<unk>") == 5  # f, é, 4, 2 and !, which the corpus lacks
    assert "  " in cases.full_case(4).text and " " not in tokens(4)
    assert "▁theme" in tokens(6) and tokens(6).count("<unk>") == 4
    assert {merge.rank for merge in cases.full_case(5).merges} == set(range(1, 41))


def test_read_case_hides_only_the_vocabulary(tmp_path: Path) -> None:
    shown = read_case(CaseInput(case=3), tmp_path)
    whole = full_case(CaseInput(case=3), tmp_path)
    assert (shown.text, shown.merges, shown.unk_token, shown.unk_id) == (
        whole.text,
        whole.merges,
        whole.unk_token,
        whole.unk_id,
    )
    assert shown.vocab_size == len(whole.vocab)
    assert not hasattr(shown, "vocab")


def test_merge_once_applies_one_round(tmp_path: Path) -> None:
    step = merge_once(MergeInput(case=2, symbols=["▁", "a", "a", "a", "a", "a"]), tmp_path)
    assert step.merged is not None and (step.merged.left, step.merged.right, step.merged.rank) == ("a", "a", 1)
    assert step.symbols == ["▁", "aa", "aa", "a"] and step.done is False
    final = merge_once(MergeInput(case=2, symbols=["▁aa", "aa", "a"]), tmp_path)
    assert final.merged is None and final.done is True and final.symbols == ["▁aa", "aa", "a"]
    assert merge_once(MergeInput(case=2, symbols=[]), tmp_path).done is True
    with pytest.raises(ValueError, match="space"):
        merge_once(MergeInput(case=2, symbols=["a a"]), tmp_path)


def test_lookup_ids_gives_null_for_unknown_tokens(tmp_path: Path) -> None:
    case = cases.full_case(3)
    ids = {entry.token: entry.id for entry in case.vocab}
    assert lookup_ids(LookupInput(case=3, tokens=["▁the", "é", "<unk>"]), tmp_path).ids == [ids["▁the"], None, 0]
    with pytest.raises(ValueError):
        lookup_ids(LookupInput(case=3, tokens=[""]), tmp_path)
