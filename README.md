# BPE Tokenize

A `nanoagent.plugin/v0001` plugin that asks an agent to tokenize a text with
byte-pair encoding (BPE), the way GPT-style tokenizers do, and checks every
token and id against a tokenizer the verifier runs on its own.

## Tools

| Tool | Does | Code |
|---|---|---|
| `read_case` | Reads a case's text, its merge rules in rank order, and its unk token and id; not the vocabulary | `src/plugin/case.py` |
| `merge_once` | Applies one merge round to one word's symbols: the rule of least rank among its adjacent pairs joins every occurrence of its pair, left to right without overlap | `src/plugin/merge.py` |
| `lookup_ids` | Looks up tokens' ids in a case's vocabulary, null for a token it does not hold | `src/plugin/lookup.py` |
| `full_case` | Reads a case's text, merge rules and whole vocabulary: the `tokenize-case` task's reference, which the agent never sees | `src/plugin/case.py` |

### Tasks

| Task | Does | Runner |
|---|---|---|
| `tokenize-case` | Tokenizes a case's text with BPE and gives each token's id | `agent`, with `read_case`, `merge_once` and `lookup_ids`; reference `full_case` |

The task's input is a case number, 1 to 6. Each case trains its merge rules on
its own corpus (`src/plugin/cases.py`), so a case is the same on every run, and
each is there to test one thing:

| Case | Text | Tests |
|---|---|---|
| 1 | `lowest newer wider low` | words seen and unseen in training |
| 2 | `aaaaa banana aa aaa` | a pair that overlaps itself, joined left to right |
| 3 | `the café cat ate 42 hats!` | characters the vocabulary does not hold, each its own unk token |
| 4 | `  the   mat sat  on the hat  ` | leading, trailing and repeated spaces, which give no token |
| 5 | `the merger of rarer tokens is lending common pieces to words` | 40 rules, where taking the longest token or merging by position goes wrong |
| 6 | `The THE the theme thee` | upper case the vocabulary does not hold, and whole words that become one token |

The agent splits the text into words, writes each word as `▁` and its
characters, calls `merge_once` until it returns `done`, joins the words'
symbols, and looks up their ids, writing a symbol the vocabulary does not hold
as `<unk>`. The tools do every step that follows from a rule; the agent
decides which to call, on what, and when to stop.

## Layout

```text
.
├── nanoagent.yaml   # manifest (a slimconfig ManifestConfig): id, runtime, permissions, tools, tasks
├── contract.py      # each tool's and task's typed input and output: slimconfig Output classes
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env             # environment variable NAMEs this plugin needs (none)
├── src/plugin/      # the MCP server (__main__.py, its IMPLEMENTATIONS), the tools, bpe.py and the cases
├── tests/           # test_verifier.py (right, wrong, missing, malformed and boundary answers) and test_bpe.py
├── ui/              # outputs are drawn from their JSON Schema
├── product/         # the product page a catalog shows: product.yaml and icon.svg
└── verifier/        # deterministic verifier package, with its own BPE encoder (verifier/bpe.py)
```

`verifier/` imports only the standard library and `contract`. For the task it
takes the pinned `full_case` output and tokenizes the text again with its own
encoder, written apart from the plugin's: it joins pairs with a stack where the
plugin scans by index. It then checks that the submitted tokens and ids have
the same length, that every token is in the vocabulary under the submitted id,
and that the tokens equal its own, naming the first that differs. The score is
the share of positions whose token and id are both right. For each tool call
it checks the case number, that a merge round keeps the text its symbols spell
and joins its pair at every occurrence, and that each token gets one id.

`tests/test_bpe.py` checks the two encoders agree on 500 random corpora and
texts, and that an agent following the instructions with the tools reaches the
verifier's answer on 100 more. `tests/test_verifier.py` checks the verifier
approves each case's right answer and refuses, for every case, wrong answers
an agent could give: no `▁`, the longest token first, one merge round only,
merges by position instead of rank, unknown characters kept raw, spaces kept
as tokens, missing, extra or reordered tokens, a wrong id, and ids that do not
match the tokens in number.

## Develop and run

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

NanoAgent passes the folder the plugin may write to as `NANOAGENT_WORKSPACE`;
this plugin writes nothing.
