"""lookup_ids: tokens' ids in a case's vocabulary."""

from __future__ import annotations

from pathlib import Path

from contract import LookupInput, LookupOutput
from plugin import cases


def lookup_ids(given: LookupInput, workspace: Path) -> LookupOutput:
    for token in given.tokens:
        if not token:
            raise ValueError("a token must not be empty")
    ids = {entry.token: entry.id for entry in cases.full_case(given.case).vocab}
    return LookupOutput(ids=[ids.get(token) for token in given.tokens])
