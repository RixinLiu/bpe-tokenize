"""read_case shows the agent a case without its vocabulary; full_case, the task's reference, shows all of it."""

from __future__ import annotations

from pathlib import Path

from contract import CaseInput, CaseOutput, FullCaseOutput
from plugin import cases


def full_case(given: CaseInput, workspace: Path) -> FullCaseOutput:
    return cases.full_case(given.case)


def read_case(given: CaseInput, workspace: Path) -> CaseOutput:
    case = cases.full_case(given.case)
    return CaseOutput(
        case=case.case,
        text=case.text,
        merges=case.merges,
        unk_token=case.unk_token,
        unk_id=case.unk_id,
        vocab_size=len(case.vocab),
    )
