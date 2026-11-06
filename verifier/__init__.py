"""The deterministic verifier: each tool's checks live in its own module, on contract instances."""

from contract import Task, Tool

from . import answer, case, lookup, merge

# tool -> (verify_input(value), verify_output(input_value, output_value, workspace))
TOOLS = {
    Tool.READ_CASE: (case.verify_input, case.verify_read_output),
    Tool.MERGE_ONCE: (merge.verify_input, merge.verify_output),
    Tool.LOOKUP_IDS: (lookup.verify_input, lookup.verify_output),
    Tool.FULL_CASE: (case.verify_input, case.verify_full_output),
}
# task -> its output's check, verify_task(arguments, output, reference, workspace)
TASKS = {Task.TOKENIZE_CASE: answer.verify_task}


def verify_input(tool, value):
    return TOOLS[Tool(tool)][0](value)


def verify_output(tool, input_value, output_value, workspace):
    return TOOLS[Tool(tool)][1](input_value, output_value, workspace)


def verify_task(task, arguments, output, reference, workspace):
    return TASKS[Task(task)](arguments, output, reference, workspace)
