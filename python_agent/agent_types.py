"""Sub-agent type definitions.

Each ``AgentType`` defines a sub-agent's identity (preamble), which
prompt blocks compose its system prompt, and which tools it can use.
The main agent's ``spawn_agent`` tool looks up types from ``AGENT_TYPES``
to configure the child ``Agent`` instance.

Adding a new agent type: define a new ``AgentType`` constant, add it to
``AGENT_TYPES``, and update the ``spawn_agent`` guidance in the WORKFLOW
section of ``prompts.py``.
"""

from dataclasses import dataclass

from .prompts import PromptBlock
from .tools.base import ToolName


@dataclass(frozen=True)
class AgentType:
    """Defines a sub-agent's identity, system prompt composition, and tool set.

    ``system_preamble`` replaces the generic IDENTITY block — it tells the
    sub-agent who it is and what its role is.

    ``prompt_blocks`` selects which standard sections to include. IDENTITY
    should NOT be listed — the preamble replaces it.

    ``tools`` lists which tools the sub-agent can use. ``make_plan`` and
    ``send_response`` are always added automatically (required by the
    agent loop) and should not be listed here.

    All sub-agents auto-approve tool calls and cannot spawn children —
    these are structural invariants, not per-type configuration.
    """

    name: str
    description: str
    system_preamble: str
    prompt_blocks: tuple[PromptBlock, ...]
    tools: tuple[ToolName, ...]


# ───────────────────────────────────────────────────────────────────────────
# Built-in agent types
# ───────────────────────────────────────────────────────────────────────────

CODE_REVIEWER = AgentType(
    name="code_reviewer",
    description=(
        "Reviews code quality, patterns, and architecture. Reports "
        "findings with file:line references and names which coding "
        "principle each violates."
    ),
    system_preamble=(
        "You are a code reviewer. Analyze code for quality, patterns, "
        "and architecture issues. Report findings with file:line "
        "references and name which coding principle each finding violates.\n\n"
        "READ-ONLY MODE — you cannot create, modify, or delete files. "
        "Use your tools only to read and search the codebase."
    ),
    prompt_blocks=(
        PromptBlock.WORKFLOW,
        PromptBlock.PLANNING,
        PromptBlock.CODE_QUALITY,
        PromptBlock.TONE_AND_STYLE,
        PromptBlock.OUTPUT_EFFICIENCY,
    ),
    tools=(
        ToolName.READ_FILE,
        ToolName.GLOB_FILES,
        ToolName.GREP_FILES,
        ToolName.RUN_BASH,
    ),
)

BUG_HUNTER = AgentType(
    name="bug_hunter",
    description=(
        "Finds bugs through hypothesis-driven testing. Forms concrete "
        "breakage scenarios, writes and executes tests, decides from "
        "results not prior belief."
    ),
    system_preamble=(
        "You are a bug hunter. Find bugs through hypothesis-driven testing. "
        "Form concrete breakage scenarios, write tests that attempt the "
        "breakage, and decide from results — not from prior belief.\n\n"
        "READ-ONLY MODE — you cannot create, modify, or delete files. "
        "Use run_python to execute test scripts. Use run_bash and read "
        "tools only to read and search the codebase."
    ),
    prompt_blocks=(
        PromptBlock.WORKFLOW,
        PromptBlock.PLANNING,
        PromptBlock.BUG_HUNTING,
        PromptBlock.EXPERIMENTATION,
        PromptBlock.TONE_AND_STYLE,
        PromptBlock.OUTPUT_EFFICIENCY,
    ),
    tools=(
        ToolName.READ_FILE,
        ToolName.GLOB_FILES,
        ToolName.GREP_FILES,
        ToolName.RUN_BASH,
        ToolName.RUN_PYTHON,
    ),
)

AGENT_TYPES: dict[str, AgentType] = {
    CODE_REVIEWER.name: CODE_REVIEWER,
    BUG_HUNTER.name: BUG_HUNTER,
}


def get_agent_type(name: str) -> AgentType:
    """Look up an agent type by name. Raises ValueError on unknown name."""
    if name not in AGENT_TYPES:
        raise ValueError(f"Unknown agent type: {name!r}")
    return AGENT_TYPES[name]
