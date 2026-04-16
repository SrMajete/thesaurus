"""Query processor — the agentic execution logic.

Implements the fundamental loop: call the LLM → execute requested tools →
feed results back → repeat until the LLM produces a final ``send_response``
call.

Tool calls are partitioned into batches: consecutive read-only tools run
in parallel, consecutive non-read-only tools run serially with permission
checks. Order is always preserved.

``make_plan`` and ``send_response`` are special: their input fields are
streamed live to the user during the API call (see
``api_client._STREAM_FIELDS_BY_TOOL``), so the processor doesn't render
them after the fact. The processor's job for these tools is to persist
the plan on the agent (for next-turn system prompt injection) and to emit
``tool_result`` blocks for message history consistency — every
``tool_use`` needs a matching ``tool_result`` or the next API call is
rejected.

This module is pure logic — it reads and writes state on the Agent instance
it receives, but owns no state itself.
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from .api_client import ToolCall, stream_response
from .messages import assistant_message, tool_result, user_message
from .prompts import build_system_prompt
from .tools.base import Tool, ToolName, find_tool

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Agent loop
# ───────────────────────────────────────────────────────────────────────────


async def run_loop(agent: "Agent") -> None:
    """Run the agentic loop until the LLM produces a final response.

    Reads and writes ``agent.messages``. Uses ``agent.callbacks`` for I/O.
    ``agent.max_turns`` prevents infinite tool loops.

    Invariant protected by a ``try/finally`` around the body: every
    ``tool_use`` block in an appended assistant message must have a
    matching ``tool_result`` in the next user message. If an exception
    (``KeyboardInterrupt`` mid-execution, a callback raising, etc.)
    would otherwise orphan the ``tool_use`` blocks, the finally stubs
    them with error ``tool_result``s so the session survives the next
    API call. The exception still propagates.
    """
    for turn in range(agent.max_turns):
        logger.info("═══ Turn %d/%d ═══", turn + 1, agent.max_turns)
        logger.debug(
            "Latest message:\n%s",
            json.dumps(agent.messages[-1], indent=2, default=str),
        )

        # Rebuild system prompt each turn so dynamic context stays fresh.
        # Pass the current plan so it gets injected as a dynamic section,
        # giving the model a live view of its own roadmap across turns.
        # env_info is cached on the agent — see Agent.__init__ — so we
        # don't spawn git subprocesses every turn.
        system = build_system_prompt(
            env_info=agent.env_info, current_plan=agent.plan
        )

        response = await stream_response(
            client=agent.client,
            model=agent.model,
            messages=agent.messages,
            system=system,
            tools=agent.api_tools,
            on_text=agent.callbacks.on_text,
            on_thinking=agent.callbacks.on_thinking,
            on_tool_start=agent.callbacks.on_tool_start,
        )

        # Accumulate session-wide token usage. Input sums fresh input +
        # cache writes + cache reads — all tokens the model processed.
        # The cached total tracks the cache_creation + cache_read
        # portion separately so UIs can surface cache efficiency.
        agent.total_input_tokens += (
            response.input_tokens
            + response.cache_creation_input_tokens
            + response.cache_read_input_tokens
        )
        agent.total_cached_input_tokens += (
            response.cache_creation_input_tokens
            + response.cache_read_input_tokens
        )
        agent.total_output_tokens += response.output_tokens

        agent.messages.append(assistant_message(response.content))

        # No tool calls → done (fallback, shouldn't happen with tool_choice: any)
        if not response.tool_calls:
            return

        # Guard the stretch between the assistant-message append above and
        # the user-message append below: if anything raises in between
        # (KeyboardInterrupt, a callback, a tool, _execute_tool_calls), we
        # owe the API a user message with tool_results — otherwise the
        # next call 400s on the orphaned tool_use blocks.
        results_appended = False
        try:
            # Extract all make_plan calls — the prompt forbids more than one
            # per turn, but handle the misbehaving case so every tool_use gets
            # a matching tool_result and the next API call isn't rejected.
            plan_calls = [c for c in response.tool_calls if c.name == ToolName.MAKE_PLAN]
            if plan_calls:
                # Use the first plan's content for agent.plan (predictable if the
                # model misbehaves and emits multiple). The streaming path in
                # api_client.py already rendered each plan's content live.
                first = plan_calls[0]
                agent.plan = _format_plan(
                    first.input.get("thinking", ""),
                    first.input.get("roadmap", ""),
                )
                if agent.plan:
                    logger.info("Plan: %s", agent.plan)

            # Check for send_response tool calls — the "I'm done" signal
            respond_calls = [c for c in response.tool_calls if c.name == ToolName.SEND_RESPONSE]
            if respond_calls:
                # The send_response header and body were streamed live during
                # the API call. All that's left is to emit tool_results so the
                # message history stays valid — every tool_use block needs its
                # match.
                respond_ids = {c.id for c in respond_calls}
                plan_ids = {c.id for c in plan_calls}
                all_results = []
                for call in response.tool_calls:
                    if call.id in respond_ids:
                        response_text = call.input.get("response", "")
                        all_results.append(tool_result(call.id, response_text))
                    elif call.id in plan_ids:
                        all_results.append(tool_result(call.id, "Plan acknowledged."))
                    else:
                        # Model violated the prompt and emitted other tools
                        # alongside send_response. Fire both callbacks so the
                        # user sees the dropped tool. on_tool_start is
                        # required first: the TUI asserts every tool_result
                        # has a matching registered tool_start.
                        skipped = "Skipped — send_response ended the turn."
                        agent.callbacks.on_tool_start(call.id, call.name, call.input)
                        agent.callbacks.on_tool_result(
                            call.id, call.name, call.input, skipped, True
                        )
                        all_results.append(tool_result(call.id, skipped))
                agent.messages.append(user_message(all_results))
                results_appended = True
                return

            # Execute remaining tools (make_plan already handled above)
            remaining_calls = [c for c in response.tool_calls if c.name != ToolName.MAKE_PLAN]
            plan_results = [tool_result(c.id, "Plan acknowledged.") for c in plan_calls]

            # Execute remaining tool calls
            tool_results = await _execute_tool_calls(agent, remaining_calls) if remaining_calls else []
            agent.messages.append(user_message(plan_results + tool_results))
            results_appended = True
        finally:
            if not results_appended:
                # Preserve the tool_use ↔ tool_result invariant so the next
                # API call isn't rejected. The exception still propagates.
                stub_results = [
                    tool_result(
                        c.id,
                        "Interrupted before execution completed.",
                        is_error=True,
                    )
                    for c in response.tool_calls
                ]
                agent.messages.append(user_message(stub_results))

    # max_turns exhausted — logger.warning surfaces this to stderr via
    # the stderr handler, and the assistant message makes the model aware.
    message = f"Reached max_turns ({agent.max_turns}) — stopping."
    logger.warning(message)
    agent.messages.append(assistant_message([{"type": "text", "text": message}]))


# ───────────────────────────────────────────────────────────────────────────
# Tool execution with parallel read-only support
# ───────────────────────────────────────────────────────────────────────────


def _format_plan(thinking: str, roadmap: str) -> str:
    """Combine the make_plan tool's thinking and roadmap into a single
    markdown block for injection into the next turn's system prompt.

    CLI display is handled separately by the streaming renderer in
    ``api_client.py``, which prints ``make_plan.thinking`` and
    ``make_plan.roadmap`` deltas live as the model writes them — this
    function is not involved in display. Keeping the format stable here
    means the model always sees its plan in the same shape it wrote it.
    """
    parts: list[str] = []
    if thinking:
        parts.append(f"## Thinking\n\n{thinking.strip()}")
    if roadmap:
        parts.append(f"## Roadmap\n\n{roadmap.strip()}")
    return "\n\n".join(parts)


def _strip_reason(params: dict[str, Any]) -> dict[str, Any]:
    """Drop the auto-injected ``reason`` key before unpacking into a tool's
    ``execute()`` (which doesn't accept it as a kwarg).
    """
    return {k: v for k, v in params.items() if k != "reason"}


async def _execute_tool_calls(
    agent: "Agent", tool_calls: list[ToolCall]
) -> list[dict[str, Any]]:
    """Execute tool calls in batches grouped by dispatch flags.

    Consecutive calls with the same ``(needs_permission, is_parallelizable)``
    flag pair form a batch. For each batch: Phase 1 collects a decision
    (asking permission if the bucket requires it), Phase 2 executes the
    approved decisions (in parallel if the bucket allows it). Original
    call order is preserved both within and across batches.
    """
    results: list[dict[str, Any]] = []
    for (needs_permission, is_parallelizable), batch in _partition(agent, tool_calls):
        decisions = await _collect_decisions(
            agent, batch, needs_permission=needs_permission
        )
        results.extend(await _execute_decisions(
            agent, decisions,
            needs_permission=needs_permission,
            is_parallelizable=is_parallelizable,
        ))
    return results


def _partition(
    agent: "Agent", tool_calls: list[ToolCall]
) -> list[tuple[tuple[bool, bool], list[ToolCall]]]:
    """Group consecutive tool calls with the same ``(needs_permission,
    is_parallelizable)`` flag pair into batches.

    Preserves the original call order; only consecutive same-bucket calls
    are merged. Unknown tools default to ``(True, False)`` — the
    approve-serial bucket — so they flow through the permission-asking
    path and surface their 'unknown tool' error in the clearest place.

    Example with ``[read, read, python, bash, write, read]``:
      1. ``[read, read]``    → ``(False, True)``   auto-parallel
      2. ``[python, bash]``  → ``(True,  True)``   approve-parallel
      3. ``[write]``         → ``(True,  False)``  approve-serial
      4. ``[read]``          → ``(False, True)``   auto-parallel
    """
    batches: list[tuple[tuple[bool, bool], list[ToolCall]]] = []
    for call in tool_calls:
        tool = find_tool(call.name, agent.tools)
        key: tuple[bool, bool] = (
            (tool.needs_permission, tool.is_parallelizable)
            if tool is not None
            else (True, False)
        )
        if batches and batches[-1][0] == key:
            batches[-1][1].append(call)
        else:
            batches.append((key, [call]))
    return batches


async def _collect_decisions(
    agent: "Agent",
    calls: list[ToolCall],
    *,
    needs_permission: bool,
) -> list[tuple[ToolCall, Tool | None, bool]]:
    """Phase 1. Build a ``(call, tool, allowed)`` tuple for every call.

    On the auto path (``needs_permission=False``), every known tool is
    auto-approved. On the approve path, ``ask_permission`` runs for each
    known tool in batch order — the user sees all prompts upfront before
    any execution starts. Unknown tools are captured as
    ``(call, None, False)`` in both paths so Phase 2 can emit a matching
    tool_result for every tool_use block the model sent (the API rejects
    the next turn otherwise).

    Batching permissions upfront is a Fail-Fast win: the user can abort
    mid-approval (Ctrl+C → ``ask_permission`` returns False, denying all
    remaining calls) before any tool has executed.
    """
    decisions: list[tuple[ToolCall, Tool | None, bool]] = []
    for call in calls:
        tool = find_tool(call.name, agent.tools)
        if tool is None:
            logger.warning("Unknown tool requested: %s", call.name)
            decisions.append((call, None, False))
            continue
        if needs_permission:
            allowed = await agent.callbacks.ask_permission(
                call.id, call.name, call.input
            )
            logger.info(
                "Permission for %s: %s",
                call.name,
                "granted" if allowed else "denied",
            )
        else:
            allowed = True
        decisions.append((call, tool, allowed))
    return decisions


async def _execute_decisions(
    agent: "Agent",
    decisions: list[tuple[ToolCall, Tool | None, bool]],
    *,
    needs_permission: bool,
    is_parallelizable: bool,
) -> list[dict[str, Any]]:
    """Phase 2. Execute the decisions from Phase 1 and build tool_results.

    On the auto path (``needs_permission=False``), ``on_tool_start`` fires
    for every approved call to render the tool header. On the approve
    path, ``ask_permission`` in Phase 1 already rendered the header, so
    ``on_tool_start`` is skipped here to avoid printing it twice. Unknown
    and denied calls fire ``on_tool_result`` with their error message so
    the user sees a consistent ✗ line for every non-approved outcome.

    When ``is_parallelizable`` is True, ``run_one`` invocations run
    concurrently via ``asyncio.gather``; denied/unknown decisions still
    complete in one event-loop tick because their branches don't await
    anything, so they cost nothing to include. When False, calls run in
    batch order via a plain serial loop.

    KNOWN ISSUE — terminal output interleaving: when
    ``is_parallelizable`` is True and ``needs_permission`` is False, two
    concurrent ``on_tool_start``/``on_tool_result`` pairs can race the
    cursor and produce garbled terminal output. The fix buffers each
    task's output and drains it in original call order; deferred because
    the issue is cosmetic and has not been reproduced in a real session.
    """
    async def run_one(
        call: ToolCall, tool: Tool | None, allowed: bool
    ) -> tuple[str, bool]:
        if tool is None:
            err = f"Error: unknown tool '{call.name}'"
            agent.callbacks.on_tool_result(call.id, call.name, call.input, err, True)
            return err, True
        if not allowed:
            denied = "Permission denied by the user."
            agent.callbacks.on_tool_result(call.id, call.name, call.input, denied, True)
            return denied, True
        if not needs_permission:
            # Auto path: the header has not been rendered yet, so fire
            # on_tool_start. The approve path skips this because
            # ask_permission already rendered the header in Phase 1.
            agent.callbacks.on_tool_start(call.id, call.name, call.input)
        try:
            result = await tool.execute(**_strip_reason(call.input))
            is_error = False
        except Exception as e:
            logger.exception("Tool %s failed", call.name)
            result = f"Error: {e}"
            is_error = True
        agent.callbacks.on_tool_result(call.id, call.name, call.input, result, is_error)
        return result, is_error

    if is_parallelizable:
        completed = await asyncio.gather(
            *(run_one(call, tool, allowed) for call, tool, allowed in decisions)
        )
    else:
        completed = [
            await run_one(call, tool, allowed)
            for call, tool, allowed in decisions
        ]

    return [
        tool_result(call.id, result, is_error=is_error)
        for (call, _, _), (result, is_error) in zip(decisions, completed)
    ]
