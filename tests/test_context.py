"""Tests for context pruning."""

from thesaurus.core.context import PLACEHOLDER, prune_tool_results


def _user_tool_results(*contents: str, is_error: bool = False) -> dict:
    """Build a user message with one tool_result per content string."""
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": f"id{i}", "content": c,
             **({"is_error": True} if is_error else {})}
            for i, c in enumerate(contents)
        ],
    }


def _assistant_msg(*names: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": f"id{i}", "name": n, "input": {}}
                    for i, n in enumerate(names)],
    }


class TestPruneToolResults:
    def test_empty_messages(self) -> None:
        assert prune_tool_results([]) == 0

    def test_no_tool_results_nothing_pruned(self) -> None:
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        assert prune_tool_results(messages) == 0
        assert messages[0]["content"] == "hello"

    def test_last_two_user_turns_protected(self) -> None:
        big = "A" * 2000
        messages = [
            {"role": "user", "content": "start"},
            _assistant_msg("read_file"),
            _user_tool_results(big),  # old — prunable
            _assistant_msg("read_file"),
            _user_tool_results(big),  # protected (2nd to last)
            _assistant_msg("read_file"),
            _user_tool_results(big),  # protected (last)
        ]
        saved = prune_tool_results(messages, protect_last_n_turns=2)
        assert saved > 0
        assert messages[2]["content"][0]["content"] == PLACEHOLDER
        assert messages[4]["content"][0]["content"] == big
        assert messages[6]["content"][0]["content"] == big

    def test_error_results_protected(self) -> None:
        big = "ERROR" * 500
        messages = [
            _assistant_msg("run_bash"),
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": big, "is_error": True},
            ]},
            _assistant_msg("read_file"),
            _user_tool_results("recent"),
            _assistant_msg("read_file"),
            _user_tool_results("latest"),
        ]
        prune_tool_results(messages, protect_last_n_turns=2)
        assert messages[1]["content"][0]["content"] == big  # error kept

    def test_short_content_not_pruned(self) -> None:
        """Content shorter than placeholder should not be pruned (would grow)."""
        short = "Plan acknowledged."
        messages = [
            _assistant_msg("make_plan"),
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": short},
            ]},
            _assistant_msg("read_file"),
            _user_tool_results("a"),
            _assistant_msg("read_file"),
            _user_tool_results("b"),
        ]
        prune_tool_results(messages, protect_last_n_turns=2)
        assert messages[1]["content"][0]["content"] == short

    def test_already_cleared_not_reprocessed(self) -> None:
        messages = [
            _assistant_msg("read_file"),
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": PLACEHOLDER},
            ]},
            _assistant_msg("read_file"),
            _user_tool_results("a"),
            _assistant_msg("read_file"),
            _user_tool_results("b"),
        ]
        saved = prune_tool_results(messages, protect_last_n_turns=2)
        assert saved == 0

    def test_non_list_user_content_skipped(self) -> None:
        messages = [
            {"role": "user", "content": "original user text"},  # string content
            _assistant_msg("read_file"),
            _user_tool_results("a"),
            _assistant_msg("read_file"),
            _user_tool_results("b"),
        ]
        prune_tool_results(messages, protect_last_n_turns=2)
        assert messages[0]["content"] == "original user text"

    def test_non_string_tool_result_content_skipped(self) -> None:
        """If tool_result content is a list of blocks (e.g. image + text), skip."""
        messages = [
            _assistant_msg("read_file"),
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "x",
                 "content": [{"type": "text", "text": "abc"}]},
            ]},
            _assistant_msg("read_file"),
            _user_tool_results("a"),
            _assistant_msg("read_file"),
            _user_tool_results("b"),
        ]
        saved = prune_tool_results(messages, protect_last_n_turns=2)
        assert saved == 0  # the list-content block wasn't pruned

    def test_tokens_saved_estimate(self) -> None:
        """Estimated savings ≈ len // 4."""
        content = "x" * 4000
        messages = [
            _assistant_msg("read_file"),
            _user_tool_results(content),
            _assistant_msg("read_file"),
            _user_tool_results("a"),
            _assistant_msg("read_file"),
            _user_tool_results("b"),
        ]
        saved = prune_tool_results(messages, protect_last_n_turns=2)
        assert saved == 4000 // 4

    def test_non_tool_result_blocks_skipped(self) -> None:
        """Other block types within a list-content user message are ignored."""
        big = "x" * 500
        messages = [
            _assistant_msg("read_file"),
            {"role": "user", "content": [
                {"type": "text", "text": big},  # not a tool_result
                {"type": "tool_result", "tool_use_id": "x", "content": big},
            ]},
            _assistant_msg("read_file"),
            _user_tool_results("a"),
            _assistant_msg("read_file"),
            _user_tool_results("b"),
        ]
        prune_tool_results(messages, protect_last_n_turns=2)
        assert messages[1]["content"][0]["text"] == big  # text block untouched
        assert messages[1]["content"][1]["content"] == PLACEHOLDER
