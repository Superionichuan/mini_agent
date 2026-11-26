"""The screen line of a tool call and of a thought in the awkward cases: line breaks of any kind,
a long path, arguments that are not an object, tool-call markup without fullwidth bars, and an
ordinary thought that mentions a token. Only the screen changes; the session log keeps all."""
import json

import pytest

from mini_agent.core.parser import parse_response
from mini_agent.ui.common import call_args_text, thought_text

BAR = "\uff5c"     # the fullwidth bar of DeepSeek's tool-call markup


@pytest.mark.parametrize("args", [
    {"cmd": "echo ok\n\n\n"},
    "echo first\necho second",
    {"cmd": "echo first\recho second"},
], ids=["trailing-newlines", "not-an-object", "carriage-return"])
def test_a_tool_call_is_shown_on_one_line(args):
    shown = call_args_text(args)
    assert len(shown.splitlines()) <= 1, repr(shown)


def test_a_long_path_keeps_the_file_name():
    shown = call_args_text({"path": "directory/" * 9 + "important.txt", "content": "hello"})
    assert "important.txt" in shown, shown
    assert len(shown.splitlines()) == 1


def test_a_carriage_return_counts_as_a_line_break():
    assert "2 lines" in call_args_text({"content": "first\rsecond\r"})


def test_markup_without_fullwidth_bars_is_replaced_by_the_note():
    reply = ('<function_calls><invoke name="calc"><parameter name="expr">1+1</parameter>'
             '</invoke></function_calls>')
    thought, action, args = parse_response(reply)
    assert (action, args) == ("calc", {"expr": "1+1"})
    assert "the action was read from it" in thought_text(thought)


def test_an_ordinary_thought_that_mentions_a_token_is_kept():
    thought = "The string <|example|> is data; next I will read the file."
    reply = json.dumps({"thought": thought, "action": "read_file", "args": {"path": "a.txt"}})
    parsed, _, _ = parse_response(reply)
    assert thought_text(parsed) == thought


def test_text_before_the_markup_is_shown_on_one_line():
    thought = f"I will inspect.\nThen calculate.\n<{BAR}DSML{BAR}function_calls>"
    shown = thought_text(thought)
    assert len(shown.splitlines()) == 1, repr(shown)
    assert shown.startswith("I will inspect. Then calculate.")
