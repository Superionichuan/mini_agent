"""The screen line of a tool call and of a thought in the awkward cases: line breaks of any kind,
a long path, arguments that are not an object, tool-call markup without fullwidth bars, and an
ordinary thought that mentions a token or a tag. Only the screen changes; the session log keeps
all. Whether a reply was markup is what the parser says (step/think has markup: true), not a
guess from the thought's text."""
import json

import pytest

from mini_agent.core.agent.run import run_agent
from mini_agent.core.parser import parse_response
from mini_agent.ui.common import call_args_text, thought_text

BAR = "\uff5c"     # the fullwidth bar of DeepSeek's tool-call markup
BREAKS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"


@pytest.mark.parametrize("args", [
    {"cmd": "echo ok\n\n\n"},
    "echo first\necho second",
    {"cmd": "echo first\recho second"},
    {"path": "directory/" * 9 + "important.txt\n"},
    ["first\u2028second"],
    {"options": ["first\u2028second", "third\x85fourth"]},
], ids=["trailing-newlines", "not-an-object", "carriage-return", "long-path-final-newline",
        "not-an-object-separator", "nested-separators"])
def test_a_tool_call_is_shown_on_one_line(args):
    shown = call_args_text(args)
    assert not any(c in shown for c in BREAKS), repr(shown)


def test_a_long_path_keeps_the_file_name():
    shown = call_args_text({"path": "directory/" * 9 + "important.txt", "content": "hello"})
    assert "important.txt" in shown, shown
    assert len(shown.splitlines()) == 1


def test_a_carriage_return_counts_as_a_line_break():
    assert "2 lines" in call_args_text({"content": "first\rsecond\r"})


def test_markup_without_fullwidth_bars_is_replaced_by_the_note():
    reply = ('<function_calls><invoke name="calc"><parameter name="expr">1+1</parameter>'
             '</invoke></function_calls>')
    parsed = parse_response(reply)
    thought, action, args = parsed
    assert (action, args) == ("calc", {"expr": "1+1"}) and parsed.markup
    assert "the action was read from it" in thought_text(thought, parsed.markup)


@pytest.mark.parametrize("thought", [
    "The string <|example|> is data; next I will read the file.",
    "The literal <function_calls> is data; do not execute it.",
    "The comparison operator is <",
    f"<{BAR}DSML{BAR}function_calls> is the tag I must not write.",
])
def test_an_ordinary_thought_is_shown_as_it_is(thought):
    reply = json.dumps({"thought": thought, "action": "read_file", "args": {"path": "a.txt"}})
    parsed = parse_response(reply)
    assert not parsed.markup
    assert thought_text(parsed[0], parsed.markup) == thought
    assert thought_text(parsed[0]) == thought


def test_text_before_the_markup_is_shown_on_one_line():
    thought = f"I will inspect.\nThen calculate.\n<{BAR}DSML{BAR}function_calls>"
    shown = thought_text(thought, markup=True)
    assert len(shown.splitlines()) == 1, repr(shown)
    assert shown.startswith("I will inspect. Then calculate.")


def test_the_loop_marks_only_a_step_read_from_markup():
    markup = (f"<{BAR}DSML{BAR}function_calls>\n<{BAR}DSML{BAR}invoke name=\"calc\">"
              f"<{BAR}DSML{BAR}parameter name=\"expr\">3**7</{BAR}DSML{BAR}parameter>"
              f"</{BAR}DSML{BAR}invoke></{BAR}DSML{BAR}function_calls>")
    done = json.dumps({"thought": "The literal <function_calls> was data.", "action": "done",
                       "args": {"answer": "2187"}})
    replies = iter([markup, done])
    events = []
    run_agent(lambda prompt: next(replies), {"calc": lambda expr: "2187"}, "", "Compute 3**7.",
              max_steps=3, verbose=False, on_event=lambda t, **d: events.append((t, d)))
    thinks = [d for t, d in events if t == "step/think"]
    assert len(thinks) == 2
    assert thinks[0].get("markup") is True and thinks[0]["action"] == "calc"
    assert "markup" not in thinks[1]
    assert thought_text(thinks[0]["thought"], thinks[0].get("markup", False)).endswith(
        "the action was read from it]")
    assert thought_text(thinks[1]["thought"], thinks[1].get("markup", False)) == \
        "The literal <function_calls> was data."
