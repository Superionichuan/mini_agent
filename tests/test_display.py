"""What the screen shows of a tool call and a thought: the arguments in one short line (the
content of write_file by its size), and a note in brackets for a reply in the model's own
tool-call markup. The session log and the history the model sees keep both whole."""
import io
import json
from pathlib import Path

import pytest
from rich.console import Console

import mini_agent
from mini_agent.core.parser import parse_response
from mini_agent.ui import repl
from mini_agent.ui.common import ARG_TEXT_MAX, call_args_text, thought_text


def test_write_file_content_is_shown_by_its_size():
    content = "".join(f"print({i})\n" for i in range(24))
    assert (call_args_text({"path": "primes.py", "content": content})
            == f"path=primes.py, content: 24 lines, {len(content)} characters")
    assert call_args_text({"path": "one.txt", "content": "1"}) == \
        "path=one.txt, content: 1 line, 1 character"


def test_a_long_or_multi_line_string_shows_its_first_line_and_size():
    code = "import numpy as np\nx = np.arange(10)\nprint(x.sum())\n"
    assert call_args_text({"code": code}) == \
        f"code=import numpy as np … (3 lines, {len(code)} characters)"
    long = "echo " + "x" * 200
    assert call_args_text({"cmd": long}) == \
        f"cmd={long[:ARG_TEXT_MAX]} … (1 line, {len(long)} characters)"


def test_short_values_are_shown_as_they_are():
    assert call_args_text({"expr": "3**7"}) == "expr=3**7"
    assert call_args_text({"path": "a.txt", "n": 3}) == "path=a.txt, n=3"
    assert call_args_text({"cmd": "python primes.py\n"}) == "cmd=python primes.py"
    assert call_args_text({}) == ""
    assert call_args_text("") == ""


def test_other_long_values_are_cut():
    todos = [{"content": f"step {i}", "status": "pending"} for i in range(20)]
    text = call_args_text({"todos": todos})
    assert text == "todos=" + json.dumps(todos)[:ARG_TEXT_MAX] + " …"


def test_a_reply_in_tool_call_markup_is_shown_with_a_note():
    recording = Path(mini_agent.__file__).parent / "llm" / "recordings" / "tutorial.jsonl"
    replies = [json.loads(line)["response"]
               for line in recording.read_text(encoding="utf-8").splitlines() if line.strip()]
    markup = [r for r in replies if 'invoke name="' in r]
    assert markup, "the tutorial recording has a reply in tool-call markup"
    thought, action, _ = parse_response(markup[0])
    assert action                                  # the parser still reads the action
    shown = thought_text(thought)
    assert shown.endswith("[the reply was in the model's own tool-call markup; "
                          "the action was read from it]")
    assert "\uff5c" not in shown and "DSML" not in shown


def test_an_ordinary_thought_is_shown_as_it_is():
    assert thought_text("I will compute 3**7 with calc.") == "I will compute 3**7 with calc."
    assert thought_text("A table | with | bars") == "A table | with | bars"
    assert thought_text(None) == ""


def test_the_terminal_mode_prints_the_short_line():
    console = Console(file=io.StringIO(), width=200, color_system=None)
    content = "".join(f"line {i}\n" for i in range(50))
    repl._print_tool(console, {"name": "write_file", "args": {"path": "big.txt", "content": content}},
                     {"content": "OK: big.txt"})
    out = console.file.getvalue()
    assert f"write_file(path=big.txt, content: 50 lines, {len(content)} characters)" in out
    assert "line 49" not in out


def test_the_full_screen_mode_uses_the_same_texts():
    pytest.importorskip("textual")
    from mini_agent.ui import tui
    card = tui.ToolCard("bash", {"cmd": "echo one\necho two"})
    assert "bash(cmd=echo one … (2 lines, 17 characters))" in card.render().plain
    think = tui.ThinkMsg(1, "<\uff5cDSML\uff5cfunction_calls>", "1 in / 2 out")
    assert "[the reply was in the model's own tool-call markup" in think.render().plain
