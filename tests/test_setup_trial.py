"""The "Try it:" line that setup prints can be pasted into a terminal as it is: the shell gets the
suggested task and nothing else, and offline the pasted line answers."""
import importlib
import io
import os
import shlex
import subprocess
import sys

import pytest

setup = importlib.import_module("mini_agent.cli.setup")


def trial_line(provider, monkeypatch, capsys):
    monkeypatch.setattr(setup, "stdin_is_terminal", lambda: False)
    monkeypatch.setattr(setup.sys, "stdin", io.StringIO("test-key\n"))
    assert setup.main(["--provider", provider, "--model", "test-model", "--key-stdin",
                       "--base-url", "http://localhost.invalid/v1", "--no-test"]) == 0
    line = next(s for s in capsys.readouterr().out.splitlines() if s.startswith("Try it:"))
    return line.split("Try it:", 1)[1].strip()


@pytest.mark.parametrize("provider", list(setup.BY_NAME))
def test_the_pasted_trial_line_is_the_suggested_task(provider, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("MINI_AGENT_HOME", str(tmp_path / "home"))
    task = "Compute 3**7." if provider == "offline" else "What is 2**10?"
    assert shlex.split(trial_line(provider, monkeypatch, capsys)) == ["mini_agent", task]


def test_the_pasted_offline_trial_line_answers(monkeypatch, capsys, tmp_path):
    home = str(tmp_path / "home")
    monkeypatch.setenv("MINI_AGENT_HOME", home)
    argv = shlex.split(trial_line("offline", monkeypatch, capsys))[1:]
    work = tmp_path / "work"
    work.mkdir()
    # only the saved settings count: nothing from this process's environment
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "USERPROFILE": str(tmp_path),
           "MINI_AGENT_HOME": home, "PYTHONIOENCODING": "utf-8",
           "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    done = subprocess.run([sys.executable, "-m", "mini_agent", *argv], cwd=work, env=env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "2187" in done.stdout
