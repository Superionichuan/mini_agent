"""The System line names a Python command that the bash tool's shell can run as it is, and the
line stays one line, so that offline replay keeps dropping it before matching."""
import importlib
import os
import subprocess

import pytest

from mini_agent.llm.offline import normalize

shellmod = importlib.import_module("mini_agent.tools.shell.bash")


def no_python_on_path(monkeypatch, executable):
    monkeypatch.setattr(shellmod, "WINDOWS", False)
    monkeypatch.setattr(shellmod.shutil, "which", lambda name: None)
    monkeypatch.setattr(shellmod.sys, "executable", str(executable))


@pytest.mark.skipif(os.name == "nt", reason="a POSIX shell runs the command")
def test_the_fallback_interpreter_runs_even_with_a_dollar_in_its_path(monkeypatch, tmp_path):
    exe = tmp_path / "venv$NOT_SET" / "python"
    exe.parent.mkdir()
    exe.write_text('printf "started"\n')
    exe.chmod(0o700)
    monkeypatch.delenv("NOT_SET", raising=False)
    no_python_on_path(monkeypatch, exe)
    done = subprocess.run(["bash", "-c", shellmod.python_command("bash")],
                          capture_output=True, text=True)
    assert (done.returncode, done.stdout) == (0, "started"), done.stderr


def test_powershell_gets_the_fallback_interpreter_as_a_literal_string(monkeypatch):
    no_python_on_path(monkeypatch, "C:/Users/O'Neil/$env/python.exe")
    assert shellmod.python_command("powershell") == "& 'C:/Users/O''Neil/$env/python.exe'"


def test_the_system_line_stays_one_line(monkeypatch, tmp_path):
    no_python_on_path(monkeypatch, tmp_path / "line\nbreak" / "python")
    monkeypatch.setattr(shellmod, "shell", lambda: ("bash", "bash"))
    line = shellmod.describe_system()
    assert len(line.splitlines()) == 1, repr(line)
    before = "Workspace: here\nSystem: Linux; the bash tool runs bash\n\n[Goal]\nsomething\n"
    after = before.replace("Linux; the bash tool runs bash", line)
    assert normalize(before) == normalize(after)


@pytest.mark.skipif(os.name == "nt", reason="a POSIX shell runs the command")
def test_the_printed_command_runs_a_path_with_a_line_break(monkeypatch, tmp_path):
    exe = tmp_path / "line\nbreak" / "python"
    exe.parent.mkdir()
    exe.write_text("printf started\n")
    exe.chmod(0o700)
    no_python_on_path(monkeypatch, exe)
    monkeypatch.setattr(shellmod, "shell", lambda: ("bash", "bash"))
    command = shellmod.describe_system().split("; Python: ", 1)[1]
    assert command.isprintable(), repr(command)
    done = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert (done.returncode, done.stdout) == (0, "started"), (command, done.stderr)


def test_sh_is_not_given_a_command_it_cannot_run(monkeypatch, tmp_path):
    no_python_on_path(monkeypatch, tmp_path / "line\nbreak" / "python")
    monkeypatch.setattr(shellmod, "shell", lambda: ("sh", "sh"))
    assert shellmod.describe_system().endswith("; Python: none on PATH")


def test_a_backslash_in_a_posix_path_is_kept(monkeypatch):
    no_python_on_path(monkeypatch, "/opt/odd\\name/bin/python")
    assert shellmod.python_command("bash") == "'/opt/odd\\name/bin/python'"
