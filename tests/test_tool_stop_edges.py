"""Stopping a tool's command in the awkward cases: a second stop while the first is being handled,
a taskkill that cannot start or fails (the Windows branch, driven on POSIX with WINDOWS set and
taskkill replaced), and a process that left the command's group but still holds its output pipes.

POSIX only: real processes and signals.
"""
import importlib
import os
import signal
import subprocess
import sys
import time

import pytest

from mini_agent.ui.common import StopSignal

mod = importlib.import_module("mini_agent.tools.shell.bash")

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX processes and signals")

REAL_POPEN = subprocess.Popen
killers = []         # the real wait of each fake taskkill whose wait is made to fail

IGNORING = """import signal, time
for name in ("SIGINT", "SIGTERM", "SIGHUP"):
    signal.signal(getattr(signal, name), signal.SIG_IGN)
print("ready", flush=True)
time.sleep(60)
"""


def reap(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.kill()
    proc.communicate(timeout=5)


def spawn_then_stop(monkeypatch, first_stop, taskkill=None):
    """Run the tool's command for real, but end the first wait for it with `first_stop` once the
    command has printed "ready". taskkill: None, "cannot start", "fails" (exit status 1) or
    "wait fails" (waiting for it raises OSError)."""
    commands = []

    def popen(argv, **kwargs):
        kwargs.pop("creationflags", None)
        if argv[0] == "taskkill":
            if taskkill == "cannot start":
                raise OSError("taskkill could not start")
            killer = REAL_POPEN([sys.executable, "-c", "raise SystemExit(1)"], **kwargs)
            if taskkill == "wait fails":
                real_wait = killer.wait
                killers.append(real_wait)

                def failed_wait(*args, **kw):
                    raise OSError("waiting for taskkill failed")
                killer.wait = failed_wait
            return killer
        kwargs["start_new_session"] = True
        proc = REAL_POPEN(argv, **kwargs)
        commands.append(proc)
        communicate, first = proc.communicate, [True]

        def interrupted(*args, **kw):
            if first[0]:
                first[0] = False
                assert proc.stdout.readline() == b"ready\n"
                raise first_stop
            return communicate(*args, **kw)

        proc.communicate = interrupted
        return proc

    monkeypatch.setattr(mod.subprocess, "Popen", popen)
    return commands


def windows_branch(monkeypatch):
    monkeypatch.setattr(mod, "WINDOWS", True)
    monkeypatch.setattr(mod.subprocess, "CREATE_NEW_PROCESS_GROUP", 0, raising=False)


def test_a_second_stop_while_the_first_signal_is_sent_still_ends_the_command(monkeypatch):
    commands = spawn_then_stop(monkeypatch, StopSignal(signal.SIGTERM))
    real_signal_group = mod._signal_group

    def signal_group(pgid, sig):
        sent = real_signal_group(pgid, sig)
        if sig == signal.SIGTERM:
            os.kill(os.getpid(), signal.SIGINT)      # a real Ctrl+C right after the first signal
        return sent

    monkeypatch.setattr(mod, "_signal_group", signal_group)
    old = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with pytest.raises(KeyboardInterrupt):
            mod.run_tool([sys.executable, "-c", IGNORING], timeout=10)
        assert commands[0].poll() is not None, "the command still runs after the second stop"
    finally:
        signal.signal(signal.SIGINT, old)
        for proc in commands:
            reap(proc)


def test_a_taskkill_that_cannot_start_keeps_the_stop_and_ends_the_command(monkeypatch):
    stop = StopSignal(signal.SIGTERM)
    commands = spawn_then_stop(monkeypatch, stop, taskkill="cannot start")
    windows_branch(monkeypatch)
    try:
        with pytest.raises(StopSignal) as raised:
            mod.run_tool([sys.executable, "-c", IGNORING], timeout=10)
        assert raised.value is stop
        assert commands[0].poll() is not None, "the command still runs"
    finally:
        for proc in commands:
            reap(proc)


def test_a_failing_taskkill_keeps_the_stop_and_ends_the_command(monkeypatch):
    """taskkill exits 1 while the command still runs: the command itself is ended. (What it
    started can only be reached through taskkill on Windows; see the README.)"""
    stop = StopSignal(signal.SIGTERM)
    commands = spawn_then_stop(monkeypatch, stop, taskkill="fails")
    windows_branch(monkeypatch)
    try:
        with pytest.raises(StopSignal) as raised:
            mod.run_tool([sys.executable, "-c", IGNORING], timeout=10)
        assert raised.value is stop
        assert commands[0].poll() is not None, "the command still runs"
    finally:
        for proc in commands:
            reap(proc)


def test_a_taskkill_whose_wait_fails_keeps_the_stop_and_ends_the_command(monkeypatch):
    """Waiting for taskkill raises OSError: the command itself is still ended, and the stop
    that came in is what goes on (not the OSError)."""
    stop = StopSignal(signal.SIGTERM)
    commands = spawn_then_stop(monkeypatch, stop, taskkill="wait fails")
    windows_branch(monkeypatch)
    try:
        with pytest.raises(StopSignal) as raised:
            mod.run_tool([sys.executable, "-c", IGNORING], timeout=10)
        assert raised.value is stop
        assert commands[0].poll() is not None, "the command still runs"
    finally:
        for proc in commands:
            reap(proc)
        while killers:
            killers.pop()(timeout=5)


def test_a_process_that_left_the_group_does_not_hold_up_the_stop(monkeypatch, tmp_path):
    pidfile = tmp_path / "escaped.pid"
    code = f"""import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if os.fork() == 0:
    os.setsid()
    open({str(pidfile)!r}, "w").write(str(os.getpid()))
    time.sleep(60)
else:
    while not os.path.exists({str(pidfile)!r}):
        time.sleep(0.01)
    print("ready", flush=True)
    time.sleep(60)
"""
    commands = spawn_then_stop(monkeypatch, StopSignal(signal.SIGTERM))
    start = time.monotonic()
    try:
        with pytest.raises(StopSignal):
            mod.run_tool([sys.executable, "-c", code], timeout=10)
        elapsed = time.monotonic() - start
        assert elapsed < mod.GRACE + 1, f"the stop took {elapsed:.2f} s"
        assert commands[0].poll() is not None
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        for proc in commands:
            reap(proc)
