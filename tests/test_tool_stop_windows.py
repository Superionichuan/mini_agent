"""Stopping a tool's command on Windows, with real processes: taskkill /T ends the command and
the process it started. When taskkill cannot start, exits with an error, or its wait fails, the
command itself is still ended and the stop goes on, while the process it started may keep
running (a documented limit).

Windows only: real process groups, taskkill and the Win32 process API. The failures are
injected; they say nothing about how often they happen.
"""
import ctypes
import importlib
import os
import subprocess
import sys
import time

import pytest

mod = importlib.import_module("mini_agent.tools.shell.bash")

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows only: taskkill and process groups")

STILL_ACTIVE = 259
QUERY_LIMITED_INFORMATION = 0x1000
TERMINATE = 0x0001


def _open(pid, access):
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.c_void_p
    return kernel32, kernel32.OpenProcess(access, False, pid)


def alive(pid):
    """True while the process with this id is running."""
    kernel32, handle = _open(pid, QUERY_LIMITED_INFORMATION)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        assert kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def ended_within(pid, seconds):
    end = time.monotonic() + seconds
    while alive(pid):
        if time.monotonic() > end:
            return False
        time.sleep(0.05)
    return True


def terminate(pid):
    kernel32, handle = _open(pid, TERMINATE)
    if handle:
        kernel32.TerminateProcess(ctypes.c_void_p(handle), 1)
        kernel32.CloseHandle(ctypes.c_void_p(handle))


@pytest.mark.parametrize("failure", ["none", "launch", "exit", "wait"],
                         ids=["taskkill_runs", "taskkill_cannot_start", "taskkill_fails",
                              "taskkill_wait_fails"])
def test_a_stop_ends_the_command_whatever_taskkill_does(monkeypatch, tmp_path, failure):
    real_popen = subprocess.Popen
    child_pid = tmp_path / "child.pid"
    ready = tmp_path / "ready"
    worker = ("import subprocess, sys, time\n"
              "from pathlib import Path\n"
              "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
              f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
              f"Path({str(ready)!r}).write_text('ready')\n"
              "time.sleep(60)\n")
    original = KeyboardInterrupt("a stop during the test")
    commands, killers, calls = [], [], []

    def popen(argv, **kw):
        calls.append((argv, kw.copy()))
        if argv[0] == "taskkill":
            if failure == "launch":
                raise OSError("taskkill could not start (injected)")
            if failure in ("exit", "wait"):
                p = real_popen([sys.executable, "-c", "raise SystemExit(1)"], **kw)
            else:
                p = real_popen(argv, **kw)
            killers.append((p, p.wait))
            if failure == "wait":
                def failing_wait(*a, **k):
                    raise OSError("waiting for taskkill failed (injected)")
                p.wait = failing_wait
            return p
        p = real_popen(argv, **kw)
        commands.append(p)
        communicate = p.communicate

        def stop_once_ready(*a, **k):          # the stop comes once the command has a child
            p.communicate = communicate
            end = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < end:
                time.sleep(0.02)
            assert ready.exists()
            raise original
        p.communicate = stop_once_ready
        return p

    monkeypatch.setattr(mod.subprocess, "Popen", popen)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            mod.run_tool([sys.executable, "-c", worker], timeout=15)
        assert caught.value is original                     # the stop goes on
        assert commands[0].poll() is not None               # the command itself has ended
        pid = int(child_pid.read_text())
        if failure == "none":
            assert ended_within(pid, 5)                       # taskkill /T reached the child
        else:
            assert alive(pid)                                 # the documented limit
        assert all(kw.get("creationflags", 0) & subprocess.CREATE_NEW_PROCESS_GROUP
                   for _, kw in calls)                       # the command and taskkill alike
    finally:
        if child_pid.exists():
            terminate(int(child_pid.read_text()))
        for p in commands:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=5)
        for p, wait in killers:
            p.wait = wait
            if p.poll() is None:
                p.kill()
            wait(timeout=5)
