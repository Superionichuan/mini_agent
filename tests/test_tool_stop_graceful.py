"""A stopped tool command first gets the signal that matches the cause and GRACE seconds to clean
up (git removes its lock file, for example); whatever still runs after that is killed. A stop
that comes during that time still stops the task.

POSIX only (process groups and signals).
"""
import os
import signal
import time
import uuid

import pytest

from test_keepalive import Flaky, env_of, start, wait_for
from test_stop_after_timeout import call, finish, with_tool_timeout
from test_tool_stop import bash_module, running

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups and signals")


def marker() -> str:
    return f"sleep 30.{uuid.uuid4().int % 10**6:06d}"


def cleaning(sleeper: str) -> str:
    """A command that writes which signal it got to cleaned.txt, then exits; `ready` exists once
    its traps are set."""
    traps = "; ".join(f"trap 'echo {name} > cleaned.txt; exit 1' {name}"
                      for name in ("INT", "TERM", "HUP"))
    return f"{traps}; touch ready; {sleeper}; touch late"


def ignoring(sleeper: str) -> str:
    """A command that ignores INT, TERM and HUP (its children too), so only SIGKILL ends it."""
    return f"trap '' INT TERM HUP; touch ready; {sleeper}; touch late"


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP],
                         ids=lambda s: signal.Signals(s).name)
def test_a_stopped_command_gets_the_same_signal_first_and_cleans_up(tmp_path, sig):
    sleeper = marker()
    with Flaky([call("bash", cmd=cleaning(sleeper))]) as server:
        proc = start(["--yes", "run it"], env_of(server), tmp_path)
        try:
            assert wait_for(lambda: (tmp_path / "ready").exists() and running(sleeper), 60)
            t0 = time.time()
            proc.send_signal(sig)
            out, err = finish(proc)
            took = time.time() - t0
        finally:
            if proc.poll() is None:
                proc.kill()
    assert proc.returncode == 128 + sig, (proc.returncode, out[-600:], err[-600:])
    name = signal.Signals(sig).name[3:]
    assert (tmp_path / "cleaned.txt").read_text().strip() == name
    assert took < 4, took
    time.sleep(0.5)
    assert not running(sleeper)
    assert not (tmp_path / "late").exists()


def test_the_timeout_sends_sigterm_first(tmp_path, monkeypatch):
    sleeper = marker()
    monkeypatch.setattr(bash_module, "TIMEOUT", 1)
    monkeypatch.chdir(tmp_path)
    t0 = time.time()
    assert bash_module.bash(cleaning(sleeper)) == "Error: timeout"
    assert time.time() - t0 < 4
    assert (tmp_path / "cleaned.txt").read_text().strip() == "TERM"
    time.sleep(0.5)
    assert not running(sleeper) and not (tmp_path / "late").exists()


def test_a_command_that_ignores_the_signal_is_killed_after_the_grace_time(tmp_path, monkeypatch):
    sleeper = marker()
    monkeypatch.setattr(bash_module, "TIMEOUT", 1)
    monkeypatch.chdir(tmp_path)
    t0 = time.time()
    assert bash_module.bash(ignoring(sleeper)) == "Error: timeout"
    took = time.time() - t0
    assert 1 + bash_module.GRACE - 0.5 < took < 1 + bash_module.GRACE + 4, took
    time.sleep(0.5)
    assert not running(sleeper) and not (tmp_path / "late").exists()


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT],
                         ids=lambda s: signal.Signals(s).name)
def test_a_stop_during_the_grace_time_after_a_timeout_stops_the_task(tmp_path, sig):
    """The timeout ends the call at about 1 s; the command ignores SIGTERM, so the grace time
    lasts until about 3 s. A stop at about 1.8 s must stop the task: no further model call,
    no further tool call."""
    sleeper = marker()
    with Flaky([call("bash", cmd=ignoring(sleeper)), call("bash", cmd="touch after")]) as server:
        proc = start(["--yes", "run it"], with_tool_timeout(env_of(server), tmp_path, 1), tmp_path)
        try:
            assert wait_for(lambda: (tmp_path / "ready").exists(), 60)
            time.sleep(1.8)
            proc.send_signal(sig)
            out, err = finish(proc)
            requests = server.requests
        finally:
            if proc.poll() is None:
                proc.kill()
    assert proc.returncode == 128 + sig, (proc.returncode, out[-600:], err[-600:])
    assert requests == 1
    assert not (tmp_path / "after").exists()
    time.sleep(0.5)
    assert not running(sleeper)
