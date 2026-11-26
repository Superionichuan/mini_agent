"""Run a shell command

Linux and macOS: bash (or /bin/sh where bash is missing).
Windows: Git Bash when Git for Windows is installed, otherwise PowerShell.
MINI_AGENT_SHELL=bash | sh | powershell | pwsh | <path to a shell> picks one.
Output is read as UTF-8, so text such as Chinese comes back intact.
"""
import base64
import os
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

from ..workspace import cwd

TIMEOUT = 120
WINDOWS = os.name == "nt"

# No progress records (e.g. "Preparing modules for first use"), no colour codes in messages
# (PowerShell 7); PowerShell writes in the console code page unless told otherwise
_PS_PREFIX = ("$ProgressPreference = 'SilentlyContinue'; "
              "if ($PSStyle) { $PSStyle.OutputRendering = 'PlainText' }; "
              "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
              "$OutputEncoding = [System.Text.Encoding]::UTF8; ")
_CLIXML = "#< CLIXML"
_PS_ESCAPE = re.compile(r"_x([0-9A-Fa-f]{4})_")
_PS_LABEL = {"warning": "WARNING: ", "verbose": "VERBOSE: ", "debug": "DEBUG: "}
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def from_clixml(err: str) -> str:
    """The text of PowerShell's error output.

    Started with -EncodedCommand, PowerShell writes its error, warning and progress streams
    to stderr as CLIXML. This keeps the text of the messages and drops progress records;
    anything else comes back unchanged.
    """
    text = err.lstrip("﻿ \t\r\n")
    if not text.startswith(_CLIXML):
        return err
    parts = []
    for block in text.split(_CLIXML):
        if not block.strip():
            continue
        try:
            root = ET.fromstring(block.strip())
        except ET.ParseError:
            return err
        for element in root:
            if element.tag.rsplit("}", 1)[-1] == "S" and element.text:
                piece = _PS_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), element.text)
                label = _PS_LABEL.get((element.get("S") or "").lower())
                if label:                    # one record per element, as the console shows it
                    piece = label + piece + ("" if piece.endswith("\n") else "\n")
                parts.append(piece)
    joined = "".join(parts).encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return _ANSI.sub("", joined).replace("\r\n", "\n")


def _git_bash() -> Optional[str]:
    """bash.exe from Git for Windows (not the WSL launcher in System32)."""
    found = shutil.which("bash")
    if found and not any(part in found.lower() for part in ("\\system32\\", "\\windowsapps\\")):
        return found
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"),
                 os.environ.get("LOCALAPPDATA") and str(Path(os.environ["LOCALAPPDATA"]) / "Programs")):
        if base and (Path(base) / "Git" / "bin" / "bash.exe").is_file():
            return str(Path(base) / "Git" / "bin" / "bash.exe")
    return None


def shell() -> Tuple[str, str]:
    """(kind, executable) of the shell commands run in; kind is bash, sh or powershell."""
    choice = os.environ.get("MINI_AGENT_SHELL", "").strip()
    if choice:
        low = Path(choice).name.lower()
        exe = shutil.which(choice) or choice
        if low.startswith(("powershell", "pwsh")):
            return "powershell", exe
        return ("sh" if low.startswith("sh") else "bash"), exe
    if WINDOWS:
        git = _git_bash()
        if git:
            return "bash", git
        return "powershell", shutil.which("pwsh") or shutil.which("powershell") or "powershell"
    found = shutil.which("bash")
    return ("bash", found) if found else ("sh", "/bin/sh")


def _store_placeholder(path: str) -> bool:
    """True for the Windows app alias python.exe that only opens the Microsoft Store."""
    return "\\windowsapps\\" in path.replace("/", "\\").lower()


def python_command(kind: Optional[str] = None) -> str:
    """The command that starts a real Python 3 in the shell the bash tool uses.

    On Windows the py launcher comes first, and the Microsoft Store placeholder
    python.exe is skipped. With no usable Python on PATH, it is the interpreter
    that runs mini_agent.
    """
    if kind is None:
        kind = shell()[0]
    if WINDOWS:
        if shutil.which("py"):
            return "py -3"
        for name in ("python", "python3"):
            found = shutil.which(name)
            if found and not _store_placeholder(found):
                return name
    else:
        for name in ("python3", "python"):
            if shutil.which(name):
                return name
    exe = sys.executable.replace("\\", "/")
    if kind == "powershell":                       # a literal string: no $ expansion
        return "& '" + exe.replace("'", "''") + "'"
    return shlex.quote(exe)                        # sh, bash and Git Bash


def describe_system() -> str:
    """One line for the prompt: the operating system, the shell the bash tool uses,
    and the command for Python. Offline replay drops this line before matching, so
    it must stay one line."""
    kind, exe = shell()
    if WINDOWS:
        if kind == "powershell":
            base = "Windows; the bash tool runs PowerShell commands (use PowerShell syntax)"
        else:
            base = "Windows; the bash tool runs Git Bash (Windows paths also work, e.g. C:/Users)"
    else:
        system = "macOS" if os.uname().sysname == "Darwin" else os.uname().sysname
        base = f"{system}; the bash tool runs {'bash' if kind == 'bash' else 'sh'}"
    line = f"{base}; Python: {python_command(kind)}"
    return "".join(c if c.isprintable() else "?" for c in line)   # one line, whatever the path


def command_line(cmd: str) -> list:
    """The argument list that runs cmd in the shell."""
    kind, exe = shell()
    if kind == "powershell":
        script = base64.b64encode((_PS_PREFIX + cmd).encode("utf-16-le")).decode("ascii")
        return [exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", script]
    return [exe, "-c", cmd]


def utf8_env() -> dict:
    """The environment for child processes: Python children print UTF-8 too."""
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return env


def decode(data: bytes) -> str:
    return (data or b"").decode("utf-8", errors="replace")


def run_tool(argv: list, timeout: float) -> Tuple[bytes, bytes]:
    """Run a tool's command in the agent's folder and return (stdout, stderr).

    The command gets no terminal input and a process group of its own. When the wait ends
    early (the timeout, Ctrl+C, a stop signal), the command is stopped together with every
    process it started, and the exception goes on (subprocess.TimeoutExpired for the timeout).
    """
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS
             else {"start_new_session": True})
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=cwd(), env=utf8_env(), **group)
    try:
        return proc.communicate(timeout=timeout)
    except BaseException as cause:
        late_stop = stop_tree(proc, cause)
        if late_stop is not None and not isinstance(cause, KeyboardInterrupt):
            raise late_stop        # a stop that came while the timed-out command was cleaned up
        raise


GRACE = 2.0     # seconds a stopped command has to clean up before it is killed (POSIX)
REAP = 0.5      # seconds to read what is left in a killed command's pipes


def stop_tree(proc: subprocess.Popen,
              cause: Optional[BaseException] = None) -> Optional[KeyboardInterrupt]:
    """Stop a tool's command and what it started (its process group; on Windows its process
    tree), then reap it.

    POSIX: the group first gets the signal that matches the cause (SIGINT for Ctrl+C, the stop
    signal itself for SIGTERM or SIGHUP, SIGTERM for the timeout), so that a program can clean
    up (git removes its lock file, for example); whatever still runs after GRACE seconds is
    killed. Windows: the tree is killed at once (taskkill /T); the command itself is killed
    too, which also covers a taskkill that cannot start or fails.

    A stop (Ctrl+C or a stop signal) that comes meanwhile ends the wait at once and is
    returned, so that the caller can raise it once the command is gone; the command is killed
    and reaped whatever comes in between."""
    stop = None
    try:
        stop = _ask_to_stop(proc, cause)
    except KeyboardInterrupt as e:       # a second stop while the first is handled: kill now
        stop = e
    except Exception:                    # asking failed (taskkill, say): the kill below still runs
        pass
    for _ in range(3):                   # kill and reap, even when yet another stop comes
        try:
            _kill(proc)
            return _reap(proc) or stop
        except KeyboardInterrupt as e:
            stop = stop or e
    return stop


def _ask_to_stop(proc: subprocess.Popen,
                 cause: Optional[BaseException]) -> Optional[KeyboardInterrupt]:
    """Windows: taskkill /T, before the parent dies (it follows the parent links). POSIX: the
    signal that matches the cause, then up to GRACE seconds for the group to end."""
    if WINDOWS:
        return _taskkill_tree(proc.pid)
    import signal
    if hasattr(cause, "signum"):                   # SIGTERM, SIGHUP (StopSignal)
        first = cause.signum
    elif isinstance(cause, KeyboardInterrupt):
        first = signal.SIGINT
    else:                                          # the timeout, or any other error
        first = signal.SIGTERM
    if _signal_group(proc.pid, first):
        return _wait_for_group(proc, GRACE)
    return None


def _kill(proc: subprocess.Popen) -> None:
    """Kill the command, and on POSIX its whole process group."""
    if not WINDOWS:
        import signal
        _signal_group(proc.pid, signal.SIGKILL)
    try:
        proc.kill()
    except OSError:
        pass


def _reap(proc: subprocess.Popen) -> Optional[KeyboardInterrupt]:
    """Read what is left in the killed command's pipes for at most REAP seconds (a process that
    left its group may still hold them open), then collect its exit status. A stop that comes
    meanwhile is returned."""
    stop = None
    try:
        proc.communicate(timeout=REAP)
    except KeyboardInterrupt as e:       # StopSignal included
        stop = e
    except BaseException:                # the timeout: something outside the group holds a pipe
        if not WINDOWS:                  # the reads ran in this thread, so the pipes can close
            for pipe in (proc.stdout, proc.stderr):
                try:
                    pipe.close()
                except Exception:
                    pass
    try:
        proc.wait(timeout=1)
    except KeyboardInterrupt as e:
        stop = stop or e
    except BaseException:
        pass
    return stop


def _signal_group(pgid: int, sig: int) -> bool:
    """Send sig to the process group; False when no process of it is left to signal."""
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _wait_for_group(proc: subprocess.Popen, seconds: float) -> Optional[KeyboardInterrupt]:
    """Wait until every process of the command's group has ended, at most `seconds`, reading
    its output meanwhile so that a full pipe does not hold it up. A Ctrl+C or stop signal
    ends the wait at once and is returned."""
    import time
    end = time.monotonic() + seconds
    try:
        proc.communicate(timeout=seconds)            # the command itself ends, its pipes close
        while time.monotonic() < end and _signal_group(proc.pid, 0):
            time.sleep(0.05)                         # processes it started, writing elsewhere
    except KeyboardInterrupt as e:                   # Ctrl+C, StopSignal
        return e
    except BaseException:
        pass
    return None


def _taskkill_tree(pid: int) -> Optional[KeyboardInterrupt]:
    """Kill a process tree on Windows. taskkill runs in a process group of its own, so that a
    second Ctrl+C on the console does not reach it, and it is waited for even when that Ctrl+C
    reaches this process (the Ctrl+C is returned instead)."""
    stop = None
    try:
        killer = subprocess.Popen(["taskkill", "/F", "/T", "/PID", str(pid)],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    except OSError:                      # taskkill could not start; the command is killed next
        return None
    for _ in range(3):
        try:
            killer.wait(timeout=30)
            break
        except KeyboardInterrupt as e:
            stop = stop or e
        except (subprocess.TimeoutExpired, OSError):   # the command is killed next in any case
            break
    return stop


def bash(cmd: str) -> str:
    """Run a shell command (bash; on Windows Git Bash, or PowerShell when Git Bash is missing)"""
    try:
        stdout, stderr = run_tool(command_line(cmd), TIMEOUT)
        output = decode(stdout) + from_clixml(decode(stderr))
        return output if output else "OK"
    except subprocess.TimeoutExpired:
        return "Error: timeout"
    except Exception as e:
        return f"Error: {e}"
