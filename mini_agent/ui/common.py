"""What the plain terminal mode and the full-screen interface share (no textual needed)."""
import json
import os
import re
from pathlib import Path
from typing import Optional

from .. import paths

# One JSONL event log per session (an interactive session or a one-shot task);
# `mini_agent -c` continues the latest one of the current folder.
SESS_DIR = paths.home() / "sessions"

PLAN_GUIDANCE = ("[Plan mode] You are in plan mode: explore and plan only; do not run tools "
                 "with side effects (file writes / mutating bash). Produce a complete step-by-step "
                 "plan first (markdown, starting with a # heading) and wait for user confirmation.\n")


def compact_notice(d: dict) -> str:
    """One line for a context/compact event (the lifecycle slimmed the earlier steps)."""
    return (f"✻ context compacted before step {d.get('step')}: {d.get('before_tokens', 0):,} → "
            f"{d.get('after_tokens', 0):,} tokens ({d.get('method', '?')})")


def stop_notice(d: dict) -> str:
    """The line after a task that stopped on its token budget, its step limit, a full
    context window, or an approval nobody could give; "" otherwise."""
    if d.get("reason") == "token_limit":
        return (f"This task used up its token budget ({d.get('detail', '')} tokens). "
                "`mini_agent -c` continues it with a new budget.")
    if d.get("reason") == "max_steps":
        return "This task reached its step limit. `mini_agent -c` continues it."
    if d.get("reason") == "needs_approval":
        return (f"This task needs {d.get('tool') or 'a tool that must be approved'}, and there is "
                "no terminal to ask for approval. `mini_agent -c --yes` continues it and allows "
                "it; or run `mini_agent -c` in a terminal to be asked.")
    if d.get("reason") == "context_full":
        return ("The next step of this task does not fit in the model's context window, even "
                "after compaction. Choose a model with a larger window and run `mini_agent -c`, "
                "or start a smaller task.")
    if d.get("reason") == "no_recording":
        return ("Offline mode only replays the tutorial's recorded tasks; run `mini_agent setup` "
                "to use a model.")
    if d.get("reason") == "llm_error" and re.search(r"\bHTTP 40[13]\b", str(d.get("answer", ""))):
        return "Check the key with `mini_agent doctor`, or run `mini_agent setup` again."
    return ""


def provider_label(llm) -> str:
    """The provider's name for the status line, as setup and doctor show it: openrouter,
    deepseek, openai_compat, claude, ollama, offline, ..."""
    if getattr(llm, "provider", None):
        return llm.provider
    name = type(llm).__name__.replace("LLM", "").lower()
    return {"openaicompat": "openai_compat"}.get(name, name)


def shows_answer(d: dict) -> bool:
    """False for a turn/end whose answer is replaced by its stop notice on the command line:
    an offline miss (the answer is the notebook's wording, for the library)."""
    return d.get("reason") != "no_recording"


def exit_status(reason) -> int:
    """The exit status of the command for a task that ended with this turn/end reason:
    0 done (the agent finished; its answer says whether it managed), 3 a risky call needs
    approval and nobody could give it, 130 interrupted, 1 any other stop."""
    return {"done": 0, "needs_approval": 3, "interrupted": 130}.get(reason, 1)


class StopSignal(KeyboardInterrupt):
    """SIGTERM or SIGHUP in a process that runs a task, raised like Ctrl+C: the running
    tool's process is stopped, the turn ends as interrupted, the exit status is 128 + signum."""

    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def stop_signals_raise() -> None:
    """In a process that runs a task: SIGTERM and SIGHUP (a service stop, a closed terminal,
    --keep-alive stopping its run) raise StopSignal instead of ending the process at once,
    so that the tool it is running stops with it (POSIX; main thread only)."""
    import signal

    def handler(signum, frame):
        raise StopSignal(signum)

    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name) and os.name != "nt":
            try:
                signal.signal(getattr(signal, name), handler)
            except (OSError, ValueError):     # not the main thread
                pass


def interrupted_status(e: BaseException) -> int:
    """The exit status after an interruption: 130 for Ctrl+C, 128 + the signal for StopSignal."""
    return 128 + e.signum if isinstance(e, StopSignal) else 130


def interrupted_line(e: BaseException) -> str:
    if isinstance(e, StopSignal):
        import signal
        return f"Stopped ({signal.Signals(e.signum).name}). `mini_agent -c` continues this task."
    return "Interrupted. `mini_agent -c` continues this task."


EXIT_BUSY = 4     # the session is in use by another mini_agent process


def hold_session(path) -> str:
    """Lock the session log for this process until it ends (one process per session):
    "" when held, else why not (another mini_agent process runs it)."""
    from ..session_lock import SessionBusy, SessionLock
    try:
        SessionLock(path).acquire()
        return ""
    except SessionBusy as e:
        return str(e)


def session_conversation(agent, log, plain_text, until_seq=None) -> str:
    """The earlier conversation of the session for the next task: with a lifecycle, within
    its token budget (summarised by the model only when pruning is not enough); without
    one, the old rule (compact the log when it passes 2,500 characters).
    until_seq: the conversation as it was up to that event, for a task continued with -c
    (it gets the text it started with; read only, nothing is compacted)."""
    if getattr(agent, "lifecycle", None) is not None:
        return agent.lifecycle.conversation(log, until_seq=until_seq)
    if until_seq is not None:
        return plain_text(until_seq=until_seq)
    if len(plain_text(limit=999999)) > 2500:
        log.compact(agent.llm)
    return plain_text()


def mark_message(agent, log, seq) -> None:
    """Tell the lifecycle which logged user message the next run answers (session file and
    seq), so user.md records it once, even when -c runs a task again that was sent but
    never started."""
    if getattr(agent, "lifecycle", None) is not None and seq is not None:
        agent.lifecycle.message_id = f"{Path(log.path).name}#{seq}"


def _session_order(path: Path):
    """Sort key: <seconds>[_<n>] names (new_session_path) by seconds, then n as a number
    (so _100 comes after _99); any other name by its modification time."""
    m = re.fullmatch(r"(\d+)(?:_(\d+))?", path.stem)
    if m:
        return int(m.group(1)), int(m.group(2) or 1), path.name
    return int(path.stat().st_mtime), 0, path.name


def folder_of(agent) -> str:
    """The agent's workspace as an absolute path: recorded with every user message, so
    that `mini_agent -c` continues only a session of the folder it is run in."""
    return str(Path(agent.workspace).resolve())


def same_folder(a, b) -> bool:
    """Two paths name the same folder (links resolved; case ignored on Windows)."""
    return os.path.normcase(os.path.realpath(str(a))) == os.path.normcase(os.path.realpath(str(b)))


def _scan(path: Path):
    """(holds a user message or a started turn, the folder its last user message was sent
    in, or None). A session that was opened and left without a message (an empty file, or
    only a setting such as plan mode) has nothing to continue."""
    import json
    found, folder = False, None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict) or event.get("type") not in ("user/message", "turn/start"):
                    continue
                found = True
                data = event.get("data")
                if event["type"] == "user/message" and isinstance(data, dict) \
                        and isinstance(data.get("workspace"), str):
                    folder = data["workspace"]
    except OSError:
        return False, None
    return found, folder


def _sessions(directory: Optional[Path] = None):
    """(path, folder) of every session log that holds a message, newest first."""
    directory = directory or SESS_DIR
    if not directory.is_dir():
        return
    files = [p for p in directory.glob("*.jsonl") if p.is_file() and p.stat().st_size > 0]
    for path in sorted(files, key=_session_order, reverse=True):
        found, folder = _scan(path)
        if found:
            yield path, folder


def _unfinished(path: Path) -> bool:
    from ..session import SessionLog
    return SessionLog(path).unfinished_turn() is not None


def latest_session(directory: Optional[Path] = None, workspace=None,
                   unfinished: bool = False) -> Optional[Path]:
    """The most recent session log that holds a user message, or None.
    workspace: only a session run in that folder (a log that does not record its folder
    never matches); unfinished: only one whose last task did not finish."""
    for path, folder in _sessions(directory):
        if workspace is not None and (folder is None or not same_folder(folder, workspace)):
            continue
        if unfinished and not _unfinished(path):
            continue
        return path
    return None


def latest_folder_elsewhere(workspace, directory: Optional[Path] = None,
                            unfinished: bool = True) -> Optional[str]:
    """The folder of the most recent session run in another folder than workspace (with an
    unfinished task, when unfinished), or None: named when -c has nothing to continue here."""
    for path, folder in _sessions(directory):
        if folder is None or same_folder(folder, workspace):
            continue
        if unfinished and not _unfinished(path):
            continue
        return folder
    return None


def nothing_to_continue(workspace, terminal: bool, directory: Optional[Path] = None) -> str:
    """Why `mini_agent -c` has nothing to continue in this folder, or "" when it has.
    On a terminal it continues this folder's latest session; with no terminal, only an
    unfinished task of this folder. It never acts on another folder: it names it."""
    if latest_session(directory, workspace=workspace, unfinished=not terminal) is not None:
        return ""
    what = "session" if terminal else "unfinished task"
    lines = [f"mini_agent -c: no {what} to continue in this folder ({Path(workspace).resolve()})."]
    elsewhere = latest_folder_elsewhere(workspace, directory, unfinished=True)
    if elsewhere is None and terminal:
        elsewhere = latest_folder_elsewhere(workspace, directory, unfinished=False)
        if elsewhere is not None:
            lines.append(f"The latest session is in {elsewhere}: "
                         "cd there and run `mini_agent -c`.")
    elif elsewhere is not None:
        lines.append(f"The latest unfinished task is in {elsewhere}: "
                     "cd there and run `mini_agent -c`.")
    if len(lines) == 1:
        lines.append('Start one with `mini_agent "task"` or `mini_agent`.')
    return "\n".join(lines)


def resume_state(unfinished: dict):
    """The State to continue an unfinished turn with (SessionLog.unfinished_turn()):
    the steps rebuilt from its events, in-turn compactions applied; a tool call that
    has no result is not run again (the model is told it did not finish). None for a
    pending task (its turn never began): run it as a new turn."""
    if unfinished.get("pending"):
        return None
    from ..core.state import State
    from ..lifecycle import restore_steps
    restored = restore_steps(unfinished["events"])
    if isinstance(restored, State):
        return restored
    return State(goal=unfinished["goal"], steps=list(restored))


def note_approval_now(state, unfinished: dict, approve, can_ask: bool) -> None:
    """A task that stopped because nobody could approve a risky call (needs_approval), now
    continued where that call can run: say so after the refused call's result, so that the
    model calls the tool again instead of working round it. approve: the policy of this run
    ("auto", "ask", "deny" or an Approval); can_ask: someone is there to answer "ask"."""
    if state is None or not state.steps or unfinished.get("reason") != "needs_approval":
        return
    policy = getattr(approve, "policy", approve)
    if policy == "auto":
        note = "Now it may run: the task was continued with risky tools allowed."
    elif policy == "ask" and can_ask:
        note = "Now it can be approved: the task was continued at a terminal, where the user is asked."
    else:
        return
    last = state.steps[-1]
    if "did not run: it needs approval" in str(last.get("result", "")):
        last["result"] = f"{last['result']}. {note} Call it again if it is still needed."


def new_session_path(directory: Optional[Path] = None) -> Path:
    """A fresh session log, named by the time so that the latest sorts last; a second
    session in the same second gets _02, _03, ... The empty file is created here, at once,
    so two sessions (in one program or two) never get the same name."""
    import time
    directory = directory or SESS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp, n = str(int(time.time())), 1
    while True:
        path = directory / (f"{stamp}.jsonl" if n == 1 else f"{stamp}_{n:02d}.jsonl")
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL))
            return path
        except FileExistsError:
            n += 1


# --- what the screen shows of a tool call and a thought (the session log keeps both whole) ---

ARG_TEXT_MAX = 80      # a longer string argument is shown by its first line and its size
MORE = "\u2026"          # the mark for text left out

# A reply in tool-call markup: the parser reads the action from the first `invoke name="..."`
# and the thought is the text before it, e.g. "<|DSML|function_calls>\n<|DSML|" with fullwidth
# bars (U+FF5C, sometimes doubled), or "<function_calls><" without bars. Whether a reply was
# markup is the parser's to say (step/think has markup: true); these only find where the
# markup's opening tags begin in such a thought.
_TAG = "<(?:[\uff5c|]{1,2}[A-Za-z_]+[\uff5c|]{1,2})?"
_CALLS = re.compile(_TAG + "function_calls>")         # the opening tag of the calls
_CUT = re.compile(_TAG + "\\s*$")                    # where the parser cut before `invoke`
_BARRED = re.compile("<[\uff5c|]{1,2}[A-Za-z_]+[\uff5c|]{1,2}")   # a tag with bars
_BREAKS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"   # every line break str.splitlines() knows


def _size(text: str) -> str:
    lines = len(text.splitlines())
    return (f"{lines} line{'' if lines == 1 else 's'}, "
            f"{len(text)} character{'' if len(text) == 1 else 's'}")


def _json_text(value) -> str:
    """A value that is not a string, as JSON on one line (json.dumps writes U+0085, U+2028 and
    U+2029 as they are, and each of them breaks a line on the screen)."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    for c in "\x85\u2028\u2029":
        text = text.replace(c, "\\u%04x" % ord(c))
    return text


def _value_text(value: str, key=None) -> str:
    """One string argument as one line: the content of write_file by its size; a short
    one-line value as it is; a long path cut in the middle, so that the file name stays; any
    other long or multi-line value by its first line and its size."""
    if key == "content":
        return f"{key}: {_size(value)}"
    lines = value.splitlines()
    line = lines[0] if lines else ""             # without the line break at its end
    if len(lines) <= 1 and len(line) <= ARG_TEXT_MAX:
        shown = line
    elif len(lines) <= 1 and key == "path":
        half = (ARG_TEXT_MAX - 1) // 2
        shown = line[:half] + MORE + line[-half:]
    else:
        first = next((line for line in lines if line.strip()), "")[:ARG_TEXT_MAX]
        shown = f"{first} {MORE} ({_size(value)})"
    return shown if key is None else f"{key}={shown}"


def call_args_text(args) -> str:
    """A tool call's arguments as one short line for the screen: name=value pairs (see
    _value_text for strings); other values are cut at ARG_TEXT_MAX characters. Arguments that
    are not an object are shown the same way as one value. The approval preview and the
    session log keep them whole."""
    if isinstance(args, str):
        text = _value_text(args)
    elif not isinstance(args, dict):
        text = _json_text(args)
        text = text if len(text) <= ARG_TEXT_MAX else text[:ARG_TEXT_MAX] + " " + MORE
    else:
        parts = []
        for key, value in args.items():
            if isinstance(value, str):
                parts.append(_value_text(value, key))
            else:
                value_text = _json_text(value)
                if len(value_text) > ARG_TEXT_MAX:
                    value_text = value_text[:ARG_TEXT_MAX] + " " + MORE
                parts.append(f"{key}={value_text}")
        text = ", ".join(parts)
    return "".join(" " if c in _BREAKS else c for c in text)    # one line, whatever came in


def thought_text(thought, markup: bool = False) -> str:
    """The thought as the screen shows it. With markup=True (the parser read the action from
    the model's own tool-call markup instead of the JSON the prompt asks for; the step/think
    event then has markup: true), the thought is only the markup's opening tags, after any
    text that came before them: the tags are replaced by a note in brackets, on one line. Any
    other thought is shown as it is. The session log keeps the reply as it was."""
    thought = "" if thought is None else str(thought)
    if not markup:
        return thought
    found = [m.start() for m in (_CALLS.search(thought), _CUT.search(thought),
                                 _BARRED.search(thought)) if m]
    start = min(found) if found else len(thought)
    before = " ".join(line.strip() for line in thought[:start].splitlines() if line.strip())
    note = "[the reply was in the model's own tool-call markup; the action was read from it]"
    return f"{before} {note}" if before else note
