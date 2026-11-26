"""Offline replay says so when no recording matches exactly and a near one is used."""
import json

from mini_agent.llm.offline import OfflineLLM, describe

PROMPT = ("Workspace: /w\n\n[Rules]\nUse one tool per step.\n\n[Goal]\nCompute 3**7.\n\n"
          "[Steps taken so far]\n1. calc({'expr': '3**7'}) -> 2187\n")
CHANGED = PROMPT.replace("Use one tool per step.", "Use one tool per step. Answer in French.")


def recording(tmp_path):
    rec = describe(PROMPT)
    rec.update(response='{"action": "done", "args": {"answer": "2187"}}', model="m")
    path = tmp_path / "rec.jsonl"
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    return path


def test_exact_match_is_silent(tmp_path, capsys):
    llm = OfflineLLM(recording(tmp_path))
    assert "2187" in llm.call(PROMPT)
    assert capsys.readouterr().err == ""


def test_near_match_is_announced_once_per_task(tmp_path, capsys):
    llm = OfflineLLM(recording(tmp_path))
    assert describe(CHANGED)["key"] != describe(PROMPT)["key"]
    assert "2187" in llm.call(CHANGED)            # still replays the recorded step
    err = capsys.readouterr().err
    assert "not recorded" in err and "closest recorded step" in err
    llm.call(CHANGED)
    assert capsys.readouterr().err == ""          # once per task, not every step
    OfflineLLM(recording(tmp_path)).call(CHANGED)  # a new model object tells again
    assert "not recorded" in capsys.readouterr().err


def test_each_run_of_the_same_task_is_told_again(tmp_path, capsys):
    """Once per run: the same Agent running the same task again gets the notice again, since
    that run's output needs it too."""
    from mini_agent import Agent
    from mini_agent.llm.base import BaseLLM

    done = json.dumps({"action": "done", "args": {"answer": "finished"}})

    class Recorder(BaseLLM):
        def __init__(self):
            self.rows = []

        def call(self, prompt):
            self.rows.append(dict(describe(prompt), response=done))
            return done

    recorder = Recorder()
    Agent(llm=recorder, workspace=tmp_path, skills=[], use_memory=False).run("a task", verbose=False)
    path = tmp_path / "rec.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recorder.rows), encoding="utf-8")
    agent = Agent(llm=OfflineLLM(path), workspace=tmp_path, skills=[], use_memory=False)
    agent.tool_desc += "\nChanged tool instructions."
    for _ in range(2):
        assert agent.run("a task", verbose=False) == "finished"
        assert "closest recorded step" in capsys.readouterr().err
