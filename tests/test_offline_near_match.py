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
