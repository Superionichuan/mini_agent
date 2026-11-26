"""The System line may carry anything on one line: offline replay ignores it."""
from mini_agent.llm.offline import normalize


def test_a_longer_system_line_keeps_the_replay_key():
    base = "Workspace: /w\nSystem: Linux; the bash tool runs bash\n\nTools: calc\n"
    longer = base.replace("runs bash", "runs bash; Python: python3")
    assert normalize(base) == normalize(longer)
