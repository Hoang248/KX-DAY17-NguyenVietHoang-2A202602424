from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates


def make_config(tmp_path: Path):
    """Build an isolated, low-threshold offline configuration."""

    config = load_config(tmp_path)
    config.compact_threshold_tokens = 40
    config.compact_keep_messages = 2
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """User.md supports safe creation, reading, replacement and size checks."""

    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("alice") == ""
    path = store.write_text("alice", "# User Profile\n\n- name: Alice\n")
    assert path.name == "User.md"
    assert store.read_text("alice").endswith("- name: Alice\n")
    assert store.file_size("alice") > 0
    assert store.edit_text("alice", "Alice", "Alicia")
    assert "Alicia" in store.read_text("alice")
    assert not store.edit_text("alice", "missing", "replacement")
    assert store.path_for("alice/../../escape").parent.parent == (tmp_path / "profiles").resolve()
    updates = extract_profile_updates("Mình ở Đà Nẵng và đang làm backend engineer.")
    assert updates["location"] == "Đà Nẵng"


def test_compact_trigger(tmp_path: Path) -> None:
    """Long input moves older messages into a summary and keeps recent turns."""

    manager = CompactMemoryManager(threshold_tokens=20, keep_messages=2)
    for index in range(5):
        manager.append("thread-1", "user", f"message-{index}-" + "x" * 60)
    context = manager.context("thread-1")
    assert manager.compaction_count("thread-1") > 0
    assert len(context["messages"]) <= 2
    assert context["summary"]


def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced persists a fact across threads; baseline remains thread-scoped."""

    baseline = BaselineAgent(config=make_config(tmp_path / "baseline"), force_offline=True)
    baseline.reply("dungct", "thread-a", "Mình tên là DũngCT.")
    baseline_answer = baseline.reply("dungct", "thread-b", "Mình tên gì?")["answer"]
    assert "DũngCT" not in baseline_answer

    advanced = AdvancedAgent(config=make_config(tmp_path / "advanced"), force_offline=True)
    advanced.reply("dungct", "thread-a", "Mình tên là DũngCT.")
    advanced_answer = advanced.reply("dungct", "thread-b", "Mình tên gì?")["answer"]
    assert "DũngCT" in advanced_answer
    assert (tmp_path / "advanced" / "state" / "profiles" / "dungct" / "User.md").is_file()


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compaction bounds Advanced prompt load relative to the baseline history."""

    baseline = BaselineAgent(config=make_config(tmp_path / "baseline"), force_offline=True)
    advanced = AdvancedAgent(config=make_config(tmp_path / "advanced"), force_offline=True)
    long_message = "Mình đang ghi một đoạn context dài để kiểm tra token load. " + "x" * 180
    for index in range(10):
        baseline.reply("stress", "long", f"{long_message} {index}")
        advanced.reply("stress", "long", f"{long_message} {index}")

    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
