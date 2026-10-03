from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read and lightly validate a benchmark conversation list."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("conversations", payload.get("items", []))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of conversations in {path}.")
    conversations: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Conversation {index} in {path} is not an object.")
        if not isinstance(item.get("turns"), list):
            raise ValueError(f"Conversation {index} in {path} has no turns list.")
        conversations.append(item)
    return conversations


def recall_points(answer: str, expected: list[str]) -> float:
    """Score no/partial/full expected-fact coverage as 0, 0.5 or 1."""

    if not expected:
        return 0.0
    normalized_answer = str(answer).casefold()
    matched = sum(1 for item in expected if str(item).casefold() in normalized_answer)
    if matched == 0:
        return 0.0
    if matched == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Return a transparent offline quality proxy, not an expert judgment."""

    if not str(answer).strip():
        return 0.0
    return recall_points(answer, expected)


def _read_metric(agent: Any, method_name: str, thread_ids: set[str], fallback: int) -> int:
    method = getattr(agent, method_name, None)
    if not callable(method):
        return fallback
    return sum(max(0, int(method(thread_id))) for thread_id in thread_ids)


def _answer_from_result(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("answer", result.get("response", result.get("text", ""))))
    return str(result)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent using identical history and fresh recall threads."""

    history_thread_ids: set[str] = set()
    all_thread_ids: set[str] = set()
    fallback_agent_tokens = 0
    fallback_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    memory_start: dict[str, int] = {}

    memory_size_method = getattr(agent, "memory_file_size", None)
    for index, conversation in enumerate(conversations):
        conversation_id = str(conversation.get("id", f"conversation-{index}"))
        user_id = str(conversation.get("user_id", f"user-{index}"))
        if callable(memory_size_method) and user_id not in memory_start:
            memory_start[user_id] = max(0, int(memory_size_method(user_id)))

        history_thread = f"history:{agent_name}:{conversation_id}"
        history_thread_ids.add(history_thread)
        all_thread_ids.add(history_thread)
        for turn in conversation.get("turns", []):
            result = agent.reply(user_id, history_thread, str(turn))
            if isinstance(result, dict):
                fallback_agent_tokens += int(result.get("agent_tokens", 0) or 0)
                fallback_prompt_tokens += int(result.get("prompt_tokens_delta", 0) or 0)

        for question_index, question_data in enumerate(conversation.get("recall_questions", [])):
            if not isinstance(question_data, dict):
                continue
            question = str(question_data.get("question", ""))
            expected = [str(item) for item in question_data.get("expected_contains", [])]
            recall_thread = f"recall:{agent_name}:{conversation_id}:{question_index}"
            all_thread_ids.add(recall_thread)
            result = agent.reply(user_id, recall_thread, question)
            answer = _answer_from_result(result)
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))
            if isinstance(result, dict):
                fallback_agent_tokens += int(result.get("agent_tokens", 0) or 0)
                fallback_prompt_tokens += int(result.get("prompt_tokens_delta", 0) or 0)

    growth = 0
    if callable(memory_size_method):
        for user_id, initial_size in memory_start.items():
            final_size = max(0, int(memory_size_method(user_id)))
            growth += max(0, final_size - initial_size)

    agent_tokens = _read_metric(agent, "token_usage", all_thread_ids, fallback_agent_tokens)
    prompt_tokens = _read_metric(agent, "prompt_token_usage", all_thread_ids, fallback_prompt_tokens)
    compactions = _read_metric(agent, "compaction_count", history_thread_ids, 0)
    average_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=average_recall,
        response_quality=average_quality,
        memory_growth_bytes=growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows without requiring the optional ``tabulate`` package."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.agent_name,
                    str(row.agent_tokens_only),
                    str(row.prompt_tokens_processed),
                    f"{row.recall_score:.2f}",
                    f"{row.response_quality:.2f}",
                    str(row.memory_growth_bytes),
                    str(row.compactions),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    """Run standard and long-context offline benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    root = Path(__file__).resolve().parent.parent
    base_config = load_config(root)
    suites = (
        ("Standard Benchmark", root / "data" / "conversations.json", "standard"),
        ("Long-Context Stress Benchmark", root / "data" / "advanced_long_context.json", "stress"),
    )
    with tempfile.TemporaryDirectory(prefix="day17-benchmark-") as run_state:
        for title, dataset_path, state_name in suites:
            suite_config = replace(base_config, state_dir=Path(run_state) / state_name)
            suite_config.state_dir.mkdir(parents=True, exist_ok=True)
            conversations = load_conversations(dataset_path)
            agents = (
                ("Baseline", BaselineAgent(config=suite_config, force_offline=True)),
                ("Advanced", AdvancedAgent(config=suite_config, force_offline=True)),
            )
            rows = [run_agent_benchmark(name, agent, conversations, suite_config) for name, agent in agents]
            print(f"## {title}")
            print(format_rows(rows))
            print()


if __name__ == "__main__":
    main()
