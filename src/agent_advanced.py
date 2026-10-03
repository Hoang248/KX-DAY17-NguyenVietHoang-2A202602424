from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent with short-term, persistent and compact memory layers.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to an optional live model, with deterministic offline fallback."""

        if not self.force_offline and self.langchain_agent is None:
            self._maybe_build_langchain_agent()
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Run the deterministic memory path used by tests and benchmarks."""

        self._start_turn(user_id, thread_id, message)
        answer = self._offline_response(user_id, thread_id, message)
        return self._finish_turn(user_id, thread_id, answer)

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate profile + compact summary + recent-message prompt load."""

        context = self.compact_memory.context(thread_id)
        recent = context.get("messages", [])
        recent_text = "\n".join(
            f"{item.get('role', 'message')}: {item.get('content', '')}"
            for item in recent
            if isinstance(item, dict)
        )
        profile_text = self.profile_store.read_text(user_id)
        summary_text = str(context.get("summary", ""))
        return estimate_tokens("\n".join(part for part in (profile_text, summary_text, recent_text) if part))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Answer recall questions from the persistent profile, not stale text."""

        facts = self.profile_store.facts(user_id)
        lowered = message.lower()
        requested: list[tuple[str, str]] = []

        def request(label: str, key: str) -> None:
            if (label, key) not in requested:
                requested.append((label, key))

        if "tên" in lowered or "ai là" in lowered:
            request("Tên", "name")
        if "nghề" in lowered or "công việc" in lowered:
            request("Nghề nghiệp", "profession")
        if "đồ uống" in lowered or "uống yêu thích" in lowered:
            request("Đồ uống", "favorite_drink")
        if "món ăn" in lowered or "món ruột" in lowered:
            request("Món ăn", "favorite_food")
        if "nuôi" in lowered or "con gì" in lowered or "corgi" in lowered:
            request("Thú cưng", "pet")
        if (
            "style" in lowered
            or "kiểu trả lời" in lowered
            or "cách trả lời" in lowered
            or "bullet" in lowered
        ):
            request("Style", "response_style")
        if "ở đâu" in lowered or "nơi ở" in lowered or "còn ở" in lowered:
            request("Nơi ở", "location")
        if "mối quan tâm" in lowered or "quan tâm chính" in lowered:
            request("Mối quan tâm", "interests")

        if requested:
            parts = [f"{label}: {facts[key]}" for label, key in requested if facts.get(key)]
            if parts:
                return " | ".join(parts)
            return "Mình chưa có thông tin đó trong bộ nhớ dài hạn."

        if facts and any(marker in lowered for marker in ("nhớ", "ghi nhớ", "mô tả", "tóm tắt")):
            return " | ".join(f"{key}: {value}" for key, value in facts.items())
        return "Mình đã cập nhật bộ nhớ dài hạn và ngữ cảnh của thread này."

    def _maybe_build_langchain_agent(self):
        """Best-effort model construction; deterministic memory remains canonical."""

        if self.force_offline or self.langchain_agent is not None:
            return self.langchain_agent
        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except (ImportError, RuntimeError, ValueError):
            self.langchain_agent = None
        return self.langchain_agent

    @staticmethod
    def _merge_fact(key: str, existing: str | None, new_value: str) -> str:
        """Merge additive preferences while replacing correction-sensitive facts."""

        if not existing or key not in {"response_style", "interests"}:
            return new_value
        parts = [part.strip() for part in re.split(r"[,;]", f"{existing},{new_value}") if part.strip()]
        merged: list[str] = []
        seen: set[str] = set()
        for part in parts:
            marker = part.casefold()
            if marker not in seen:
                seen.add(marker)
                merged.append(part)
        return ", ".join(merged)

    def _persist_profile_updates(self, user_id: str, message: str) -> dict[str, str]:
        updates = extract_profile_updates(message)
        existing = self.profile_store.facts(user_id)
        for key, value in updates.items():
            merged = self._merge_fact(key, existing.get(key), value)
            self.profile_store.upsert_fact(user_id, key, merged)
            existing[key] = merged
        return updates

    def _start_turn(self, user_id: str, thread_id: str, message: str) -> int:
        self._persist_profile_updates(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return prompt_tokens

    def _finish_turn(self, user_id: str, thread_id: str, answer: str) -> dict[str, Any]:
        self.compact_memory.append(thread_id, "assistant", answer)
        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens
        return {
            "thread_id": thread_id,
            "user_id": user_id,
            "answer": answer,
            "response": answer,
            "text": answer,
            "agent_tokens": answer_tokens,
            "token_usage": self.thread_tokens[thread_id],
            "prompt_tokens": self.thread_prompt_tokens.get(thread_id, 0),
            "memory_file_size": self.memory_file_size(user_id),
            "compactions": self.compaction_count(thread_id),
        }

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Use a configured chat model while retaining the same memory contract."""

        self._start_turn(user_id, thread_id, message)
        try:
            result = self.langchain_agent.invoke(message)
            answer = getattr(result, "content", result)
            if isinstance(answer, list):
                answer = " ".join(str(item) for item in answer)
            answer = str(answer)
        except Exception:
            self.langchain_agent = None
            answer = self._offline_response(user_id, thread_id, message)
        return self._finish_turn(user_id, thread_id, answer)
