from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Deterministic baseline with short-term, thread-scoped memory only.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent when dependencies exist.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return a response while keeping all state keyed only by ``thread_id``."""

        if not self.force_offline and self.langchain_agent is None:
            self._maybe_build_langchain_agent()

        # The benchmark and tests use force_offline=True. If a live model was
        # explicitly configured but cannot be invoked, falling back to this
        # deterministic path keeps the lab reproducible and inspectable.
        if self.langchain_agent is not None and not self.force_offline:
            try:
                result = self.langchain_agent.invoke(message)
                answer = getattr(result, "content", result)
                if isinstance(answer, list):
                    answer = " ".join(str(item) for item in answer)
                answer = str(answer)
                state = self.sessions.setdefault(thread_id, SessionState())
                prompt_load = self._prompt_load(state, message)
                state.messages.append({"role": "user", "content": message})
                state.messages.append({"role": "assistant", "content": answer})
                state.prompt_tokens_processed += prompt_load
                state.token_usage += estimate_tokens(answer)
                return self._result(thread_id, answer, state)
            except Exception:
                # Live-provider errors are not allowed to corrupt thread state.
                self.langchain_agent = None

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Store the turn and answer from the current thread's prior messages."""

        state = self.sessions.setdefault(thread_id, SessionState())
        prompt_load = self._prompt_load(state, message)
        state.messages.append({"role": "user", "content": message})
        answer = self._offline_response(thread_id, message)
        state.messages.append({"role": "assistant", "content": answer})
        state.prompt_tokens_processed += prompt_load
        state.token_usage += estimate_tokens(answer)
        return self._result(thread_id, answer, state)

    @staticmethod
    def _prompt_load(state: SessionState, message: str) -> int:
        prior = " ".join(item.get("content", "") for item in state.messages)
        return estimate_tokens(" ".join(part for part in (prior, message) if part))

    @staticmethod
    def _result(thread_id: str, answer: str, state: SessionState) -> dict[str, Any]:
        return {
            "thread_id": thread_id,
            "answer": answer,
            "response": answer,
            "text": answer,
            "agent_tokens": estimate_tokens(answer),
            "prompt_tokens": state.prompt_tokens_processed,
            "token_usage": state.token_usage,
        }

    def _thread_facts(self, thread_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for item in self.sessions.get(thread_id, SessionState()).messages:
            if item.get("role") != "user":
                continue
            facts.update(extract_profile_updates(item.get("content", "")))
        return facts

    def _offline_response(self, thread_id: str, message: str) -> str:
        """Answer common recall questions from this thread only."""

        facts = self._thread_facts(thread_id)
        lowered = message.lower()
        requested: list[tuple[str, str]] = []
        if "tên" in lowered or "ai là" in lowered:
            requested.append(("Tên", "name"))
        if "đồ uống" in lowered or "uống" in lowered:
            requested.append(("Đồ uống", "favorite_drink"))
        if "món ăn" in lowered or "món ruột" in lowered:
            requested.append(("Món ăn", "favorite_food"))
        if "nghề" in lowered or "công việc" in lowered:
            requested.append(("Nghề nghiệp", "profession"))
        if "ở đâu" in lowered or "nơi ở" in lowered or "ở huế" in lowered or "ở đà nẵng" in lowered:
            requested.append(("Nơi ở", "location"))
        if "style" in lowered or "kiểu trả lời" in lowered or "cách trả lời" in lowered:
            requested.append(("Style", "response_style"))

        if requested:
            parts = [f"{label}: {facts[key]}" for label, key in requested if facts.get(key)]
            if parts:
                return " | ".join(parts)
            return "Mình chưa có thông tin đó trong thread này."

        if facts and any(marker in lowered for marker in ("nhớ", "ghi nhớ", "mô tả", "tóm tắt")):
            parts = [f"{key}: {value}" for key, value in facts.items()]
            return " | ".join(parts)
        return "Mình đã ghi nhận tạm thời trong thread này."

    def _maybe_build_langchain_agent(self):
        """Best-effort live model construction; offline behavior remains canonical."""

        if self.force_offline or self.langchain_agent is not None:
            return self.langchain_agent
        try:
            self.langchain_agent = build_chat_model(self.config.model)
        except (ImportError, RuntimeError, ValueError):
            self.langchain_agent = None
        return self.langchain_agent
