from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Return a stable, provider-independent token estimate."""

    if not text or not text.strip():
        return 0
    return max(1, ceil(len(text.strip()) / 4))


_FACT_LINE = re.compile(r"(?m)^- (?P<key>[a-z][a-z0-9_-]*): (?P<value>.*)$")
_DEFAULT_PROFILE = "# User Profile\n\n"


@dataclass
class UserProfileStore:
    """Small markdown-backed store for one persistent profile per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Return ``<root>/<safe-user-id>/User.md`` without path traversal."""

        raw = str(user_id).strip()
        safe = re.sub(r"[^\w.-]+", "_", raw, flags=re.UNICODE).strip("._")
        if not safe:
            safe = "anonymous"
        if safe.upper() in {"CON", "PRN", "AUX", "NUL", "COM1", "LPT1"}:
            safe = f"user_{safe}"
        return Path(self.root_dir).resolve() / safe / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        updated = content.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.is_file() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Read structured ``- key: value`` facts from the markdown profile."""

        return {match.group("key"): match.group("value").strip() for match in _FACT_LINE.finditer(self.read_text(user_id))}

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Insert or replace one structured fact while preserving other markdown."""

        normalized_key = re.sub(r"[^a-z0-9_-]+", "_", key.lower()).strip("_")
        if not normalized_key:
            raise ValueError("Fact key must contain at least one alphanumeric character.")
        normalized_value = re.sub(r"\s+", " ", str(value)).strip()
        if not normalized_value:
            raise ValueError("Fact value must not be empty.")

        content = self.read_text(user_id) or _DEFAULT_PROFILE
        line = f"- {normalized_key}: {normalized_value}"
        pattern = re.compile(rf"(?m)^- {re.escape(normalized_key)}:.*$")
        if pattern.search(content):
            content = pattern.sub(line, content, count=1)
        else:
            content = content.rstrip() + "\n" + line + "\n"
        return self.write_text(user_id, content)


def _clean_fact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n,.;:!?()[]")


def _sentences(message: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", message) if part.strip()]


def _title_case_location_pattern() -> str:
    # Requiring an uppercase first character avoids treating phrases such as
    # "gần sông Hương" or "quán cà phê" as the user's current location.
    return r"([A-ZÀ-ÖØ-ÞĐ][\wÀ-ỹ-]*(?:\s+[A-ZÀ-ÖØ-ÞĐ][\wÀ-ỹ-]*){0,2})"


def _extract_location(message: str) -> str | None:
    pattern = _title_case_location_pattern()
    candidates: list[str] = []
    location_patterns = (
        rf"\b(?i:(?:hiện(?:\s+tại)?|bây giờ)\s+(?:mình\s+)?(?:đang\s+)?ở)\s+{pattern}",
        rf"\b(?i:(?:mình|tôi)\s+(?:vẫn\s+)?(?:đang\s+)?(?:ở|sống\s+ở|làm việc ở))\s+{pattern}",
        rf"\b(?i:nơi ở hiện tại(?: của mình)?\s*(?:là|:))\s*{pattern}",
    )
    for sentence in _sentences(message):
        lowered = sentence.lower()
        if any(marker in lowered for marker in ("không phải nơi ở", "chỉ là nơi", "vừa bay ra", "đi họp")):
            continue
        for location_pattern in location_patterns:
            candidates.extend(match.group(1) for match in re.finditer(location_pattern, sentence))
    return _clean_fact(candidates[-1]) if candidates else None


def _extract_profession(message: str) -> str | None:
    role = r"((?:[A-Za-zÀ-ỹ0-9][\w.+#/-]*\s+){0,2}(?:engineer|manager|developer|scientist|designer|analyst|architect|kỹ sư|quản lý))"
    patterns = (
        rf"\b(?:nghề nghiệp(?: hiện tại)?(?: của mình)?\s*(?:là|vẫn là|:)|(?:mình|tôi)\s+(?:hiện\s+)?(?:đang\s+)?làm|(?:đang|hiện đang)\s+làm|(?:mình|tôi)\s+đã\s+chuyển\s+sang|(?:giờ|hiện giờ)\s+chuyển\s+sang|chuyển\s+sang)\s+{role}\b",
    )
    candidates: list[str] = []
    for sentence in _sentences(message):
        lowered = sentence.lower()
        if "đùa" in lowered or "hay là" in lowered:
            continue
        for profession_pattern in patterns:
            candidates.extend(match.group(1) for match in re.finditer(profession_pattern, sentence, flags=re.IGNORECASE))
    return _clean_fact(candidates[-1]) if candidates else None


def _extract_style(message: str) -> str | None:
    lowered = message.lower()
    if message.strip().endswith("?") and any(marker in lowered for marker in ("nhắc lại", "thế nào", "gì", "không")):
        return None
    style_markers = ("trả lời", "câu trả lời", "style", "giải thích", "ngắn gọn", "bullet", "lan man")
    if not any(marker in lowered for marker in style_markers):
        return None

    parts: list[str] = []
    if "3 bullet" in lowered:
        parts.append("3 bullet ngắn")
    elif "bullet" in lowered:
        parts.append("có bullet")
    if "ngắn gọn" in lowered or "trả lời ngắn" in lowered or "không thích" in lowered and "lan man" in lowered:
        parts.append("ngắn gọn")
    if "rõ ý" in lowered:
        parts.append("rõ ý")
    if "có cấu trúc" in lowered:
        parts.append("có cấu trúc")
    if "ví dụ thực chiến" in lowered:
        parts.append("có ví dụ thực chiến")
    elif "ví dụ thực tế" in lowered or "ví dụ" in lowered:
        parts.append("có ví dụ thực tế")
    if "trade-off" in lowered or "đánh đổi" in lowered:
        parts.append("nhấn trade-off")

    unique_parts = list(dict.fromkeys(parts))
    return ", ".join(unique_parts) if unique_parts else None


def _extract_drink(message: str) -> str | None:
    patterns = (
        r"\b(?:đồ uống|thức uống) yêu thích(?: của mình)?\s*(?:là|:)\s*([^,.!?;\n]+)",
        r"\b(?:mình\s+)?(?:vẫn\s+)?uống\s+(?!yêu thích\b|của mình\b)([^,.!?;\n]+)",
    )
    candidates: list[str] = []
    for sentence in _sentences(message):
        lowered = sentence.lower()
        if sentence.rstrip().endswith("?"):
            continue
        if any(marker in lowered for marker in ("nhớ lại", "nhắc lại", "hỏi lại", "sẽ còn hỏi")):
            continue
        if "uống" not in lowered and "đồ uống" not in lowered and "thức uống" not in lowered:
            continue
        for pattern in patterns:
            for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                value = re.split(r"\b(?:nhưng|như cũ|để)\b", match.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
                value = _clean_fact(value)
                if value and value.lower() not in {"rồi đọc issue mới", "rồi", "gì", "đâu", "nào"} and not value.lower().startswith(("và ", "yêu thích", "của mình")):
                    candidates.append(value)
    return candidates[-1] if candidates else None


def _extract_food(message: str) -> str | None:
    patterns = (
        r"\bmón ăn yêu thích(?: của mình)?\s*(?:là|:)\s*([^,.!?;\n]+)",
        r"\b(?:mình\s+)?(?:ăn|ăn món)\s+(?!yêu thích\b)([^,.!?;\n]+)",
    )
    candidates: list[str] = []
    for sentence in _sentences(message):
        if sentence.rstrip().endswith("?"):
            continue
        for pattern in patterns:
            for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                value = re.split(r"\b(?:và|rồi|thấy|nhưng)\b", match.group(1), maxsplit=1, flags=re.IGNORECASE)[0]
                value = _clean_fact(value)
                if value and value.lower() not in {"sáng", "trưa", "tối"}:
                    candidates.append(value)
    return candidates[-1] if candidates else None


def _extract_pet(message: str) -> str | None:
    pattern = r"\bnuôi\s+(?:một\s+)?(?:bé\s+)?([^,.!?;\n]+)"
    candidates = []
    for sentence in _sentences(message):
        if sentence.rstrip().endswith("?"):
            continue
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            value = _clean_fact(match.group(1))
            if value:
                candidates.append(value)
    return candidates[-1] if candidates else None


def _extract_interests(message: str) -> str | None:
    lowered = message.lower()
    if not any(marker in lowered for marker in ("thích", "quan tâm", "mối quan tâm")):
        return None
    known_terms = (
        ("Python", "python"),
        ("AI ứng dụng", "ai ứng dụng"),
        ("AI agent", "ai agent"),
        ("MLOps", "mlops"),
        ("RAG", "rag"),
        ("evaluation", "evaluation"),
        ("benchmark memory", "benchmark memory"),
        ("async Python", "async python"),
    )
    found = [display for display, needle in known_terms if needle in lowered]
    return ", ".join(dict.fromkeys(found)) if found else None


def _looks_like_recall_question(message: str) -> bool:
    lowered = message.lower().strip()
    if lowered.endswith("?"):
        return True
    return any(
        marker in lowered
        for marker in (
            "nhắc lại",
            "nhớ lại",
            "thử nhớ",
            "bạn biết",
            "hãy nhắc",
            "tên mình là gì",
            "mình tên gì",
            "đồ uống yêu thích của mình là gì",
            "món ăn yêu thích của mình là gì",
            "hiện tại mình đang ở đâu",
            "hiện tại mình làm nghề gì",
        )
    )


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract only high-confidence, stable profile facts from one user turn."""

    if not message or not message.strip() or _looks_like_recall_question(message):
        return {}

    updates: dict[str, str] = {}
    name_match = re.search(
        r"\b(?:mình\s+)?tên(?:\s+mình)?\s*(?:là|=)\s+([A-Za-zÀ-ỹ0-9][^,.;!?\n]*)",
        message,
        flags=re.IGNORECASE,
    )
    if name_match:
        updates["name"] = _clean_fact(name_match.group(1))

    for key, extractor in (
        ("location", _extract_location),
        ("profession", _extract_profession),
        ("response_style", _extract_style),
        ("favorite_drink", _extract_drink),
        ("favorite_food", _extract_food),
        ("pet", _extract_pet),
        ("interests", _extract_interests),
    ):
        value = extractor(message)
        if value:
            updates[key] = value
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a bounded deterministic summary from older chat messages."""

    if not messages or max_items <= 0:
        return ""

    def render(message: dict[str, str]) -> str:
        role = str(message.get("role", "message")).strip() or "message"
        content = re.sub(r"\s+", " ", str(message.get("content", "")).strip())
        if len(content) > 240:
            content = content[:237].rstrip() + "..."
        return f"{role}: {content}"

    if len(messages) > max_items:
        head_count = max(1, max_items // 2)
        tail_count = max_items - head_count
        selected = list(messages[:head_count])
        selected.append({"role": "summary", "content": f"{len(messages) - max_items} earlier messages omitted"})
        selected.extend(messages[-tail_count:] if tail_count else [])
    else:
        selected = list(messages)
    return "\n".join(render(message) for message in selected)


@dataclass
class CompactMemoryManager:
    """Keep recent messages in full and compact older context at a threshold."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _new_state(self) -> dict[str, object]:
        return {"messages": [], "summary": "", "compactions": 0}

    @staticmethod
    def _token_count(thread_state: dict[str, object]) -> int:
        messages = thread_state.get("messages", [])
        summary = str(thread_state.get("summary", ""))
        message_text = " ".join(str(item.get("content", "")) for item in messages if isinstance(item, dict))
        return estimate_tokens(summary) + estimate_tokens(message_text)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread_state = self.state.setdefault(thread_id, self._new_state())
        messages = thread_state.setdefault("messages", [])
        if not isinstance(messages, list):
            messages = []
            thread_state["messages"] = messages
        messages.append({"role": str(role), "content": str(content)})

        threshold = max(1, int(self.threshold_tokens))
        keep = max(1, int(self.keep_messages))
        while self._token_count(thread_state) > threshold and len(messages) > keep:
            cutoff = max(1, len(messages) - keep)
            older = messages[:cutoff]
            prior_summary = str(thread_state.get("summary", ""))
            summary_inputs: list[dict[str, str]] = []
            if prior_summary:
                summary_inputs.append({"role": "summary", "content": prior_summary})
            summary_inputs.extend(older)
            thread_state["summary"] = summarize_messages(summary_inputs, max_items=6)
            thread_state["messages"] = messages[cutoff:]
            messages = thread_state["messages"]
            thread_state["compactions"] = int(thread_state.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread_state = self.state.get(thread_id, self._new_state())
        messages = thread_state.get("messages", [])
        return {
            "messages": [dict(item) for item in messages if isinstance(item, dict)],
            "summary": str(thread_state.get("summary", "")),
            "compactions": int(thread_state.get("compactions", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
