"""
observability.py
-----------------
Ghi lại toàn bộ bước nội bộ của agent (prompt gửi LLM, phản hồi thô, quyết
định gọi tool, kết quả tool) dưới dạng JSON Lines (.jsonl) — mỗi dòng một
sự kiện, có timestamp và session_id.

Vì sao JSONL chứ không phải .docx hay .log thuần:
- Mỗi dòng là một JSON độc lập -> đọc lại bằng script rất dễ
  (pandas.read_json(path, lines=True), hoặc `jq` trên dòng lệnh).
- Có thể nạp thẳng vào công cụ quan sát khác nếu sau này cần.
- Không phá hỏng cấu trúc nếu chương trình crash giữa chừng (khác với một
  file JSON lớn duy nhất).

Đây là lựa chọn phổ biến trong thực tế để log agent, khác với transcript.py
(dành cho báo cáo con người đọc).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class JsonlTraceLogger(BaseCallbackHandler):
    """Callback handler ghi mỗi sự kiện của agent thành một dòng JSON."""

    def __init__(self, path: Path, session_id: str):
        super().__init__()
        self.path = Path(path)
        self.session_id = session_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _emit(self, event: str, data: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "event": event,
            **data,
        }
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            # Ghi log không được phép làm agent sập — chỉ cảnh báo.
            logger.warning("Không ghi được trace vào %s: %s", self.path, exc)

    # ------------------------------------------------------------------ #
    # LLM
    # ------------------------------------------------------------------ #
    def on_llm_start(
        self, serialized: dict, prompts: list[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._emit("llm_start", {"prompt": "\n\n".join(prompts)})

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        rendered = [
            {"role": getattr(m, "type", m.__class__.__name__), "content": getattr(m, "content", str(m))}
            for msg_list in messages
            for m in msg_list
        ]
        self._emit("chat_model_start", {"messages": rendered})

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            text = response.generations[0][0].text
        except Exception:
            text = str(response)
        self._emit("llm_end", {"output": text})

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._emit("llm_error", {"error": str(error)})

    # ------------------------------------------------------------------ #
    # Agent
    # ------------------------------------------------------------------ #
    def on_agent_action(self, action: AgentAction, *, run_id: UUID, **kwargs: Any) -> None:
        self._emit(
            "agent_action",
            {"tool": action.tool, "tool_input": action.tool_input, "log": action.log},
        )

    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, **kwargs: Any) -> None:
        self._emit("agent_finish", {"return_values": finish.return_values})

    # ------------------------------------------------------------------ #
    # Tool
    # ------------------------------------------------------------------ #
    def on_tool_start(
        self, serialized: dict, input_str: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        name = serialized.get("name", "unknown_tool")
        self._emit("tool_start", {"tool": name, "input": input_str})

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._emit("tool_end", {"output": str(output)})

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._emit("tool_error", {"error": str(error)})
