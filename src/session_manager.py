"""
session_manager.py
------------------
Quản lý trạng thái và phiên làm việc (Session Persistence):
- Lưu trữ lịch sử hội thoại (chat_history) và metadata xuống đĩa.
- Khôi phục phiên làm việc cũ khi người dùng truyền --session <session_id>.
- Liệt kê các phiên làm việc đã lưu kèm thông tin tóm tắt.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)

logger = logging.getLogger(__name__)

STATE_FILENAME = "session_state.json"


class SessionManager:
    """Quản lý việc lưu, nạp và liệt kê các phiên làm việc."""

    @staticmethod
    def save_session_state(
        session_dir: Path,
        chat_history: list[BaseMessage],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Lưu chat_history và metadata vào session_dir/session_state.json."""
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        state_file = session_dir / STATE_FILENAME

        meta = metadata.copy() if metadata else {}
        meta["updated_at"] = datetime.now().isoformat()
        if "created_at" not in meta:
            meta["created_at"] = meta["updated_at"]

        payload = {
            "metadata": meta,
            "chat_history": messages_to_dict(chat_history),
        }

        try:
            state_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Không thể lưu trạng thái phiên vào %s: %s", state_file, exc)

    @staticmethod
    def load_session_state(
        session_dir: Path,
    ) -> tuple[list[BaseMessage], dict[str, Any]]:
        """Nạp chat_history và metadata từ session_dir/session_state.json."""
        session_dir = Path(session_dir)
        state_file = session_dir / STATE_FILENAME
        if not state_file.exists():
            return [], {}

        try:
            content = state_file.read_text(encoding="utf-8")
            data = json.loads(content)
            raw_messages = data.get("chat_history", [])
            messages = messages_from_dict(raw_messages)
            metadata = data.get("metadata", {})
            return messages, metadata
        except Exception as exc:
            logger.error("Lỗi khi đọc file trạng thái %s: %s", state_file, exc)
            return [], {}

    @staticmethod
    def list_sessions(base_dir: Path) -> list[dict[str, Any]]:
        """Duyệt thư mục base_dir để lấy danh sách các phiên làm việc."""
        base_dir = Path(base_dir)
        if not base_dir.exists():
            return []

        results: list[dict[str, Any]] = []
        for item in base_dir.iterdir():
            if not item.is_dir():
                continue
            state_file = item / STATE_FILENAME
            if state_file.exists():
                try:
                    data = json.loads(state_file.read_text(encoding="utf-8"))
                    meta = data.get("metadata", {})
                    results.append(
                        {
                            "session_id": item.name,
                            "path": str(item.resolve()),
                            "created_at": meta.get("created_at", ""),
                            "updated_at": meta.get("updated_at", ""),
                            "last_topic": meta.get("last_topic", "(Chưa có chủ đề)"),
                            "turns": meta.get("turns", 0),
                        }
                    )
                except Exception:
                    results.append(
                        {
                            "session_id": item.name,
                            "path": str(item.resolve()),
                            "created_at": "",
                            "updated_at": "",
                            "last_topic": "(Lỗi đọc session)",
                            "turns": 0,
                        }
                    )
            else:
                # Thư mục có thể chứa transcript.md nhưng chưa có state_file
                transcript_file = item / "transcript.md"
                if transcript_file.exists():
                    results.append(
                        {
                            "session_id": item.name,
                            "path": str(item.resolve()),
                            "created_at": "",
                            "updated_at": "",
                            "last_topic": "(Phiên cũ)",
                            "turns": 0,
                        }
                    )

        # Sắp xếp theo session_id (chứa timestamp chuẩn YYYYMMDD-HHMMSS) giảm dần
        results.sort(key=lambda s: s["session_id"], reverse=True)
        return results

