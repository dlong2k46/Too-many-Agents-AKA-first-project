"""
transcript.py
-------------
Ghi lại lịch sử hội thoại (câu hỏi + kết quả) để CON NGƯỜI đọc lại.

Định dạng chính là Markdown (.md):
- Đọc được trực tiếp trên GitHub/GitLab, mở bằng bất kỳ editor nào.
- Diff được qua Git.
- Nhẹ, nhanh, ghi chi tiết nguồn và kết quả tra cứu.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from schemas import ResearchResponse


@dataclass
class ToolStep:
    tool: str
    input: Any
    output: str


@dataclass
class Turn:
    question: str
    raw_output: str
    structured: Optional[ResearchResponse]
    tool_calls: list[str]
    timestamp: datetime = field(default_factory=datetime.now)
    tool_steps: list[ToolStep] = field(default_factory=list)


class MarkdownTranscriptWriter:
    """Ghi từng lượt hội thoại ra một file Markdown, cập nhật ngay sau mỗi lượt."""

    def __init__(self, path: Path, title: str = "Lịch sử nghiên cứu của Agent"):
        self.path = Path(path)
        self.title = title
        self.turns: list[Turn] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.turn_offset: int = self._detect_turn_offset()

    def _detect_turn_offset(self) -> int:
        """Phát hiện số lượt cao nhất đã có trong file transcript.md nếu file đã tồn tại."""
        if not self.path.exists():
            return 0
        try:
            content = self.path.read_text(encoding="utf-8")
            import re

            matches = re.findall(r"^## Lượt (\d+)", content, flags=re.MULTILINE)
            if matches:
                return max(int(m) for m in matches)
        except Exception:
            pass
        return 0

    def add_turn(
        self,
        question: str,
        raw_output: str,
        structured: Optional[ResearchResponse],
        tool_calls: list[str],
        tool_steps: Optional[list[ToolStep]] = None,
    ) -> None:
        turn = Turn(
            question=question,
            raw_output=raw_output,
            structured=structured,
            tool_calls=tool_calls,
            tool_steps=tool_steps or [],
        )
        self.turns.append(turn)
        current_index = self.turn_offset + len(self.turns)

        # Nếu file chưa tồn tại hoặc rỗng, ghi tiêu đề trước
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.path.write_text(f"# {self.title}\n\n", encoding="utf-8")

        rendered = self._render_turn(turn, current_index)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(rendered + "\n")

    def _render_turn(self, turn: Turn, index: int) -> str:
        lines: list[str] = [
            f"## Lượt {index}",
            f"_{turn.timestamp:%d/%m/%Y %H:%M:%S}_",
            "",
            "**Câu hỏi:**",
            turn.question,
            "",
        ]

        if turn.structured is not None:
            s = turn.structured
            lines.append(f"**Chủ đề:** {s.topic}")
            lines.append("")
            lines.append("**Tóm tắt:**")
            lines.append(s.summary)
            lines.append("")
            if s.sources:
                lines.append("**Nguồn tham khảo:**")
                lines.extend(f"- {src}" for src in s.sources)
                lines.append("")
            if turn.tool_steps:
                lines.append("**Kết quả tra cứu từ các nguồn:**")
                for step in turn.tool_steps:
                    lines.append(f"- **Nguồn / Tool:** `{step.tool}`")
                    lines.append(f"  - **Truy vấn:** `{step.input}`")
                    lines.append("  - **Kết quả trả về:**")
                    lines.append("    ```")
                    for out_line in step.output.strip().splitlines():
                        lines.append(f"    {out_line}")
                    lines.append("    ```")
                lines.append("")
            elif turn.tool_calls:
                lines.append("**Tool đã gọi (theo executor):**")
                lines.extend(f"- {t}" for t in turn.tool_calls)
                lines.append("")
        else:
            lines.append("> ⚠ Không phân tích được JSON. Nội dung thô:")
            lines.append("")
            lines.append("```")
            lines.append(turn.raw_output)
            lines.append("```")
            lines.append("")
            if turn.tool_steps:
                lines.append("**Kết quả tra cứu từ các nguồn:**")
                for step in turn.tool_steps:
                    lines.append(f"- **Nguồn / Tool:** `{step.tool}`")
                    lines.append(f"  - **Truy vấn:** `{step.input}`")
                    lines.append("  - **Kết quả trả về:**")
                    lines.append("    ```")
                    for out_line in step.output.strip().splitlines():
                        lines.append(f"    {out_line}")
                    lines.append("    ```")
                lines.append("")

        lines.append("---")
        return "\n".join(lines)


