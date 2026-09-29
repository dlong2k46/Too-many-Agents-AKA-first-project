"""
ui.py
-----
Giao diện dòng lệnh phong phú (Rich Terminal UI):
- Spinner hiển thị trạng thái động trong thời gian thực khi agent đang tìm kiếm hoặc suy nghĩ.
- Trình bày kết quả nghiên cứu dưới dạng Markdown và bảng tổng hợp đẹp mắt.
- Hỗ trợ hiệu ứng streaming hiển thị tóm tắt.
- Bảng hiển thị danh sách các phiên làm việc (Session List).
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from schemas import ResearchResponse

console = Console()


class SimpleNodeSpinner:
    """Spinner tượng trưng đơn giản cho từng Node đang xử lý."""

    def __init__(self, console_instance: Console | None = None) -> None:
        self.console = console_instance or console
        self._status = None

    def start(self, message: str = "Đang nghiên cứu và thu thập dữ liệu...") -> None:
        if self._status is None:
            self._status = self.console.status(
                f"[bold cyan]{message}[/bold cyan]",
                spinner="dots",
            )
            self._status.start()
        else:
            self._status.update(f"[bold cyan]{message}[/bold cyan]")

    def update(self, message: str) -> None:
        if self._status is not None:
            self._status.update(f"[bold cyan]{message}[/bold cyan]")
        else:
            self.start(message)

    def stop(self) -> None:
        if self._status is not None:
            try:
                self._status.stop()
            except Exception:
                pass
            self._status = None



def render_research_result(
    structured: ResearchResponse,
    real_tool_calls: list[str],
    stream: bool = True,
) -> None:
    """In kết quả nghiên cứu đẹp mắt bằng Rich."""
    console.print()

    # Panel tiêu đề chủ đề
    title_text = Text(f"📌 CHỦ ĐỀ: {structured.topic}", style="bold cyan")
    console.print(Panel(title_text, border_style="cyan"))

    console.print("[bold green]📝 TÓM TẮT BÁO CÁO:[/bold green]")
    if stream:
        # Hiệu ứng gõ máy tính mượt mà cho phần tóm tắt
        summary_lines = structured.summary.split("\n")
        for line in summary_lines:
            for word in line.split(" "):
                console.print(word + " ", end="", highlight=False)
                time.sleep(0.015)
            console.print()
        console.print()
    else:
        console.print(Markdown(structured.summary))
        console.print()

    # Bảng nguồn tham khảo và công cụ
    table = Table(
        title="🔍 Nguồn dữ liệu & Công cụ sử dụng",
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("Mục", style="cyan", width=22)
    table.add_column("Chi tiết", style="white")

    if structured.sources:
        sources_str = "\n".join(f"• {src}" for src in structured.sources)
    else:
        sources_str = "(Không có nguồn ngoài hoặc dùng kiến thức nền)"
    table.add_row("Nguồn tham khảo", sources_str)

    tools_str = ", ".join(real_tool_calls) if real_tool_calls else "(Không gọi tool)"
    table.add_row("Tool đã dùng", tools_str)

    console.print(table)
    console.print()


def render_session_list(sessions: list[dict[str, Any]]) -> None:
    """Hiển thị danh sách các phiên nghiên cứu bằng Rich Table."""
    if not sessions:
        console.print("[yellow]Chưa có phiên nghiên cứu nào được lưu.[/yellow]")
        return

    table = Table(
        title="📚 Danh sách các phiên nghiên cứu (Sessions)",
        show_header=True,
        header_style="bold green",
        border_style="blue",
    )
    table.add_column("Session ID", style="cyan bold", width=26)
    table.add_column("Lần cuối cập nhật", style="dim", width=20)
    table.add_column("Số lượt", justify="center", width=8)
    table.add_column("Chủ đề gần nhất", style="white")

    for s in sessions:
        updated = s.get("updated_at") or s.get("created_at") or "-"
        if len(updated) > 19:
            updated = updated[:19].replace("T", " ")
        table.add_row(
            s["session_id"],
            updated,
            str(s.get("turns", 0)),
            s.get("last_topic", "(Chưa có chủ đề)"),
        )

    console.print(table)
    console.print(
        "\n[dim]Gợi ý: Dùng lệnh [bold cyan]python main.py --session <SESSION_ID>[/bold cyan] để tiếp tục phiên làm việc cũ.[/dim]\n"
    )


def render_graph_view(ascii_art: str, mermaid_art: str | None = None) -> None:
    """Hiển thị sơ đồ kiến trúc các Node trong LangGraph."""
    console.print()
    panel = Panel(
        ascii_art,
        title="🧩 Sơ đồ kiến trúc Đồ thị LangGraph (Multi-Agent)",
        border_style="cyan",
        subtitle="[dim]Luồng: START ➔ researcher ➔ tools ➔ critic_writer ➔ END[/dim]",
    )
    console.print(panel)

    if mermaid_art:
        console.print("[dim]Cú pháp Mermaid (sao chép để vẽ trong Mermaid Live Editor hoặc VS Code):[/dim]")
        console.print(f"```mermaid\n{mermaid_art}\n```")
    console.print()

