"""
node.py
-------
Script độc lập trực quan hóa và kiểm tra sơ đồ đồ thị (Workflow Graph) của Research Agent.

Cách dùng:
    python node.py              # In sơ đồ ASCII và cú pháp Mermaid ra terminal
    python node.py --save       # Lưu sơ đồ Mermaid vào file workflow.mermaid.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Đảm bảo in tiếng Việt mượt mà trên console Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent import build_research_graph, get_graph_ascii, get_graph_mermaid
from config import Settings


def render_workflow(settings: Settings | None = None) -> tuple[str, str]:
    """Khởi tạo đồ thị với cấu hình và trả về (ascii_art, mermaid_art)."""
    if settings is None:
        settings = Settings(
            nvidia_api_key="mock-key-for-visualization",
            model_name="deepseek-ai/deepseek-v4.1-flash",
            model_base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.2,
            max_tokens=2048,
            max_iterations=5,
            max_chat_history_messages=10,
            output_dir=Path("runs"),
            log_level="INFO",
            langsmith_tracing=False,
        )

    graph, _ = build_research_graph(settings)
    ascii_art = get_graph_ascii(graph)
    mermaid_art = get_graph_mermaid(graph)
    return ascii_art, mermaid_art


def main() -> None:
    parser = argparse.ArgumentParser(description="Trực quan hóa LangGraph Workflow")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Lưu sơ đồ Mermaid ra file workflow.mermaid.md",
    )
    args = parser.parse_args()

    ascii_art, mermaid_art = render_workflow()

    print("=" * 60)
    print("  LANGGRAPH MULTI-AGENT WORKFLOW (RESEARCH AGENT)")
    print("=" * 60)
    print("\n[1] SƠ ĐỒ ASCII:")
    print(ascii_art)
    print("\n" + "-" * 60)
    print("[2] CÚ PHÁP MERMAID (Xem trong VS Code hoặc https://mermaid.live):")
    print(f"```mermaid\n{mermaid_art}\n```")
    print("=" * 60)

    if args.save:
        out_file = Path("workflow.mermaid.md")
        content = f"# Research Agent Workflow Diagram\n\n```mermaid\n{mermaid_art}\n```\n"
        out_file.write_text(content, encoding="utf-8")
        print(f"✔ Đã lưu sơ đồ ra file: {out_file.resolve()}")


if __name__ == "__main__":
    main()
