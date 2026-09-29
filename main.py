"""
main.py
-------
Điểm vào chương trình Research Agent (Multi-Agent kiến trúc LangGraph).

Chạy chế độ hỏi liên tục:
    python main.py

Chạy một câu hỏi duy nhất rồi thoát:
    python main.py --query "AI là gì?"

Xem danh sách các phiên nghiên cứu cũ:
    python main.py --list-sessions

Tiếp tục phiên làm việc cũ:
    python main.py --session <session_id>

Xem sơ đồ kiến trúc các Node trong LangGraph:
    python main.py --show-graph
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from agent import (
    build_research_graph,
    extract_sources_from_chat_history,
    extract_sources_from_tool_steps,
    get_graph_ascii,
    get_graph_mermaid,
)
from config import ConfigError, Settings, load_settings
from observability import JsonlTraceLogger
from output_parsing import parse_agent_output
from schemas import ResearchResponse
from session_manager import SessionManager
from transcript import MarkdownTranscriptWriter, ToolStep
from ui import (
    SimpleNodeSpinner,
    console,
    render_graph_view,
    render_research_result,
    render_session_list,
)

logger = logging.getLogger(__name__)


def setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def new_run_dir(base: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{stamp}-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_run_dir(base: Path, session_arg: str | None) -> tuple[Path, bool]:
    """Trả về (run_dir, is_resumed)."""
    if not session_arg:
        return new_run_dir(base), False

    candidate = Path(session_arg)
    if candidate.is_dir():
        return candidate, True

    candidate = base / session_arg
    if candidate.is_dir():
        return candidate, True

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate, False


def run_query(
    query: str,
    graph: Any,
    parser: PydanticOutputParser,
    chat_history: list,
    transcript: MarkdownTranscriptWriter,
    run_dir: Path,
    max_history: int,
    stream: bool = True,
    spinner: SimpleNodeSpinner | None = None,
    trace_logger: JsonlTraceLogger | None = None,
) -> None:
    """Chạy một câu hỏi qua đồ thị LangGraph với spinner tượng trưng theo Node, lưu chi tiết vào transcript."""
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "chat_history": list(chat_history),
        "tool_steps": [],
        "iteration": 0,
    }

    accumulated_tool_steps: list[ToolStep] = []
    final_output = ""
    structured = None

    active_spinner = spinner or SimpleNodeSpinner(console)
    active_spinner.start("Đang nghiên cứu và thu thập dữ liệu...")

    config = {}
    if trace_logger:
        config["callbacks"] = [trace_logger]

    try:
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, updates in event.items():
                if node_name in {"researcher", "tools"}:
                    active_spinner.update("Đang nghiên cứu và thu thập dữ liệu...")
                    steps = updates.get("tool_steps", [])
                    if steps:
                        accumulated_tool_steps = steps
                elif node_name == "critic_writer":
                    active_spinner.update("Đang phản biện và tổng hợp báo cáo...")
                    final_output = updates.get("final_output", "")
                    structured = updates.get("structured_response")
    finally:
        active_spinner.stop()


    real_tool_calls = sorted({s.tool for s in accumulated_tool_steps})
    real_sources = extract_sources_from_tool_steps(accumulated_tool_steps)
    if not real_sources and chat_history:
        real_sources = extract_sources_from_chat_history(chat_history)

    if structured is not None:
        structured.tools_used = real_tool_calls or structured.tools_used
        if not structured.sources and real_sources:
            structured.sources = real_sources
        render_research_result(structured, real_tool_calls, stream=stream)
    elif final_output:
        try:
            structured = parse_agent_output(final_output, parser)
            structured.tools_used = real_tool_calls or structured.tools_used
            if not structured.sources and real_sources:
                structured.sources = real_sources
            render_research_result(structured, real_tool_calls, stream=stream)
        except OutputParserException as exc:
            logger.warning("Không phân tích được kết quả JSON: %s. Chuyển sang fallback structured.", exc)
            structured = ResearchResponse(
                topic=query,
                sources=real_sources,
                tools_used=real_tool_calls,
                summary=final_output,
            )
            render_research_result(structured, real_tool_calls, stream=stream)

    transcript.add_turn(
        question=query,
        raw_output=final_output,
        structured=structured,
        tool_calls=real_tool_calls,
        tool_steps=accumulated_tool_steps,
    )

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=final_output))
    if max_history > 0:
        chat_history[:] = chat_history[-max_history:]

    SessionManager.save_session_state(
        session_dir=run_dir,
        chat_history=chat_history,
        metadata={
            "last_topic": structured.topic if structured else "N/A",
            "turns": len(transcript.turns) + transcript.turn_offset,
        },
    )


def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Research Agent CLI (LangGraph Multi-Agent)")
    p.add_argument(
        "--query",
        help="Chạy một câu hỏi duy nhất rồi thoát (bỏ qua để vào chế độ hỏi liên tục).",
    )
    p.add_argument(
        "--session",
        help="Tiếp tục một phiên làm việc cũ bằng Session ID hoặc đường dẫn thư mục.",
    )
    p.add_argument(
        "--list-sessions",
        action="store_true",
        help="Liệt kê danh sách các phiên nghiên cứu đã lưu.",
    )
    p.add_argument(
        "--show-graph",
        action="store_true",
        help="Hiển thị sơ đồ kiến trúc các Node trong LangGraph.",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="Tắt hiệu ứng streaming khi hiển thị phần tóm tắt.",
    )
    p.add_argument(
        "--env-file", default=".env", help="Đường dẫn file .env (mặc định: .env)."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)

    try:
        settings: Settings = load_settings(args.env_file)
    except ConfigError as exc:
        console.print(f"[bold red]Lỗi cấu hình:[/bold red] {exc}", file=sys.stderr)
        return 1

    setup_logging(settings.log_level)

    # Khởi tạo đồ thị LangGraph
    graph, output_parser = build_research_graph(settings)

    if args.show_graph:
        ascii_art = get_graph_ascii(graph)
        mermaid_art = get_graph_mermaid(graph)
        render_graph_view(ascii_art, mermaid_art)
        return 0

    if args.list_sessions:
        sessions = SessionManager.list_sessions(settings.output_dir)
        render_session_list(sessions)
        return 0

    run_dir, is_resumed = resolve_run_dir(settings.output_dir, args.session)
    session_id = run_dir.name

    trace_logger = JsonlTraceLogger(run_dir / "trace.jsonl", session_id=session_id)
    transcript = MarkdownTranscriptWriter(
        run_dir / "transcript.md", title="Lịch sử nghiên cứu của Agent"
    )

    spinner = SimpleNodeSpinner(console)

    chat_history: list = []
    if is_resumed:
        chat_history, meta = SessionManager.load_session_state(run_dir)
        console.print(
            f"[bold green]✔ Đã nạp lại phiên làm việc:[/] [cyan]{session_id}[/] "
            f"([yellow]{len(chat_history) // 2} câu hỏi trước đó[/])"
        )
    else:
        console.print(f"[bold green]Phiên làm việc mới:[/] [cyan]{session_id}[/]")

    console.print(f"[dim]Transcript     :[/] {transcript.path.resolve()}")
    console.print(f"[dim]Trace kỹ thuật :[/dim] {trace_logger.path.resolve()}\n")

    stream_enabled = not args.no_stream

    try:
        if args.query:
            run_query(
                query=args.query,
                graph=graph,
                parser=output_parser,
                chat_history=chat_history,
                transcript=transcript,
                run_dir=run_dir,
                max_history=settings.max_chat_history_messages,
                stream=stream_enabled,
                spinner=spinner,
                trace_logger=trace_logger,
            )
        else:
            console.print("[dim]Gõ 'exit' hoặc 'quit' để thoát. Gõ '/graph' để xem sơ đồ Node.[/dim]\n")
            while True:
                query = console.input("[bold cyan]Nhập câu hỏi:[/] ").strip()
                if query.lower() in {"exit", "quit"}:
                    break
                if not query:
                    continue
                if query.lower() == "/graph":
                    render_graph_view(get_graph_ascii(graph), get_graph_mermaid(graph))
                    continue
                try:
                    run_query(
                        query=query,
                        graph=graph,
                        parser=output_parser,
                        chat_history=chat_history,
                        transcript=transcript,
                        run_dir=run_dir,
                        max_history=settings.max_chat_history_messages,
                        stream=stream_enabled,
                        spinner=spinner,
                        trace_logger=trace_logger,
                    )
                except Exception:
                    logger.exception("Lỗi khi xử lý câu hỏi trong LangGraph, agent vẫn tiếp tục chạy.")
                    console.print("[bold red]Có lỗi xảy ra khi xử lý câu hỏi, vui lòng thử lại.[/bold red]")
                console.print()
    except KeyboardInterrupt:
        console.print("\n[yellow]Đã dừng theo yêu cầu người dùng.[/yellow]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
