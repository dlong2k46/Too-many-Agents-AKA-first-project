"""
main.py
-------
Điểm vào chương trình Research Agent (Kiến trúc Multi-Agent bằng LangGraph).

Cách sử dụng:
    python main.py                          # Chế độ hỏi đáp liên tục qua dòng lệnh
    python main.py --query "AI Agent là gì?" # Chạy một câu hỏi duy nhất rồi thoát
    python main.py --list-sessions          # Xem danh sách các phiên nghiên cứu đã lưu
    python main.py --session <session_id>   # Nạp lại phiên nghiên cứu cũ
    python main.py --show-graph             # Xem sơ đồ kiến trúc các Node
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
import time
from typing import Any
import uuid

# Cấu hình UTF-8 cho console trên Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from src.agent import (
    build_research_graph,
    extract_sources_from_chat_history,
    extract_sources_from_tool_steps,
    get_graph_ascii,
    get_graph_mermaid,
)
from src.config import ConfigError, Settings, calculate_token_cost, load_settings
from src.observability import JsonlTraceLogger
from src.output_parsing import parse_agent_output
from src.schemas import ResearchResponse
from src.session_manager import SessionManager
from src.transcript import MarkdownTranscriptWriter, ToolStep

logger = logging.getLogger(__name__)


# =====================================================================
# Terminal Output Helpers (Không dùng thư viện bên ngoài)
# =====================================================================

def print_research_result(
    structured: ResearchResponse,
    real_tool_calls: list[str],
    elapsed_seconds: float = 0.0,
    token_usage: dict[str, int] | None = None,
    estimated_cost: float = 0.0,
) -> None:
    """In kết quả nghiên cứu sạch sẽ, rõ ràng kèm các chỉ số hiệu năng."""
    print("\n" + "=" * 60)
    print(f"📌 CHỦ ĐỀ: {structured.topic}")
    print("=" * 60)
    print("\n📝 TÓM TẮT BÁO CÁO:\n")
    print(structured.summary.strip())
    print("\n" + "-" * 60)
    print("🔍 NGUỒN THAM KHẢO & CÔNG CỤ:")
    if structured.sources:
        for src in structured.sources:
            print(f"  • {src}")
    else:
        print("  (Không có nguồn ngoài hoặc dùng kiến thức nền)")
    tools_str = ", ".join(real_tool_calls) if real_tool_calls else "(Không gọi tool)"
    print(f"  Tool đã dùng: {tools_str}")

    print("-" * 60)
    print("📈 CHỈ SỐ THỰC THI (METRICS):")
    if elapsed_seconds > 0:
        print(f"  ⏱ Thời gian xử lý   : {elapsed_seconds:.2f} giây")

    if token_usage:
        p = token_usage.get("prompt_tokens", 0)
        c = token_usage.get("completion_tokens", 0)
        t = token_usage.get("total_tokens", p + c)
        print(f"  📊 Tiêu thụ Token   : {t:,} tokens (Prompt: {p:,} | Output: {c:,})")

    if estimated_cost > 0:
        vnd_cost = int(estimated_cost * 25400)
        print(f"  💰 Chi phí ước tính : ~${estimated_cost:.5f} (~{vnd_cost:,} VNĐ)")

    if hasattr(structured, "confidence_score"):
        print(f"  🎯 Độ tin cậy (AI)  : {structured.confidence_score}/100 ({structured.confidence_reason})")

    print("=" * 60 + "\n")



def print_session_list(sessions: list[dict[str, Any]]) -> None:
    """In danh sách các phiên làm việc dưới dạng bảng text đơn giản."""
    if not sessions:
        print("Chưa có phiên nghiên cứu nào được lưu.")
        return

    print("\n" + "=" * 75)
    print("📚 DANH SÁCH CÁC PHIÊN NGHIÊN CỨU (SESSIONS)")
    print("=" * 75)
    print(f"{'Session ID':<30} | {'Cập nhật gần nhất':<19} | {'Lượt':<5} | {'Chủ đề gần nhất'}")
    print("-" * 75)
    for s in sessions:
        updated = (s.get("updated_at") or s.get("created_at") or "-")[:19].replace("T", " ")
        print(f"{s['session_id']:<30} | {updated:<19} | {s.get('turns', 0):<5} | {s.get('last_topic', '(Chưa có chủ đề)')}")
    print("=" * 75)
    print("Gợi ý: Dùng lệnh 'python main.py --session <SESSION_ID>' để tiếp tục phiên làm việc cũ.\n")


def print_graph_view(ascii_art: str, mermaid_art: str | None = None) -> None:
    """In sơ đồ workflow của LangGraph ra terminal."""
    print("\n" + "=" * 60)
    print("🧩 SƠ ĐỒ KIẾN TRÚC LANGGRAPH (MULTI-AGENT WORKFLOW)")
    print("=" * 60)
    print(ascii_art)
    if mermaid_art:
        print("-" * 60)
        print("Cú pháp Mermaid (sao chép để vẽ trong Mermaid Live Editor):")
        print(f"```mermaid\n{mermaid_art}\n```")
    print("=" * 60 + "\n")


# =====================================================================
# Session & Directory Management
# =====================================================================

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


# =====================================================================
# Query Execution Engine
# =====================================================================

def run_query(
    query: str,
    graph: Any,
    parser: PydanticOutputParser,
    chat_history: list,
    transcript: MarkdownTranscriptWriter,
    run_dir: Path,
    max_history: int,
    settings: Settings,
    trace_logger: JsonlTraceLogger | None = None,
) -> None:
    """Chạy câu hỏi qua đồ thị LangGraph, hiển thị tiến trình, đo đạc metrics và lưu kết quả."""
    start_time = time.perf_counter()
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "chat_history": list(chat_history),
        "tool_steps": [],
        "iteration": 0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    accumulated_tool_steps: list[ToolStep] = []
    accumulated_token_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    final_output = ""
    structured = None

    print("\n⏳ Đang xử lý truy vấn và thu thập dữ liệu...")

    config = {}
    if trace_logger:
        config["callbacks"] = [trace_logger]

    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, updates in event.items():
            if "token_usage" in updates and updates["token_usage"]:
                accumulated_token_usage = updates["token_usage"]
            if node_name in {"researcher", "tools"}:
                steps = updates.get("tool_steps", [])
                if steps:
                    accumulated_tool_steps = steps
            elif node_name == "critic_writer":
                final_output = updates.get("final_output", "")
                structured = updates.get("structured_response")

    elapsed_seconds = time.perf_counter() - start_time
    p_tokens = accumulated_token_usage.get("prompt_tokens", 0)
    c_tokens = accumulated_token_usage.get("completion_tokens", 0)
    estimated_cost = calculate_token_cost(
        prompt_tokens=p_tokens,
        completion_tokens=c_tokens,
        cost_per_1m_input=settings.cost_per_1m_input_tokens,
        cost_per_1m_output=settings.cost_per_1m_output_tokens,
    )

    real_tool_calls = sorted({s.tool for s in accumulated_tool_steps})
    real_sources = extract_sources_from_tool_steps(accumulated_tool_steps)
    if not real_sources and chat_history:
        real_sources = extract_sources_from_chat_history(chat_history)

    if structured is not None:
        structured.tools_used = real_tool_calls or structured.tools_used
        if not structured.sources and real_sources:
            structured.sources = real_sources
        print_research_result(
            structured,
            real_tool_calls,
            elapsed_seconds=elapsed_seconds,
            token_usage=accumulated_token_usage,
            estimated_cost=estimated_cost,
        )
    elif final_output:
        try:
            structured = parse_agent_output(final_output, parser)
            structured.tools_used = real_tool_calls or structured.tools_used
            if not structured.sources and real_sources:
                structured.sources = real_sources
            print_research_result(
                structured,
                real_tool_calls,
                elapsed_seconds=elapsed_seconds,
                token_usage=accumulated_token_usage,
                estimated_cost=estimated_cost,
            )
        except OutputParserException as exc:
            logger.warning("Không phân tích được JSON: %s. Chuyển sang fallback.", exc)
            structured = ResearchResponse(
                topic=query,
                sources=real_sources,
                tools_used=real_tool_calls,
                summary=final_output,
            )
            print_research_result(
                structured,
                real_tool_calls,
                elapsed_seconds=elapsed_seconds,
                token_usage=accumulated_token_usage,
                estimated_cost=estimated_cost,
            )

    transcript.add_turn(
        question=query,
        raw_output=final_output,
        structured=structured,
        tool_calls=real_tool_calls,
        tool_steps=accumulated_tool_steps,
        elapsed_seconds=elapsed_seconds,
        token_usage=accumulated_token_usage,
        estimated_cost=estimated_cost,
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


# =====================================================================
# CLI Entrypoint
# =====================================================================

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
        "--env-file", default=".env", help="Đường dẫn file .env (mặc định: .env)."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)

    try:
        settings: Settings = load_settings(args.env_file)
    except ConfigError as exc:
        print(f"Lỗi cấu hình: {exc}", file=sys.stderr)
        return 1

    setup_logging(settings.log_level)

    graph, output_parser = build_research_graph(settings)

    if args.show_graph:
        print_graph_view(get_graph_ascii(graph), get_graph_mermaid(graph))
        return 0

    if args.list_sessions:
        sessions = SessionManager.list_sessions(settings.output_dir)
        print_session_list(sessions)
        return 0

    run_dir, is_resumed = resolve_run_dir(settings.output_dir, args.session)
    session_id = run_dir.name

    trace_logger = JsonlTraceLogger(run_dir / "trace.jsonl", session_id=session_id)
    transcript = MarkdownTranscriptWriter(
        run_dir / "transcript.md", title="Lịch sử nghiên cứu của Agent"
    )

    chat_history: list = []
    if is_resumed:
        chat_history, _ = SessionManager.load_session_state(run_dir)
        print(f"✔ Đã nạp lại phiên làm việc: {session_id} ({len(chat_history) // 2} câu hỏi trước đó)")
    else:
        print(f"Phiên làm việc mới: {session_id}")

    print(f"Transcript     : {transcript.path.resolve()}")
    print(f"Trace kỹ thuật : {trace_logger.path.resolve()}\n")

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
                settings=settings,
                trace_logger=trace_logger,
            )
        else:
            print("Gõ 'exit' hoặc 'quit' để thoát. Gõ '/graph' để xem sơ đồ Node.\n")
            while True:
                query = input("Nhập câu hỏi: ").strip()
                if query.lower() in {"exit", "quit"}:
                    break
                if not query:
                    continue
                if query.lower() == "/graph":
                    print_graph_view(get_graph_ascii(graph), get_graph_mermaid(graph))
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
                        settings=settings,
                        trace_logger=trace_logger,
                    )
                except Exception:
                    logger.exception("Lỗi khi xử lý câu hỏi trong LangGraph.")
                    print("Có lỗi xảy ra khi xử lý câu hỏi, vui lòng thử lại.")
                print()
    except KeyboardInterrupt:
        print("\nĐã dừng theo yêu cầu người dùng.")

    return 0



if __name__ == "__main__":
    raise SystemExit(main())
