"""
agent.py
--------
Kiến trúc Multi-Agent cho tác vụ nghiên cứu chuyên sâu sử dụng LangGraph:
- Node 1 ("researcher"): Chuyên tra cứu web, Wikipedia và đọc sâu bài viết.
- Node 2 ("tools"): Thực thi các công cụ và ghi lại vết truy vết (ToolStep).
- Node 3 ("critic_writer"): Phản biện dữ liệu, loại bỏ mâu thuẫn và xuất báo cáo có cấu trúc JSON.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import Settings
from .output_parsing import extract_json_block, parse_agent_output
from .prompts import CRITIC_WRITER_SYSTEM_PROMPT, RESEARCHER_SYSTEM_PROMPT
from .schemas import ResearchResponse
from .tools import fetch_url_tool, search_tool, wiki_tool
from .transcript import ToolStep

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s\)\],\"'<>]+")


# =====================================================================
# State Definition
# =====================================================================

class ResearchState(TypedDict):
    """Trạng thái chia sẻ giữa các Node trong LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    chat_history: list[BaseMessage]
    tool_steps: list[ToolStep]
    iteration: int
    structured_response: Optional[ResearchResponse]
    final_output: str


# =====================================================================
# Helper Functions (Sources & Context)
# =====================================================================

def extract_sources_from_tool_steps(tool_steps: list[Any]) -> list[str]:
    """Trích xuất danh sách URL và tên nguồn thực tế từ các bước chạy tool."""
    sources: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        cleaned = s.strip().rstrip(".,;)")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            sources.append(cleaned)

    for step in tool_steps:
        tool_name = getattr(step, "tool", "")
        step_input = getattr(step, "input", None)
        step_output = getattr(step, "output", "")

        if tool_name == "fetch_url_tool":
            url = step_input.get("url", "") if isinstance(step_input, dict) else str(step_input or "")
            if url.startswith(("http://", "https://")):
                _add(url)
        elif tool_name == "search_tool":
            for match in _URL_RE.finditer(step_output):
                _add(match.group(0))
        elif tool_name == "wiki_tool":
            query = step_input.get("query") if isinstance(step_input, dict) else str(step_input or "")
            if query:
                _add(f"Wikipedia: {query}")
        else:
            for match in _URL_RE.finditer(step_output):
                _add(match.group(0))

    return sources


def extract_sources_from_chat_history(chat_history: list[BaseMessage]) -> list[str]:
    """Trích xuất các URL và nguồn đã có trong lịch sử trò chuyện để kế thừa qua các lượt hỏi."""
    sources: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        cleaned = s.strip().rstrip(".,;)")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            sources.append(cleaned)

    for msg in chat_history:
        text = str(getattr(msg, "content", ""))
        if "sources" in text:
            try:
                data = json.loads(extract_json_block(text))
                for src in data.get("sources", []):
                    if isinstance(src, str) and not src.startswith("Không có"):
                        _add(src)
            except Exception:
                pass
        for match in _URL_RE.finditer(text):
            _add(match.group(0))

    return sources


def format_critic_writer_context(
    query: str,
    chat_history: list[BaseMessage],
    tool_steps: list[ToolStep],
    researcher_message: str | None = None,
) -> str:
    """Tạo nội dung ngữ cảnh đầy đủ gửi cho CriticWriter Node (kết hợp lịch sử + dữ liệu cào + phân tích)."""
    blocks: list[str] = []

    # 1. Ngữ cảnh lịch sử trò chuyện
    if chat_history:
        history_lines = ["=== NGỮ CẢNH TỪ CÁC CÂU HỎI TRƯỚC ĐÓ ==="]
        for msg in chat_history[-6:]:
            role = "Người dùng" if isinstance(msg, HumanMessage) else "Agent"
            content = str(getattr(msg, "content", "")).strip()
            if role == "Agent" and "{" in content and "summary" in content:
                try:
                    parsed = json.loads(extract_json_block(content))
                    content = parsed.get("summary", content)[:600]
                except Exception:
                    content = content[:600]
            else:
                content = content[:400]
            history_lines.append(f"- {role}: {content}")
        blocks.append("\n".join(history_lines))

    # 2. Câu hỏi hiện tại
    blocks.append(f"=== CÂU HỎI NGHIÊN CỨU HIỆN TẠI ===\n{query}")

    # 3. Phân tích trực tiếp từ Researcher (nếu có)
    if researcher_message and researcher_message.strip():
        blocks.append(f"=== PHÂN TÍCH & KẾT LUẬN CỦA RESEARCHER AGENT ===\n{researcher_message.strip()}")

    # 4. Dữ liệu từ các tool đã chạy
    if tool_steps:
        findings = ["=== DỮ LIỆU THU THẬP TỪ CÔNG CỤ TRONG LƯỢT NÀY ==="]
        for s in tool_steps:
            findings.append(f"--- [Nguồn / Tool: {s.tool} | Truy vấn: {s.input}] ---\n{s.output}")
        blocks.append("\n\n".join(findings))
    elif not chat_history and not researcher_message:
        blocks.append("(Không thu thập được thông tin từ công cụ ngoài; sử dụng kiến thức nền).")

    blocks.append("Hãy phản biện, đối chiếu loại bỏ mâu thuẫn và xuất bài báo cáo JSON hoàn chỉnh.")
    return "\n\n".join(blocks)


# =====================================================================
# Node Handlers (Top-level Functions)
# =====================================================================

def researcher_step(state: ResearchState, researcher_llm: Any) -> dict[str, Any]:
    """Node Researcher: phân tích yêu cầu và quyết định gọi công cụ tra cứu."""
    sys_msg = SystemMessage(content=RESEARCHER_SYSTEM_PROMPT)
    history = state.get("chat_history", [])
    current_msgs = state.get("messages", [])

    response = researcher_llm.invoke([sys_msg] + history + current_msgs)
    iteration = state.get("iteration", 0) + (1 if getattr(response, "tool_calls", None) else 0)

    return {
        "messages": [response],
        "iteration": iteration,
    }


def tools_step(state: ResearchState, tools_by_name: dict[str, Any]) -> dict[str, Any]:
    """Node Tools: thực thi các công cụ tra cứu và ghi lại ToolStep."""
    last_message = state["messages"][-1]
    new_tool_steps = list(state.get("tool_steps", []))
    tool_messages: list[ToolMessage] = []

    for call in getattr(last_message, "tool_calls", []):
        tool_name = call["name"]
        tool_args = call.get("args", {})
        tool = tools_by_name.get(tool_name)

        if tool:
            try:
                obs = tool.invoke(tool_args)
            except Exception as exc:
                obs = f"Lỗi khi thực thi công cụ '{tool_name}': {exc}"
        else:
            obs = f"Không tìm thấy công cụ '{tool_name}'."

        tool_messages.append(
            ToolMessage(content=str(obs), tool_call_id=call["id"], name=tool_name)
        )
        new_tool_steps.append(
            ToolStep(tool=tool_name, input=tool_args, output=str(obs))
        )

    return {
        "messages": tool_messages,
        "tool_steps": new_tool_steps,
    }


def critic_writer_step(
    state: ResearchState, llm: ChatOpenAI, parser: PydanticOutputParser
) -> dict[str, Any]:
    """Node Critic & Writer: phản biện thông tin và tổng hợp thành báo cáo JSON."""
    query = state.get("query", "")
    tool_steps = state.get("tool_steps", [])
    chat_history = state.get("chat_history", [])

    researcher_msg = None
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None) and str(msg.content).strip():
            researcher_msg = str(msg.content).strip()
            break

    system_prompt = CRITIC_WRITER_SYSTEM_PROMPT.format(format_instructions=parser.get_format_instructions())
    user_context = format_critic_writer_context(
        query=query,
        chat_history=chat_history,
        tool_steps=tool_steps,
        researcher_message=researcher_msg,
    )

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_context)])
    raw_text = str(response.content)

    real_tools = sorted({s.tool for s in tool_steps})
    real_sources = extract_sources_from_tool_steps(tool_steps)
    if not real_sources and chat_history:
        real_sources = extract_sources_from_chat_history(chat_history)

    try:
        structured = parse_agent_output(raw_text, parser)
        structured.tools_used = real_tools or structured.tools_used

        if not structured.sources:
            structured.sources = real_sources
        else:
            merged: list[str] = []
            seen: set[str] = set()
            for src in structured.sources + real_sources:
                if src not in seen:
                    seen.add(src)
                    merged.append(src)
            structured.sources = merged
    except Exception as exc:
        logger.warning("Lỗi phân tích JSON ở CriticWriter: %s", exc)
        structured = ResearchResponse(
            topic=query,
            sources=real_sources,
            tools_used=real_tools,
            summary=raw_text,
        )

    return {
        "final_output": raw_text,
        "structured_response": structured,
    }


def should_continue_research(
    state: ResearchState, max_iterations: int = 5
) -> Literal["tools", "critic_writer"]:
    """Điều kiện rẽ nhánh: gọi tool tiếp hay chuyển sang tổng hợp báo cáo."""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    iteration = state.get("iteration", 0)

    if tool_calls and iteration <= max_iterations:
        return "tools"
    return "critic_writer"


# =====================================================================
# Graph Construction & Visualization
# =====================================================================

def build_llm(settings: Settings) -> ChatOpenAI:
    """Tạo đối tượng ChatOpenAI từ cấu hình hệ thống."""
    return ChatOpenAI(
        base_url=settings.model_base_url,
        api_key=settings.nvidia_api_key,
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def build_research_graph(settings: Settings) -> tuple[Any, PydanticOutputParser]:
    """Xây dựng và biên dịch đồ thị LangGraph Multi-Agent."""
    llm = build_llm(settings)
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    tools = [search_tool, wiki_tool, fetch_url_tool]
    tools_by_name = {t.name: t for t in tools}
    researcher_llm = llm.bind_tools(tools)

    def node_researcher(state: ResearchState) -> dict[str, Any]:
        return researcher_step(state, researcher_llm)

    def node_tools(state: ResearchState) -> dict[str, Any]:
        return tools_step(state, tools_by_name)

    def node_critic(state: ResearchState) -> dict[str, Any]:
        return critic_writer_step(state, llm, parser)

    def condition_route(state: ResearchState) -> Literal["tools", "critic_writer"]:
        return should_continue_research(state, max_iterations=settings.max_iterations)

    workflow = StateGraph(ResearchState)
    workflow.add_node("researcher", node_researcher)
    workflow.add_node("tools", node_tools)
    workflow.add_node("critic_writer", node_critic)

    workflow.add_edge(START, "researcher")
    workflow.add_conditional_edges(
        "researcher",
        condition_route,
        {"tools": "tools", "critic_writer": "critic_writer"},
    )
    workflow.add_edge("tools", "researcher")
    workflow.add_edge("critic_writer", END)

    return workflow.compile(), parser


def get_graph_ascii(graph: Any) -> str:
    """Trả về sơ đồ ASCII của đồ thị."""
    try:
        return graph.get_graph().draw_ascii()
    except Exception as exc:
        return f"(Không thể vẽ sơ đồ ASCII: {exc})"


def get_graph_mermaid(graph: Any) -> str:
    """Trả về cú pháp sơ đồ Mermaid của đồ thị."""
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as exc:
        return f"(Không thể vẽ sơ đồ Mermaid: {exc})"
