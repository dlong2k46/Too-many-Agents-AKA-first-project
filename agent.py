"""
agent.py
--------
Khởi tạo hệ thống Multi-Agent nghiên cứu bằng LangGraph:
- Node 1 ("researcher"): Chuyên tra cứu web, wiki và cào sâu nội dung trang web.
- Node 2 ("tools"): Thực thi các công cụ và ghi lại các bước tra cứu (ToolStep).
- Node 3 ("critic_writer"): Phản biện tính đúng đắn của dữ liệu và viết báo cáo chuyên sâu JSON.
"""

from __future__ import annotations

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

from config import Settings
from output_parsing import parse_agent_output
from prompts import CRITIC_WRITER_SYSTEM_PROMPT, RESEARCHER_SYSTEM_PROMPT
from schemas import ResearchResponse
from tools import fetch_url_tool, search_tool, wiki_tool
from transcript import ToolStep

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s\)\],\"'<>]+")


def extract_sources_from_tool_steps(tool_steps: list[Any]) -> list[str]:
    """Trích xuất danh sách URL và nguồn thực tế từ lịch sử chạy tool."""
    sources: list[str] = []
    seen = set()

    def _add(s: str) -> None:
        cleaned = s.strip().rstrip(".,;)")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            sources.append(cleaned)

    for step in tool_steps:
        tool_name = getattr(step, "tool", "")
        step_input = getattr(step, "input", None)
        step_output = getattr(step, "output", "")

        # 1. URL từ fetch_url_tool
        if tool_name == "fetch_url_tool":
            if isinstance(step_input, dict):
                url = str(step_input.get("url", ""))
            else:
                url = str(step_input or "")
            if url.startswith("http://") or url.startswith("https://"):
                _add(url)

        # 2. URL trong kết quả của search_tool
        elif tool_name == "search_tool":
            for match in _URL_RE.finditer(step_output):
                _add(match.group(0))

        # 3. Wikipedia query
        elif tool_name == "wiki_tool":
            q = (
                step_input.get("query")
                if isinstance(step_input, dict)
                else str(step_input or "")
            )
            if q:
                _add(f"Wikipedia: {q}")

        # Các trường hợp khác có link
        else:
            for match in _URL_RE.finditer(step_output):
                _add(match.group(0))

    return sources


def extract_sources_from_chat_history(chat_history: list[BaseMessage]) -> list[str]:
    """Trích xuất các URL và nguồn đã có trong lịch sử trò chuyện (khi câu hỏi sau kế thừa câu trước)."""
    sources: list[str] = []
    seen = set()

    def _add(s: str) -> None:
        cleaned = s.strip().rstrip(".,;)")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            sources.append(cleaned)

    for msg in chat_history:
        text = str(getattr(msg, "content", ""))
        # 1. Nếu tin nhắn AI chứa JSON có trường sources
        if "sources" in text:
            try:
                import json
                from output_parsing import extract_json_block
                data = json.loads(extract_json_block(text))
                for src in data.get("sources", []):
                    if isinstance(src, str) and not src.startswith("Không có"):
                        _add(src)
            except Exception:
                pass
        # 2. Bắt tất cả các URL tìm thấy trong nội dung tin nhắn
        for match in _URL_RE.finditer(text):
            _add(match.group(0))

    return sources


def format_critic_writer_context(
    query: str,
    chat_history: list[BaseMessage],
    tool_steps: list[ToolStep],
    researcher_message: str | None = None,
) -> str:
    """Tạo nội dung ngữ cảnh đầy đủ gửi cho CriticWriter Node (kết hợp history + tool + researcher)."""
    blocks: list[str] = []

    # 1. Ngữ cảnh lịch sử trò chuyện (nếu có)
    if chat_history:
        history_lines = ["=== NGỮ CẢNH TỪ CÁC CÂU HỎI TRƯỚC ĐÓ ==="]
        for msg in chat_history[-6:]:
            role = "Người dùng" if isinstance(msg, HumanMessage) else "Agent"
            content = str(getattr(msg, "content", "")).strip()
            if role == "Agent" and "{" in content and "summary" in content:
                try:
                    import json
                    from output_parsing import extract_json_block
                    parsed = json.loads(extract_json_block(content))
                    content = parsed.get("summary", content)[:600]
                except Exception:
                    content = content[:600]
            else:
                content = content[:400]
            history_lines.append(f"- {role}: {content}")
        blocks.append("\n".join(history_lines))

    # 2. Câu hỏi hiện tại của người dùng
    blocks.append(f"=== CÂU HỎI NGHIÊN CỨU HIỆN TẠI ===\n{query}")

    # 3. Phân tích hoặc đề xuất trực tiếp từ Researcher Agent (nếu có)
    if researcher_message and researcher_message.strip():
        blocks.append(
            f"=== PHÂN TÍCH & KẾT LUẬN CỦA RESEARCHER AGENT ===\n{researcher_message.strip()}"
        )

    # 4. Dữ liệu cào & tra cứu từ các tool trong lượt này
    if tool_steps:
        findings_blocks = ["=== DỮ LIỆU THU THẬP TỪ CÔNG CỤ TRONG LƯỢT NÀY ==="]
        for s in tool_steps:
            findings_blocks.append(
                f"--- [Nguồn / Tool: {s.tool} | Truy vấn: {s.input}] ---\n{s.output}"
            )
        blocks.append("\n\n".join(findings_blocks))
    else:
        if not chat_history and not researcher_message:
            blocks.append("(Không thu thập được thông tin từ công cụ ngoài; sử dụng kiến thức nền).")

    blocks.append("Hãy phản biện, đối chiếu loại bỏ mâu thuẫn và xuất bài báo cáo JSON hoàn chỉnh.")
    return "\n\n".join(blocks)


class ResearchState(TypedDict):
    """Trạng thái chia sẻ giữa các Node trong LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    chat_history: list[BaseMessage]
    tool_steps: list[ToolStep]
    iteration: int
    structured_response: Optional[ResearchResponse]
    final_output: str


def build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.model_base_url,
        api_key=settings.nvidia_api_key,
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def build_research_graph(settings: Settings) -> tuple[Any, PydanticOutputParser]:
    """Xây dựng và biên dịch Đồ thị Multi-Agent LangGraph."""
    llm = build_llm(settings)
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    tools = [search_tool, wiki_tool, fetch_url_tool]
    tools_by_name = {t.name: t for t in tools}
    researcher_llm = llm.bind_tools(tools)

    # -------------------------------------------------------------
    # Node 1: Researcher Node
    # -------------------------------------------------------------
    def researcher_node(state: ResearchState) -> dict[str, Any]:
        sys_msg = SystemMessage(content=RESEARCHER_SYSTEM_PROMPT)
        history = state.get("chat_history", [])
        current_msgs = state.get("messages", [])

        # Ghép system prompt + lịch sử trước đó + các tin nhắn trong lượt hiện tại
        response = researcher_llm.invoke([sys_msg] + history + current_msgs)
        iteration = state.get("iteration", 0) + (1 if getattr(response, "tool_calls", None) else 0)

        return {
            "messages": [response],
            "iteration": iteration,
        }

    # -------------------------------------------------------------
    # Node 2: Tools Node (Thực thi và lưu lại ToolStep)
    # -------------------------------------------------------------
    def tools_node(state: ResearchState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        new_tool_steps = list(state.get("tool_steps", []))
        tool_messages: list[ToolMessage] = []

        tool_calls = getattr(last_message, "tool_calls", [])
        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call.get("args", {})
            t = tools_by_name.get(tool_name)
            if t:
                try:
                    obs = t.invoke(tool_args)
                except Exception as exc:
                    obs = f"Lỗi khi thực thi công cụ '{tool_name}': {exc}"
            else:
                obs = f"Không tìm thấy công cụ '{tool_name}'."

            tool_messages.append(
                ToolMessage(
                    content=str(obs),
                    tool_call_id=call["id"],
                    name=tool_name,
                )
            )
            new_tool_steps.append(
                ToolStep(
                    tool=tool_name,
                    input=tool_args,
                    output=str(obs),
                )
            )

        return {
            "messages": tool_messages,
            "tool_steps": new_tool_steps,
        }

    # -------------------------------------------------------------
    # Node 3: Critic & Writer Node (Phản biện & Viết báo cáo chuẩn)
    # -------------------------------------------------------------
    def critic_writer_node(state: ResearchState) -> dict[str, Any]:
        query = state.get("query", "")
        tool_steps = state.get("tool_steps", [])
        chat_history = state.get("chat_history", [])

        # Lấy nội dung phản hồi cuối của Researcher nếu có (khi researcher tự tổng hợp)
        researcher_msg = None
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None) and str(msg.content).strip():
                researcher_msg = str(msg.content).strip()
                break

        format_inst = parser.get_format_instructions()
        system_content = CRITIC_WRITER_SYSTEM_PROMPT.format(format_instructions=format_inst)

        user_content = format_critic_writer_context(
            query=query,
            chat_history=chat_history,
            tool_steps=tool_steps,
            researcher_message=researcher_msg,
        )

        response = llm.invoke(
            [
                SystemMessage(content=system_content),
                HumanMessage(content=user_content),
            ]
        )

        raw_text = str(response.content)
        structured = None
        real_tools = sorted({s.tool for s in tool_steps})
        real_sources = extract_sources_from_tool_steps(tool_steps)

        # Nếu lượt này không gọi tool mới nhưng kế thừa từ lịch sử, kế thừa nguồn từ lịch sử
        if not real_sources and chat_history:
            real_sources = extract_sources_from_chat_history(chat_history)

        try:
            structured = parse_agent_output(raw_text, parser)
            structured.tools_used = real_tools or structured.tools_used

            # Bù hoặc hợp nhất nguồn tham khảo thực tế đã truy vấn
            if not structured.sources:
                structured.sources = real_sources
            else:
                combined_sources: list[str] = []
                seen_src: set[str] = set()
                for src in structured.sources + real_sources:
                    if src not in seen_src:
                        seen_src.add(src)
                        combined_sources.append(src)
                structured.sources = combined_sources
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

    # -------------------------------------------------------------
    # Điều kiện chuyển hướng (Conditional Edge)
    # -------------------------------------------------------------
    def should_continue_research(state: ResearchState) -> Literal["tools", "critic_writer"]:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None)
        iteration = state.get("iteration", 0)

        if tool_calls and iteration <= settings.max_iterations:
            return "tools"
        return "critic_writer"

    # -------------------------------------------------------------
    # Xây dựng StateGraph
    # -------------------------------------------------------------
    workflow = StateGraph(ResearchState)

    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("critic_writer", critic_writer_node)

    workflow.add_edge(START, "researcher")
    workflow.add_conditional_edges(
        "researcher",
        should_continue_research,
        {
            "tools": "tools",
            "critic_writer": "critic_writer",
        },
    )
    workflow.add_edge("tools", "researcher")
    workflow.add_edge("critic_writer", END)

    app = workflow.compile()
    return app, parser


def get_graph_ascii(graph: Any) -> str:
    """Trả về sơ đồ ASCII của đồ thị để in ra terminal."""
    try:
        return graph.get_graph().draw_ascii()
    except Exception as exc:
        return f"(Không thể vẽ sơ đồ ASCII: {exc})"


def get_graph_mermaid(graph: Any) -> str:
    """Trả về sơ đồ Mermaid của đồ thị."""
    try:
        return graph.get_graph().draw_mermaid()
    except Exception as exc:
        return f"(Không thể vẽ sơ đồ Mermaid: {exc})"
