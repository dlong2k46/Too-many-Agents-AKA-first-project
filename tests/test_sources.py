"""
tests/test_sources.py
---------------------
Unit tests cho logic trích xuất nguồn tự động từ tool steps (TDD).
"""

from __future__ import annotations

from src.schemas import ResearchResponse
from src.transcript import ToolStep
from src.agent import extract_sources_from_tool_steps


def test_extract_sources_from_fetch_url():
    steps = [
        ToolStep(
            tool="fetch_url_tool",
            input={"url": "https://www.cnn.com/2026/01/15/style/fashion"},
            output="Bài viết phong cách...",
        ),
        ToolStep(
            tool="fetch_url_tool",
            input="https://www.elleman.vn/phong-cach/trend",
            output="Xu hướng...",
        ),
    ]
    sources = extract_sources_from_tool_steps(steps)
    assert "https://www.cnn.com/2026/01/15/style/fashion" in sources
    assert "https://www.elleman.vn/phong-cach/trend" in sources
    assert len(sources) == 2


def test_extract_sources_from_search_tool_output():
    search_output = (
        "- Bài viết 1: Tóm tắt 1... (nguồn: https://example.com/item1)\n"
        "- Bài viết 2: Tóm tắt 2... (nguồn: https://example.com/item2)"
    )
    steps = [
        ToolStep(
            tool="search_tool",
            input={"query": "thời trang 2026"},
            output=search_output,
        )
    ]
    sources = extract_sources_from_tool_steps(steps)
    assert "https://example.com/item1" in sources
    assert "https://example.com/item2" in sources


def test_extract_sources_deduplication():
    steps = [
        ToolStep(
            tool="fetch_url_tool",
            input={"url": "https://example.com/item1"},
            output="Nội dung...",
        ),
        ToolStep(
            tool="search_tool",
            input={"query": "test"},
            output="- Item (nguồn: https://example.com/item1)",
        ),
    ]
    sources = extract_sources_from_tool_steps(steps)
    assert sources == ["https://example.com/item1"]


def test_research_response_schema_order():
    """Kiểm tra schema ResearchResponse có sources đứng trước summary."""
    schema = ResearchResponse.model_json_schema()
    properties = list(schema.get("properties", {}).keys())
    assert "sources" in properties
    assert "summary" in properties
    assert properties.index("sources") < properties.index("summary")
