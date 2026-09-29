"""
output_parsing.py
------------------
Phân tích chuỗi văn bản thô mà agent trả về thành object ResearchResponse.

Tách riêng khỏi main.py để test được mà không cần gọi LLM thật (xem
tests/test_output_parsing.py).
"""

from __future__ import annotations

import re

from langchain_core.output_parsers import PydanticOutputParser

from schemas import ResearchResponse

# Lấy khối {...} đầu tiên đến cuối cùng — phòng khi LLM bọc JSON trong
# ```json ... ``` hoặc thêm câu dẫn phía trước/sau.
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_block(text: str) -> str:
    """Trả về khối JSON trong `text`, hoặc nguyên văn `text` nếu không tìm thấy."""
    match = _JSON_BLOCK_RE.search(text)
    return match.group(0) if match else text


def parse_agent_output(
    text: str, parser: PydanticOutputParser
) -> ResearchResponse:
    """Phân tích output thô của agent thành ResearchResponse.

    Raises:
        langchain_core.exceptions.OutputParserException: nếu JSON không hợp lệ
            hoặc thiếu trường bắt buộc.
    """
    return parser.parse(extract_json_block(text))
