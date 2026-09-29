"""
schemas.py
----------
Định nghĩa hình dạng dữ liệu mà agent phải trả về, dùng chung cho:
- PydanticOutputParser (ép LLM tuân theo định dạng này qua prompt)
- transcript.py (ghi kết quả ra file cho người đọc)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchResponse(BaseModel):
    topic: str = Field(description="Chủ đề nghiên cứu")
    summary: str = Field(description="Tóm tắt nội dung nghiên cứu")
    sources: list[str] = Field(
        default_factory=list, description="Danh sách nguồn tham khảo"
    )
    tools_used: list[str] = Field(
        default_factory=list, description="Danh sách tool đã sử dụng"
    )
