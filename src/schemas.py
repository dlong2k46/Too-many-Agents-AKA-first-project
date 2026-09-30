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
    sources: list[str] = Field(
        default_factory=list, description="Danh sách nguồn tham khảo (URL hoặc tên nguồn)"
    )
    tools_used: list[str] = Field(
        default_factory=list, description="Danh sách tool đã sử dụng"
    )
    confidence_score: int = Field(
        default=90,
        description="Điểm tin cậy (1-100) dựa trên tính nhất quán và độ xác thực của nguồn",
    )
    confidence_reason: str = Field(
        default="Dữ liệu được kiểm chứng từ các nguồn tra cứu khả dụng.",
        description="Lý do ngắn gọn cho điểm tin cậy",
    )
    summary: str = Field(description="Tóm tắt nội dung nghiên cứu súc tích, có cấu trúc")

