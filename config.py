"""
config.py
---------
Tất cả tham số cấu hình của ứng dụng tập trung ở ĐÚNG MỘT NƠI.

Nguyên tắc: không rải os.getenv(...) khắp các file khác. Mọi module khác chỉ
nhận vào một object Settings đã được kiểm tra hợp lệ, thay vì tự đọc biến
môi trường — nhờ vậy khi cần đổi nguồn cấu hình (ví dụ đọc từ file YAML thay
vì .env) chỉ cần sửa đúng file này.

Nếu thiếu biến bắt buộc hoặc giá trị sai kiểu, chương trình dừng NGAY khi
khởi động với thông báo rõ ràng — thay vì lỗi khó hiểu giữa chừng lúc gọi API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised khi cấu hình bị thiếu hoặc không hợp lệ."""


@dataclass(frozen=True)
class Settings:
    # --- Model / API ---
    nvidia_api_key: str
    model_name: str
    model_base_url: str
    temperature: float
    max_tokens: int

    # --- Hành vi agent ---
    max_iterations: int
    max_chat_history_messages: int

    # --- Nơi lưu output ---
    output_dir: Path

    # --- Logging ---
    log_level: str

    # --- LangSmith (tùy chọn) ---
    langsmith_tracing: bool


def _get_float(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Biến {name}='{raw}' phải là số thực (ví dụ 0.2).") from exc


def _get_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Biến {name}='{raw}' phải là số nguyên.") from exc


def load_settings(env_file: str | Path = ".env") -> Settings:
    """Nạp .env và trả về Settings đã kiểm tra hợp lệ.

    Raises:
        ConfigError: nếu thiếu biến bắt buộc hoặc giá trị sai định dạng.
    """
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    # Nếu không có .env, vẫn cho phép chạy nếu biến đã có sẵn trong môi trường
    # (ví dụ khi deploy, biến được set trực tiếp bởi hệ thống CI/CD).

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "Thiếu NVIDIA_API_KEY. Sao chép .env.example thành .env và điền "
            "khóa API của bạn trước khi chạy."
        )

    settings = Settings(
        nvidia_api_key=api_key,
        model_name=os.getenv("MODEL_NAME", "deepseek-ai/deepseek-v4.1-flash"),
        model_base_url=os.getenv(
            "MODEL_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ),
        temperature=_get_float("MODEL_TEMPERATURE", "0.2"),
        max_tokens=_get_int("MODEL_MAX_TOKENS", "3072"),
        max_iterations=_get_int("MAX_ITERATIONS", "4"),
        max_chat_history_messages=_get_int("MAX_CHAT_HISTORY_MESSAGES", "10"),
        output_dir=Path(os.getenv("OUTPUT_DIR", "runs")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
    )

    if settings.max_iterations < 1:
        raise ConfigError("MAX_ITERATIONS phải >= 1.")
    if not (0.0 <= settings.temperature <= 2.0):
        raise ConfigError("MODEL_TEMPERATURE nên nằm trong khoảng 0.0 - 2.0.")

    return settings
