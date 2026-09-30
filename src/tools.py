"""
tools.py
--------
Định nghĩa các tool agent được phép gọi.

QUAN TRỌNG: LLM chọn tool CHỈ dựa vào tên và docstring bên dưới — không thấy
được code bên trong. Docstring càng nói rõ "làm gì / input dạng gì / khi nào
nên và không nên dùng" thì agent càng chọn đúng.
"""

from __future__ import annotations

import logging

from ddgs import DDGS
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import trafilatura

import requests

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
}


def _is_binary_or_corrupted(text: str) -> bool:
    """Kiểm tra xem chuỗi có phải là dữ liệu nhị phân/mã hóa rác không."""
    if not text:
        return False
    # 1. Có ký tự thay thế lỗi Unicode
    if "\ufffd" in text:
        return True
    sample = text[:1000]
    # 2. Chứa ký tự điều khiển nhị phân (trừ \n, \r, \t)
    if any(ord(c) < 32 and c not in "\n\r\t" for c in sample):
        return True
    # 3. Một bài viết văn bản thông thường phải có chữ cái và khoảng trắng tự nhiên
    if len(sample) >= 15:
        alpha_count = sum(1 for c in sample if c.isalpha())
        space_count = sum(1 for c in sample if c.isspace())
        if (alpha_count / len(sample)) < 0.45 or space_count == 0:
            return True

    return False





@tool
def search_tool(query: str) -> str:
    """Search the public web for recent or general information.

    Input: a short search query (3-8 keywords).
    Use this for: news, current events, product info, or facts that may
    have changed recently.
    Do NOT use this for: well-established background knowledge — use
    wiki_tool instead.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as exc:  # DDGS có thể lỗi mạng, rate-limit...
        logger.warning("search_tool lỗi khi tìm '%s': %s", query, exc)
        return f"Tìm kiếm thất bại: {exc}"

    if not results:
        return "Không tìm thấy kết quả nào cho truy vấn này."

    lines = [
        f"- {r.get('title', '(không tiêu đề)')}: {r.get('body', '')} "
        f"(nguồn: {r.get('href', '')})"
        for r in results
    ]
    return "\n".join(lines)


_wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1500)

wiki_tool = WikipediaQueryRun(
    api_wrapper=_wiki_wrapper,
    name="wiki_tool",
    description=(
        "Look up well-established background knowledge on Wikipedia. "
        "Input: the subject name, e.g. 'Albert Einstein' or 'Photosynthesis'. "
        "Do NOT use this for very recent events — use search_tool instead."
    ),
)


@tool
def fetch_url_tool(url: str) -> str:
    """Scrape and read the full text content of a specific webpage or article.

    Input: a valid webpage URL starting with http:// or https://.
    Use this when: search_tool gave you a relevant URL and you need the full,
    in-depth details, data, or technical explanation from that specific article.
    """
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return "URL không hợp lệ. Vui lòng cung cấp URL bắt đầu bằng http:// hoặc https://."

    try:
        downloaded = None
        # Ưu tiên dùng requests với Browser Headers để trang web tự động giải nén và không chặn bot
        try:
            res = requests.get(url, headers=DEFAULT_HEADERS, timeout=12)
            if res.status_code == 200 and res.text:
                downloaded = res.text
        except Exception:
            pass

        # Fallback sang trafilatura.fetch_url nếu requests không tải được
        if not downloaded:
            downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return f"Không thể trích xuất nội dung từ URL '{url}' (trang web không phản hồi hoặc bị chặn)."

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if not text or not text.strip():
            return f"Không thể trích xuất nội dung từ URL '{url}' (trang web không chứa bài viết văn bản khả dụng)."

        text = text.strip()

        # Phát hiện và loại bỏ dữ liệu nhị phân rác hoặc bị mã hóa
        if _is_binary_or_corrupted(text):
            return f"Không thể trích xuất nội dung từ URL '{url}' (nội dung trang bị mã hóa nhị phân hoặc không đọc được)."

        max_chars = 2500
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... (Đã cắt ngắn nội dung từ tổng số {len(text)} ký tự)"
        return text
    except Exception as exc:
        logger.warning("fetch_url_tool lỗi khi đọc '%s': %s", url, exc)
        return f"Lỗi khi đọc trang web '{url}': {exc}"


