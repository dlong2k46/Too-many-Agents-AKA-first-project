import unittest
from unittest.mock import patch


class TestTools(unittest.TestCase):
    @patch("tools.trafilatura.fetch_url")
    @patch("tools.trafilatura.extract")
    def test_fetch_url_tool_success(self, mock_extract, mock_fetch):
        from tools import fetch_url_tool

        mock_fetch.return_value = "<html><body><h1>Bài báo AI</h1><p>Nội dung chi tiết...</p></body></html>"
        mock_extract.return_value = "Bài báo AI\nNội dung chi tiết..."

        result = fetch_url_tool.invoke({"url": "https://example.com/ai-agent"})
        self.assertIn("Bài báo AI", result)
        self.assertIn("Nội dung chi tiết...", result)

    @patch("tools.trafilatura.fetch_url")
    def test_fetch_url_tool_failure(self, mock_fetch):
        from tools import fetch_url_tool

        mock_fetch.return_value = None
        result = fetch_url_tool.invoke({"url": "https://invalid-website.com/not-found"})
        self.assertIn("Không thể trích xuất nội dung", result)

    def test_fetch_url_tool_invalid_url(self):
        from tools import fetch_url_tool

        result = fetch_url_tool.invoke({"url": "invalid-url"})
        self.assertIn("URL không hợp lệ", result)

    @patch("tools.trafilatura.fetch_url")
    @patch("tools.trafilatura.extract")
    def test_fetch_url_tool_detects_binary_garbage(self, mock_extract, mock_fetch):
        from tools import fetch_url_tool

        mock_fetch.return_value = "dummy"
        # Giả lập chuỗi nhị phân rác chứa nhiều ký tự lỗi như trong trace.jsonl
        mock_extract.return_value = "/X:!4E@Ef00c\\5\"M-(E_~yc֡YU-Z"

        result = fetch_url_tool.invoke({"url": "https://example.com/binary-corrupted"})
        self.assertIn("mã hóa nhị phân hoặc không đọc được", result)


if __name__ == "__main__":
    unittest.main()

