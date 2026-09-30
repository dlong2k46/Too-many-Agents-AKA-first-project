import unittest
from pathlib import Path
import tempfile
from src.schemas import ResearchResponse
from src.transcript import MarkdownTranscriptWriter, ToolStep


class TestTranscript(unittest.TestCase):
    def test_markdown_transcript_with_tool_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "transcript.md"
            writer = MarkdownTranscriptWriter(file_path, title="Test Lịch sử")

            structured = ResearchResponse(
                topic="Trí tuệ nhân tạo",
                summary="AI là một lĩnh vực của khoa học máy tính...",
                sources=["https://vi.wikipedia.org/wiki/Tri_tue_nhan_tao"],
                tools_used=["wikipedia"],
            )
            tool_steps = [
                ToolStep(
                    tool="wikipedia",
                    input="Trí tuệ nhân tạo",
                    output="Trí tuệ nhân tạo (AI) là trí tuệ được thể hiện bởi máy móc...",
                )
            ]

            writer.add_turn(
                question="AI là gì?",
                raw_output="json raw output",
                structured=structured,
                tool_calls=["wikipedia"],
                tool_steps=tool_steps,
            )

            content = file_path.read_text(encoding="utf-8")

            self.assertIn("# Test Lịch sử", content)
            self.assertIn("## Lượt 1", content)
            self.assertIn("**Câu hỏi:**", content)
            self.assertIn("AI là gì?", content)
            self.assertIn("**Chủ đề:** Trí tuệ nhân tạo", content)
            self.assertIn("**Tóm tắt:**", content)
            self.assertIn("**Nguồn tham khảo:**", content)
            self.assertIn("- https://vi.wikipedia.org/wiki/Tri_tue_nhan_tao", content)
            self.assertIn("**Kết quả tra cứu từ các nguồn:**", content)
            self.assertIn("- **Nguồn / Tool:** `wikipedia`", content)
            self.assertIn("- **Truy vấn:** `Trí tuệ nhân tạo`", content)
            self.assertIn(
                "Trí tuệ nhân tạo (AI) là trí tuệ được thể hiện bởi máy móc...", content
            )

    def test_markdown_transcript_resume(self):
        """Kiểm tra khi mở lại file transcript.md đã có sẵn lượt 1, writer sẽ ghi tiếp lượt 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "transcript.md"
            writer1 = MarkdownTranscriptWriter(file_path, title="Lịch sử nghiên cứu")
            writer1.add_turn(
                question="Câu hỏi 1",
                raw_output="raw 1",
                structured=None,
                tool_calls=[],
            )

            # Mở lại writer mới trên cùng file (giống như resume session)
            writer2 = MarkdownTranscriptWriter(file_path, title="Lịch sử nghiên cứu")
            writer2.add_turn(
                question="Câu hỏi 2",
                raw_output="raw 2",
                structured=None,
                tool_calls=[],
            )

            content = file_path.read_text(encoding="utf-8")
            self.assertIn("## Lượt 1", content)
            self.assertIn("Câu hỏi 1", content)
            self.assertIn("## Lượt 2", content)
            self.assertIn("Câu hỏi 2", content)


if __name__ == "__main__":
    unittest.main()
