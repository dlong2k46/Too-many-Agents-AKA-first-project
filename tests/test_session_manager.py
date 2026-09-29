import unittest
from pathlib import Path
import tempfile
from langchain_core.messages import AIMessage, HumanMessage

from session_manager import SessionManager


class TestSessionManager(unittest.TestCase):
    def test_save_and_load_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "test-session"
            session_dir.mkdir()

            chat_history = [
                HumanMessage(content="AI là gì?"),
                AIMessage(content='{"topic": "AI", "summary": "Trí tuệ nhân tạo", "sources": []}'),
            ]
            metadata = {
                "last_topic": "AI",
                "turns": 1,
            }

            SessionManager.save_session_state(session_dir, chat_history, metadata)

            loaded_history, loaded_meta = SessionManager.load_session_state(session_dir)

            self.assertEqual(len(loaded_history), 2)
            self.assertIsInstance(loaded_history[0], HumanMessage)
            self.assertEqual(loaded_history[0].content, "AI là gì?")
            self.assertIsInstance(loaded_history[1], AIMessage)
            self.assertEqual(loaded_meta.get("last_topic"), "AI")
            self.assertEqual(loaded_meta.get("turns"), 1)

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            s1 = base_dir / "20260929-100000-aaa111"
            s1.mkdir()
            SessionManager.save_session_state(
                s1,
                [HumanMessage(content="Câu hỏi 1")],
                {"last_topic": "Chủ đề 1", "turns": 1},
            )

            s2 = base_dir / "20260929-110000-bbb222"
            s2.mkdir()
            SessionManager.save_session_state(
                s2,
                [HumanMessage(content="Câu hỏi 2")],
                {"last_topic": "Chủ đề 2", "turns": 1},
            )

            sessions = SessionManager.list_sessions(base_dir)
            self.assertEqual(len(sessions), 2)
            # Sắp xếp mới nhất trước
            self.assertEqual(sessions[0]["session_id"], "20260929-110000-bbb222")
            self.assertEqual(sessions[0]["last_topic"], "Chủ đề 2")


if __name__ == "__main__":
    unittest.main()
