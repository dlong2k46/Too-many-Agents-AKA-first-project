import unittest
from config import Settings
from pathlib import Path


class TestLangGraph(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            nvidia_api_key="fake-key-for-test",
            model_name="deepseek-ai/deepseek-v4.1-flash",
            model_base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.2,
            max_tokens=2048,
            max_iterations=5,
            max_chat_history_messages=10,
            output_dir=Path("runs"),
            log_level="INFO",
            langsmith_tracing=False,
        )

    def test_build_research_graph_nodes(self):
        from agent import build_research_graph, get_graph_ascii, get_graph_mermaid

        graph, parser = build_research_graph(self.settings)

        # Kiểm tra các node tồn tại trong đồ thị
        nodes = graph.get_graph().nodes
        self.assertIn("researcher", nodes)
        self.assertIn("tools", nodes)
        self.assertIn("critic_writer", nodes)

        # Kiểm tra sơ đồ ascii và mermaid
        ascii_art = get_graph_ascii(graph)
        self.assertIn("researcher", ascii_art)
        self.assertIn("critic_writer", ascii_art)

        mermaid_art = get_graph_mermaid(graph)
        self.assertIn("researcher", mermaid_art)
        self.assertIn("critic_writer", mermaid_art)


if __name__ == "__main__":
    unittest.main()
