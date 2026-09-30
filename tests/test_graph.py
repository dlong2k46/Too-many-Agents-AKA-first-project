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

    def test_format_critic_writer_context_includes_history_and_researcher(self):
        from agent import format_critic_writer_context
        from langchain_core.messages import AIMessage, HumanMessage

        chat_history = [
            HumanMessage(content="Xu hướng thời trang nữ 2026"),
            AIMessage(content="Xu hướng gồm phong cách Y2K và túi crochet."),
        ]
        query = "Gợi ý 1 bộ outfit đi chơi"
        researcher_msg = "Gợi ý phối áo baby tee với quần ống rộng và túi crochet Y2K."

        context = format_critic_writer_context(
            query=query,
            chat_history=chat_history,
            tool_steps=[],
            researcher_message=researcher_msg,
        )

        self.assertIn("Xu hướng thời trang nữ 2026", context)
        self.assertIn("Y2K và túi crochet", context)
        self.assertIn("Gợi ý 1 bộ outfit đi chơi", context)
        self.assertIn("áo baby tee với quần ống rộng", context)

    def test_extract_sources_from_chat_history(self):
        from agent import extract_sources_from_chat_history
        from langchain_core.messages import AIMessage, HumanMessage

        chat_history = [
            HumanMessage(content="Câu hỏi 1"),
            AIMessage(
                content='{"topic": "T1", "summary": "S1", "sources": ["https://example.com/trend1", "https://example.com/trend2"]}'
            ),
        ]

        sources = extract_sources_from_chat_history(chat_history)
        self.assertIn("https://example.com/trend1", sources)
        self.assertIn("https://example.com/trend2", sources)

    def test_node_workflow_visualization(self):
        from node import render_workflow

        ascii_art, mermaid_art = render_workflow(settings=self.settings)
        self.assertIn("researcher", ascii_art)
        self.assertIn("critic_writer", ascii_art)
        self.assertIn("researcher", mermaid_art)
        self.assertIn("critic_writer", mermaid_art)

    def test_should_continue_research(self):
        from agent import should_continue_research
        from langchain_core.messages import AIMessage

        msg_with_tool = AIMessage(
            content="",
            tool_calls=[{"name": "search_tool", "args": {"query": "test"}, "id": "1"}],
        )
        state_with_tool = {"messages": [msg_with_tool], "iteration": 1}
        self.assertEqual(
            should_continue_research(state_with_tool, max_iterations=5), "tools"
        )

        msg_no_tool = AIMessage(content="Kết quả")
        state_no_tool = {"messages": [msg_no_tool], "iteration": 1}
        self.assertEqual(
            should_continue_research(state_no_tool, max_iterations=5), "critic_writer"
        )

        state_exceeded = {"messages": [msg_with_tool], "iteration": 6}
        self.assertEqual(
            should_continue_research(state_exceeded, max_iterations=5), "critic_writer"
        )


if __name__ == "__main__":
    unittest.main()


