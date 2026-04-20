"""Dry-run integration test for Graph-as-Code without real LLM calls."""
import pandas as pd
from src.data_loader import build_arxiv_dataframe, load_category_names
from src.graph_as_code import GraphAsCodeAgent


class MockLLM:
    """Fake LLM that returns canned responses to exercise the agent loop."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.idx = 0
        self.provider = "mock"
        self.model = "mock"

    def chat(self, system_prompt: str, user_message: str, history=None) -> str:
        resp = self.responses[self.idx]
        self.idx += 1
        return resp


def test_full_loop():
    df = build_arxiv_dataframe("Datas/Arxiv.csv", "Datas/arxiv_processed.pkl")
    classes = load_category_names()
    print(f"Loaded {len(df)} nodes, {len(classes)} classes")

    # Pick a train node so we know the true label
    node_id = 0
    true_label = df.loc[node_id, 'label']
    nbrs = df.loc[node_id, 'neighbors']
    print(f"Node {node_id}: label={true_label}, neighbors={len(nbrs)}")

    # Scenario 1: LLM queries neighbors then answers correctly
    responses = [
        f"Let me check the neighbors.\ndf.loc[{node_id}, 'neighbors']",
        f"The node has {len(nbrs)} neighbors. Let me check their labels.\ndf.loc[df.loc[{node_id}, 'neighbors'], 'label'].value_counts().idxmax()",
        f"Most neighbors have label {true_label}.\nAnswer: {true_label}",
    ]
    agent = GraphAsCodeAgent(df, MockLLM(responses), classes)
    pred = agent.classify(node_id, verbose=True)
    print(f"\nPredicted: {pred}, True: {true_label}, Match: {pred == true_label}")
    assert pred == true_label, "Mock test should predict correctly"

    # Scenario 2: LLM answers directly
    responses2 = [f"I recognize this paper.\nAnswer: {true_label}"]
    agent2 = GraphAsCodeAgent(df, MockLLM(responses2), classes)
    pred2 = agent2.classify(node_id)
    assert pred2 == true_label, "Direct answer should work"

    print("\nAll mock tests passed.")


if __name__ == "__main__":
    test_full_loop()
