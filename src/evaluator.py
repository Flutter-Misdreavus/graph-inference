"""Evaluation utilities for node classification accuracy."""
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .graph_as_code import GraphAsCodeAgent
from .llm_client import LLMClient


def evaluate(
    df: pd.DataFrame,
    llm: LLMClient,
    class_labels: dict[int, str],
    test_node_ids: list[int],
    true_labels: dict[int, int],
    log_dir: str = "logs",
    verbose: bool = False,
) -> dict:
    """Run Graph-as-Code on a batch of test nodes and compute accuracy.

    Returns a dict with accuracy, predictions, and per-node logs.
    """
    agent = GraphAsCodeAgent(df, llm, class_labels)
    correct = 0
    total = len(test_node_ids)
    predictions: dict[int, int | None] = {}
    node_logs: list[dict] = []

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.jsonl")

    for idx, node_id in enumerate(test_node_ids, start=1):
        true_label = true_labels[node_id]
        start_time = time.time()
        pred = agent.classify(node_id, verbose=verbose)
        elapsed = time.time() - start_time

        predictions[node_id] = pred
        is_correct = (pred == true_label) if pred is not None else False
        if is_correct:
            correct += 1

        log_entry = {
            "node_id": int(node_id),
            "true_label": int(true_label),
            "predicted_label": int(pred) if pred is not None else None,
            "correct": bool(is_correct),
            "elapsed_sec": round(elapsed, 2),
        }
        node_logs.append(log_entry)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(
            f"[{idx}/{total}] node={node_id} true={true_label} pred={pred} "
            f"{'[OK]' if is_correct else '[FAIL]'} ({elapsed:.1f}s) "
            f"acc={correct}/{idx}={correct/idx:.2%}"
        )

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
        "log_file": log_path,
    }
