"""Main entry point for Graph-as-Code Arxiv node classification."""
import argparse
import os
import random

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

from src.data_loader import build_arxiv_dataframe, load_category_names
from src.evaluator import evaluate
from src.llm_client import LLMClient


def main():
    parser = argparse.ArgumentParser(description="Graph-as-Code on Arxiv")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--num-samples", type=int, default=None, help="Override number of test nodes")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--verbose", action="store_true", help="Print LLM reasoning per step")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    num_samples = args.num_samples or cfg["dataset"]["num_test_samples"]
    seed = args.seed if args.seed is not None else cfg["dataset"]["random_seed"]
    verbose = args.verbose or cfg.get("logging", {}).get("verbose", False)

    random.seed(seed)
    np.random.seed(seed)

    # 1. Load / build DataFrame
    df = build_arxiv_dataframe(cfg["dataset"]["node_csv"], cfg["dataset"]["processed_pkl"])
    print(f"[Run] DataFrame loaded: {len(df)} nodes")

    # 2. Identify test nodes and labels
    # OGB split: train/valid/test
    # We need to load split again to know which nodes are test nodes
    from src.data_loader import patch_torch_load
    patch_torch_load()
    from ogb.nodeproppred import NodePropPredDataset
    ogb_dataset = NodePropPredDataset(name="ogbn-arxiv")
    _, ogb_labels = ogb_dataset[0]
    split = ogb_dataset.get_idx_split()
    test_indices = split["test"]

    # Sample test nodes
    sampled_test = np.random.choice(test_indices, size=min(num_samples, len(test_indices)), replace=False)
    true_labels = {int(i): int(ogb_labels[i][0]) for i in sampled_test}
    print(f"[Run] Evaluating on {len(sampled_test)} test nodes (seed={seed})")

    # 3. Class labels mapping
    class_labels = load_category_names()
    print(f"[Run] {len(class_labels)} classes")

    # 4. Initialize LLM client
    llm = LLMClient(args.config)
    print(f"[Run] LLM: {llm.provider} / {llm.model}")

    # 5. Run evaluation
    results = evaluate(
        df=df,
        llm=llm,
        class_labels=class_labels,
        test_node_ids=[int(i) for i in sampled_test],
        true_labels=true_labels,
        log_dir=cfg.get("logging", {}).get("log_dir", "logs"),
        verbose=verbose,
    )

    print("\n" + "=" * 50)
    print(f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")
    print(f"Log file: {results['log_file']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
