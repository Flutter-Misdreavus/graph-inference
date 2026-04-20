"""Load Arxiv node features from CSV and graph structure from OGB, build DataFrame."""
import os
import pickle

import pandas as pd
import numpy as np


def patch_torch_load():
    import torch
    _orig = torch.load
    def _patched(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _orig(*args, **kwargs)
    torch.load = _patched


def build_arxiv_dataframe(node_csv_path: str, output_pkl_path: str) -> pd.DataFrame:
    """Build the Graph-as-Code DataFrame and save to pickle.

    Columns:
        node_id   -- int index
        features  -- title + abstract text
        neighbors -- list of neighbor node IDs (undirected)
        label     -- int for train/val nodes, None for test nodes
    """
    if os.path.exists(output_pkl_path):
        print(f"[DataLoader] Loading cached DataFrame from {output_pkl_path}")
        return pd.read_pickle(output_pkl_path)

    print("[DataLoader] Reading node features from CSV ...")
    df_nodes = pd.read_csv(node_csv_path)

    patch_torch_load()
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset(name='ogbn-arxiv')
    graph, labels = dataset[0]
    split = dataset.get_idx_split()

    edge_index = graph['edge_index']
    num_nodes = graph['num_nodes']

    # Build undirected neighbor lists from edge_index
    print("[DataLoader] Building neighbor lists ...")
    neighbors = [[] for _ in range(num_nodes)]
    src = edge_index[0]
    dst = edge_index[1]
    for s, d in zip(src, dst):
        neighbors[s].append(int(d))
        neighbors[d].append(int(s))

    # Deduplicate and sort neighbors
    neighbors = [sorted(list(set(nbrs))) for nbrs in neighbors]

    # Determine label per node: train/val have labels, test has None
    labels = labels.flatten().astype(int)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[split['train']] = True
    val_mask[split['valid']] = True
    test_mask[split['test']] = True

    label_col = labels.copy().astype(object)
    label_col[test_mask] = None

    # Text features: title + abstract
    df_nodes['text'] = df_nodes['title'].fillna('') + '. ' + df_nodes['abstract'].fillna('')

    df = pd.DataFrame({
        'node_id': df_nodes['ID'].values,
        'features': df_nodes['text'].values,
        'neighbors': neighbors,
        'label': label_col,
    })

    os.makedirs(os.path.dirname(output_pkl_path) or '.', exist_ok=True)
    df.to_pickle(output_pkl_path)
    print(f"[DataLoader] Saved DataFrame to {output_pkl_path}")
    return df


def load_category_names() -> dict[int, str]:
    """Return mapping from label_id to arxiv category name."""
    # Derive from the CSV; each label_id corresponds to one category
    df_nodes = pd.read_csv('Datas/Arxiv.csv')
    mapping = dict(df_nodes[['label_id', 'category']].drop_duplicates().sort_values('label_id').values)
    return mapping
