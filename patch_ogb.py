import torch
_orig_load = torch.load

def patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)

torch.load = patched_load

import numpy as np
from ogb.nodeproppred import NodePropPredDataset

dataset = NodePropPredDataset(name='ogbn-arxiv')
graph, labels = dataset[0]
print('Edge shape:', graph['edge_index'].shape)
print('Num nodes:', graph['num_nodes'])
print('Num edges:', graph['edge_index'].shape[1])

split = dataset.get_idx_split()
print('Split sizes:', {k: len(v) for k, v in split.items()})
print('Labels shape:', labels.shape)
print('Label unique:', np.unique(labels).shape)
