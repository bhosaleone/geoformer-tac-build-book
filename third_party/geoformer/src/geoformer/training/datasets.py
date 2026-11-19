"""
Dataset Generators and Data Loaders for GeoFormer and TAC Training.
Includes:
  1. Transitive Multi-Hop Knowledge Graph Dataset (un-truncated with distractor side-loops)
  2. Graph Node Classification Dataset (relational manifold / citation style)
  3. Algorithmic Modular Grokking Sequence Dataset
"""

from typing import Dict, List, Tuple, Optional
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TransitiveKnowledgeGraphDataset(Dataset):
    """
    Knowledge Graph Dataset for Zero-Shot Multi-Hop Transitive Reasoning.
    Trained strictly on 1-hop facts; evaluated on multi-hop chains (2 to 32 hops).
    """
    def __init__(
        self,
        num_entities: int = 256,
        num_chains: int = 120,
        max_hops: int = 32,
        seed: int = 42,
        alpha_0: float = 2.5,
        split: str = "train",
        eval_depth: Optional[int] = None,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_chains = num_chains
        self.max_hops = max_hops
        self.seed = seed
        self.alpha_0 = alpha_0
        self.split = split
        self.eval_depth = eval_depth

        self.samples, self.ground_truth_W = self._generate_data()

    def _generate_data(self) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        W = torch.zeros(self.num_entities, self.num_entities)
        w_decoy = 1.0 / self.alpha_0

        # Construct primary reasoning chains and side-loops on un-truncated graph
        chain_nodes = []
        used_nodes = set()
        
        # Build multi-hop chains
        chain_len = self.max_hops + 16  # Active outgoing edges past max_hops
        
        for c in range(self.num_chains):
            # Select chain of distinct nodes if possible, or staggered
            start_idx = (c * 7) % max(1, self.num_entities - chain_len)
            chain = [(start_idx + i) % self.num_entities for i in range(chain_len)]
            chain_nodes.append(chain)

            for i in range(len(chain) - 1):
                u = chain[i]
                v = chain[i + 1]
                # True geodesic edge (weight ~ 1.0)
                w_true = 1.0 + (random.random() - 0.5) * 0.04
                W[u, v] = max(W[u, v].item(), w_true)

                # Controlled decoy side-loops (d1 <-> d2)
                d1 = (self.num_entities // 2 + (u * 3)) % self.num_entities
                d2 = (self.num_entities // 2 + (u * 3 + 1)) % self.num_entities
                if d1 != u and d1 != v and d2 != u and d2 != v:
                    W[u, d1] = max(W[u, d1].item(), w_decoy)
                    W[u, d2] = max(W[u, d2].item(), w_decoy)
                    W[d1, d2] = max(W[d1, d2].item(), 0.5)
                    W[d2, d1] = max(W[d2, d1].item(), 0.5)

        # Assemble samples based on split
        samples = []
        if self.split == "train":
            # 1-hop facts only: (src, rel=0, tgt)
            seen_edges = set()
            for chain in chain_nodes:
                for i in range(len(chain) - 1):
                    u = chain[i]
                    v = chain[i + 1]
                    if (u, v) not in seen_edges:
                        seen_edges.add((u, v))
                        samples.append({
                            "src": u,
                            "rel": 0,
                            "tgt": v,
                            "hops": 1,
                        })
            random.shuffle(samples)
        else:
            # Multi-hop zero-shot evaluation split: (chain[0], rel=depth-1, chain[depth])
            depth = self.eval_depth or 4
            for chain in chain_nodes:
                if len(chain) > depth:
                    src = chain[0]
                    tgt = chain[depth]
                    samples.append({
                        "src": src,
                        "rel": min(7, depth - 1),
                        "tgt": tgt,
                        "hops": depth,
                    })

        return samples, W

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        return {
            "src": torch.tensor(item["src"], dtype=torch.long),
            "rel": torch.tensor(item["rel"], dtype=torch.long),
            "tgt": torch.tensor(item["tgt"], dtype=torch.long),
            "hops": torch.tensor(item["hops"], dtype=torch.long),
        }


class GraphNodeClassificationDataset(Dataset):
    """
    Synthetic Graph Manifold & Relational Citation Dataset.
    """
    def __init__(self, num_nodes: int = 500, in_features: int = 64, num_classes: int = 7, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.num_nodes = num_nodes
        self.in_features = in_features
        self.num_classes = num_classes

        # Class centers in feature space
        centers = torch.randn(num_classes, in_features)
        self.labels = torch.randint(0, num_classes, (num_nodes,))
        self.features = centers[self.labels] + 0.5 * torch.randn(num_nodes, in_features)

        # Adjacency matrix: higher intra-class edge probability
        A = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                same_class = (self.labels[i] == self.labels[j])
                prob = 0.08 if same_class else 0.005
                if random.random() < prob:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
        self.adj = A

        # Split masks (60% train, 20% val, 20% test)
        indices = torch.randperm(num_nodes)
        n_train = int(0.6 * num_nodes)
        n_val = int(0.2 * num_nodes)

        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        self.train_mask[indices[:n_train]] = True
        self.val_mask[indices[n_train:n_train + n_val]] = True
        self.test_mask[indices[n_train + n_val:]] = True

    def __len__(self) -> int:
        return self.num_nodes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "node_idx": torch.tensor(idx, dtype=torch.long),
            "features": self.features[idx],
            "label": self.labels[idx],
        }


def create_transitive_dataloaders(
    num_entities: int = 256,
    num_chains: int = 120,
    max_hops: int = 32,
    batch_size: int = 32,
    seed: int = 42,
    eval_depths: Optional[List[int]] = None,
) -> Tuple[DataLoader, Dict[int, DataLoader], torch.Tensor]:
    """
    Creates train DataLoader and zero-shot multi-hop test DataLoaders for each depth.
    """
    if eval_depths is None:
        eval_depths = [2, 4, 8, 16, 24, 32]

    train_dataset = TransitiveKnowledgeGraphDataset(
        num_entities=num_entities,
        num_chains=num_chains,
        max_hops=max_hops,
        seed=seed,
        split="train"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loaders = {}
    for depth in eval_depths:
        test_dataset = TransitiveKnowledgeGraphDataset(
            num_entities=num_entities,
            num_chains=num_chains,
            max_hops=max_hops,
            seed=seed + 1000,
            split="test",
            eval_depth=depth
        )
        test_loaders[depth] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loaders, train_dataset.ground_truth_W
