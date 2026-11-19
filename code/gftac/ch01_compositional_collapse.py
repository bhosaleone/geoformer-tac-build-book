# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 1 — The Euclidean Bottleneck, measured.

The first experiment of the book. We build a tiny world of disjoint directed
chains (A -> B -> C -> D), train a flat (Euclidean) one-hop scorer to
perfection on the direct facts, and then ask it the compositional questions
it was never shown: "starting at A, where do you land after two (three)
steps?"

It fails — systematically, and *below chance* — because it echoes the direct
successor it memorized. Matrix-power propagation through the same world
answers every depth perfectly, with zero training. This is compositional
collapse in miniature, and it is the reason GeoFormer exists.

Run the ledger yourself::

    python -c "from gftac.ch01_compositional_collapse import run_demo, format_ledger; print(format_ledger(run_demo()))"
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

__all__ = [
    "build_chain_world",
    "transition_matrix",
    "OneHopMemorizer",
    "propagate_distribution",
    "propagate",
    "run_single_seed",
    "run_demo",
    "format_ledger",
]


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

def build_chain_world(
    num_chains: int = 3, chain_len: int = 4
) -> Tuple[int, List[Tuple[int, int]], List[List[int]]]:
    """Construct a world of disjoint directed chains.

    Entities are the integers ``0 .. num_chains*chain_len - 1``. Chain ``c``
    occupies the block ``[c*chain_len, (c+1)*chain_len)`` with edges

        base -> base+1 -> ... -> base+chain_len-1

    Returns ``(num_entities, edges, chains)``.
    """
    n = num_chains * chain_len
    chains: List[List[int]] = []
    edges: List[Tuple[int, int]] = []
    for c in range(num_chains):
        chain = [c * chain_len + k for k in range(chain_len)]
        chains.append(chain)
        edges.extend(zip(chain[:-1], chain[1:]))
    return n, edges, chains


def transition_matrix(num_entities: int, edges: List[Tuple[int, int]]) -> torch.Tensor:
    """Row-stochastic transition kernel ``P`` of the world.

    ``P[i, j]`` is the probability of stepping ``i -> j`` in one hop. Nodes
    with no outgoing edges (the chain tails) are *dead ends*: their row is
    all zeros. We keep them explicit throughout the book — hiding dead ends
    is exactly the "oracle" artifact the research program's Tier-8 audit
    forbids (we return to this in Chapter 12).
    """
    W = torch.zeros(num_entities, num_entities)
    for i, j in edges:
        W[i, j] = 1.0
    out = W.sum(dim=1, keepdim=True)
    P = torch.where(out > 0, W / out.clamp(min=1.0), torch.zeros_like(W))
    return P


# ---------------------------------------------------------------------------
# The flat (Euclidean) baseline
# ---------------------------------------------------------------------------

class OneHopMemorizer(nn.Module):
    """A flat scorer trained only on one-hop facts.

    Every entity ``i`` receives a point embedding ``e_i`` in ``R^d`` — a
    flat vector space with no notion of distance between *facts*. The score
    of the pair ``(s, t)`` is the standard bilinear link-prediction form

        score(s, t) = <e_s, e_t> / sqrt(d) + b_t                       (1.1)

    Training minimizes cross-entropy over all entities for each observed
    edge ``(s -> t)``. The model sees *only* the direct facts — exactly the
    supervision a language model gets from next-token data on relations.
    """

    def __init__(self, num_entities: int, dim: int = 32):
        super().__init__()
        self.num_entities = num_entities
        self.dim = dim
        self.entity = nn.Embedding(num_entities, dim)
        nn.init.normal_(self.entity.weight, std=1.0 / (dim ** 0.5))
        self.bias = nn.Parameter(torch.zeros(num_entities))

    def scores(self, src: torch.Tensor) -> torch.Tensor:
        """``[B]`` entity ids -> ``[B, N]`` next-entity scores."""
        e_src = self.entity(src)          # [B, d]
        e_all = self.entity.weight        # [N, d]
        return e_src @ e_all.T / (self.dim ** 0.5) + self.bias

    def train_on_edges(
        self, edges: List[Tuple[int, int]], epochs: int = 300, lr: float = 0.05
    ) -> List[float]:
        """Full-batch Adam on the one-hop facts; returns the loss curve."""
        src = torch.tensor([i for i, _ in edges], dtype=torch.long)
        tgt = torch.tensor([j for _, j in edges], dtype=torch.long)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        history: List[float] = []
        for _ in range(epochs):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(self.scores(src), tgt)
            loss.backward()
            opt.step()
            history.append(loss.item())
        return history

    def predict(self, src: int) -> int:
        """Argmax entity for the query ``src -> ?``."""
        with torch.no_grad():
            return int(self.scores(torch.tensor([src])).argmax(dim=-1).item())


# ---------------------------------------------------------------------------
# The geometric cure: propagation readout
# ---------------------------------------------------------------------------

def propagate_distribution(P: torch.Tensor, src: int, hops: int) -> torch.Tensor:
    """k-hop readout by matrix powers: ``z = e_src P^k``.

    No parameters, no training. The world's own kernel does the routing;
    the readout is the modal entity (argmax). This is the one-pass version
    of the operation GeoFormer performs inside its attention geometry.
    """
    z = torch.zeros(P.shape[0])
    z[src] = 1.0
    with torch.no_grad():
        for _ in range(hops):
            z = z @ P
    return z


def propagate(P: torch.Tensor, src: int, hops: int) -> int:
    """Modal entity after ``hops`` steps from ``src``."""
    return int(torch.argmax(propagate_distribution(P, src, hops)).item())


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def run_single_seed(
    seed: int,
    num_chains: int = 3,
    chain_len: int = 4,
    dim: int = 32,
    epochs: int = 300,
    lr: float = 0.05,
) -> Dict:
    """Train and evaluate one world with one seed; returns the raw ledger."""
    torch.manual_seed(seed)
    n, edges, chains = build_chain_world(num_chains, chain_len)
    P = transition_matrix(n, edges)

    model = OneHopMemorizer(n, dim)
    model.train_on_edges(edges, epochs=epochs, lr=lr)

    # (a) Control: did the flat model learn the one-hop facts at all?
    #     (It must — otherwise the collapse below is just underfitting.)
    one_hop_acc = sum(model.predict(s) == t for (s, t) in edges) / len(edges)

    # (b) Compositional (zero-shot) questions, one per chain per depth.
    #     The pair (start, target) was NEVER shown as a training fact.
    depths: Dict[int, Dict[str, float]] = {}
    for hops in (2, 3):
        hits_mem = echo = hits_prop = 0
        queries = 0
        for chain in chains:
            if len(chain) <= hops:
                continue
            start, target, successor = chain[0], chain[hops], chain[1]
            pred = model.predict(start)
            hits_mem += int(pred == target)      # correct compositional answer
            echo += int(pred == successor)       # echoes the direct successor
            hits_prop += int(propagate(P, start, hops) == target)
            queries += 1
        depths[hops] = {
            "hits_memorizer": hits_mem / queries,
            "echo_rate": echo / queries,
            "hits_propagation": hits_prop / queries,
            "queries": queries,
        }

    return {
        "seed": seed,
        "num_entities": n,
        "chance": 1.0 / n,
        "one_hop_train_acc": one_hop_acc,
        "depths": depths,
    }


def _mean_std(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def run_demo(seeds: Tuple[int, ...] = (0, 1, 2, 3, 4), **kwargs) -> Dict:
    """Run the Chapter 1 experiment over a seed set; returns the ledger.

    The seed set is part of the claim: every number the chapter quotes is a
    mean (and sample standard deviation) over these seeds, and the test
    suite re-derives them on your machine.
    """
    runs = [run_single_seed(s, **kwargs) for s in seeds]
    depth_keys = sorted(runs[0]["depths"].keys())
    summary = {
        "one_hop_train_acc": _mean_std([r["one_hop_train_acc"] for r in runs]),
        "chance": runs[0]["chance"],
        "depths": {
            k: {
                metric: _mean_std([r["depths"][k][metric] for r in runs])
                for metric in ("hits_memorizer", "echo_rate", "hits_propagation")
            }
            for k in depth_keys
        },
    }
    return {"seeds": list(seeds), "per_seed": runs, "summary": summary}


# ---------------------------------------------------------------------------
# Ledger rendering
# ---------------------------------------------------------------------------

def _row(label: str, stats: Dict[str, float]) -> str:
    return f"{label:<40}{stats['mean']:>10.4f}{stats['std']:>8.4f}"


def format_ledger(result: Dict) -> str:
    """Render the Chapter 1 ledger as the fixed-width table quoted in the text."""
    s = result["summary"]
    lines = [
        "Chapter 1 — The Measured Compositional Collapse",
        "=" * 64,
        f"world: 3 disjoint chains x 4 nodes ({result['per_seed'][0]['num_entities']} entities)"
        f" | seeds: {result['seeds']}",
        "",
        f"{'metric':<40}{'mean':>10}{'std':>8}",
        "-" * 64,
    ]
    lines.append(_row("1-hop train accuracy (flat model)", s["one_hop_train_acc"]))
    for depth in sorted(s["depths"]):
        d = s["depths"][depth]
        lines.append(_row(f"{depth}-hop Hits@1 (flat model)", d["hits_memorizer"]))
        lines.append(_row(f"{depth}-hop echo rate (flat model)", d["echo_rate"]))
        lines.append(_row(f"{depth}-hop Hits@1 (propagation)", d["hits_propagation"]))
    lines.append("-" * 64)
    lines.append(f"{'chance floor (1/N)':<40}{s['chance']:>10.4f}")
    lines.append("=" * 64)
    return "\n".join(lines)


if __name__ == "__main__":
    # Run the chapter's own experiment:  python3 code/gftac/ch01_compositional_collapse.py
    print(format_ledger(run_demo()))
