# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 5 — Geodesic Highway Attention (Q-GHA).

Chapter 1 argued that the transformer's flaw is a *geometry*: its attention
scores are a Euclidean inner product, blind to the graph the data lives on.
Chapter 2 built that graph, Chapter 3 gave it a flow, Chapter 4 made it
carryable. This chapter puts the graph back into attention. The scores stay
a scaled dot product, but a term proportional to the graph distance is
subtracted from every logit, so attention is pulled along short — high
conductance — paths. The mechanism is called Geodesic Highway Attention:
the metric is the highway, and attention travels it.

Book theorems proved in the chapter and pinned by the tests:

  B.5.1  Softmax attention is a row-stochastic kernel. Every row of the
         attention map is a probability distribution, so attention is a
         Markov transition operator and its rows lie on the simplex.
  B.5.2  Concentration law. With logits S_ij - lambda * d_ij, the
         attention-weighted mean distance obeys d/dlambda E_a[d] =
         -Var_a(d) <= 0: raising lambda_geo monotonically moves attention
         onto shorter (higher-conductance) paths. As lambda -> infinity the
         attention concentrates on the minimum-distance column. (The
         entropy is *not* monotone in general; its exact derivative is
         dH/dlambda = Cov_a(d, S) - lambda Var_a(d), a boundary the chapter
         makes explicit rather than hides.)
  B.5.3  Directed-Finsler asymmetry. Adding an antisymmetric bias B to a
         base score S gives logit differences L_ij - L_ji = (S_ij - S_ji)
         + 2 B_ij; with a symmetric base S the difference is exactly 2 B_ij.
         Corollary (ratio law): A_ij / A_ji = (Z_j / Z_i) exp(2 B_ij), where
         Z_i = sum_k exp(L_ik). A directed highway therefore shifts the
         log-odds of the two directions by exactly twice the Finsler term,
         up to the row-normalizer correction — softmax of a symmetric
         matrix is not itself symmetric.

Run the demonstration ledger::

    python3 code/gftac/ch05_qgha.py
"""

from __future__ import annotations

import math
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow standalone execution: python3 code/gftac/ch05_qgha.py
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch04_ternary import TernaryLinear

__all__ = [
    "BookQGHA",
    "attention_entropy",
    "demo_ledger",
]


def attention_entropy(attn: torch.Tensor, eps: float = 1e-12) -> float:
    """Mean per-row Shannon entropy of an attention map, in nats.

    ``attn`` has shape ``[..., N, N]``; the entropy is taken over the last
    (key) axis and averaged over every leading axis. Rows are renormalized
    defensively so the value is well defined even after perturbation.
    """
    p = attn.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    return float((-(p * p.log()).sum(dim=-1)).mean())


class BookQGHA(nn.Module):
    """Multi-head geodesic highway attention with ternary projections.

    The forward pass is the standard scaled dot-product attention of a
    transformer, with one addition: before the softmax, the logits are
    penalized by ``lambda_geo * d_ij`` (and optionally biased by a directed
    Finsler term). The four projections are :class:`TernaryLinear` layers,
    so the module is a drop-in, trit-quantized attention.

        scores = Q K^T / sqrt(d_k) - lambda_geo * d + directed_finsler
        attn   = softmax(scores, dim=-1)
        out    = out_proj(attn @ V)

    Parameters
    ----------
    d_model : int
        Model width; must be divisible by ``n_heads``.
    n_heads : int
        Number of attention heads; each head has width ``d_model // n_heads``.
    lambda_geo : float
        Strength of the geodesic penalty. Zero recovers ordinary attention.
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        lambda_geo: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.lambda_geo = lambda_geo

        self.q_proj = TernaryLinear(d_model, d_model, bias=False)
        self.k_proj = TernaryLinear(d_model, d_model, bias=False)
        self.v_proj = TernaryLinear(d_model, d_model, bias=False)
        self.out_proj = TernaryLinear(d_model, d_model, bias=False)

    def scores(
        self,
        h: torch.Tensor,
        d_matrix: Optional[torch.Tensor] = None,
        directed_finsler: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pre-softmax logits, shape ``[B, n_heads, N, N]``."""
        B, N, D = h.shape
        q = self.q_proj(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if d_matrix is not None:
            logits = logits - self.lambda_geo * d_matrix.unsqueeze(0).unsqueeze(0)
        if directed_finsler is not None:
            logits = logits + directed_finsler.unsqueeze(0).unsqueeze(0)
        return logits

    def forward(
        self,
        h: torch.Tensor,
        d_matrix: Optional[torch.Tensor] = None,
        directed_finsler: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = h.shape
        logits = self.scores(h, d_matrix, directed_finsler)
        attn = F.softmax(logits, dim=-1)
        if attn.requires_grad:
            attn.retain_grad()
        v = self.v_proj(h).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(context), attn


# ----------------------------------------------------------------------
# Demonstration ledger (quoted in the chapter's Run section)
# ----------------------------------------------------------------------

def _chain_distance(n: int) -> torch.Tensor:
    """Graph distance |i - j| on a path of ``n`` nodes."""
    idx = torch.arange(n, dtype=torch.float32)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()


def demo_ledger() -> str:
    """The Chapter 5 demonstration: stochasticity, entropy, Finsler."""
    lines: List[str] = []
    lines.append("Chapter 5 — Geodesic Highway Attention, measured")
    lines.append("=" * 68)

    torch.manual_seed(0)
    d_model, n_heads, n = 16, 4, 6
    d = _chain_distance(n)
    h = torch.randn(1, n, d_model)

    # -- B.5.1: row-stochastic attention --------------------------------
    attn_mod = BookQGHA(d_model=d_model, n_heads=n_heads, lambda_geo=0.1)
    with torch.no_grad():
        out, attn = attn_mod(h, d_matrix=d)
    row_sums = attn.sum(dim=-1)
    lines.append("B.5.1 — softmax attention is row-stochastic")
    lines.append(f"  attn shape (B, heads, N, N): {tuple(attn.shape)}")
    lines.append(f"  min entry = {float(attn.min()):.3e}  (>= 0)")
    lines.append(
        f"  row sums: min {float(row_sums.min()):.10f}, "
        f"max {float(row_sums.max()):.10f}  (all 1)"
    )
    lines.append(
        f"  head 0, row 0 (first 6): "
        f"{[round(float(v), 4) for v in attn[0, 0, 0, :6]]}"
    )
    lines.append("")

    # -- B.5.2: expected distance vs lambda_geo --------------------------
    lines.append("B.5.2 — attention migrates onto shorter paths as lambda_geo rises")
    lines.append(f"  {'lambda_geo':>12}{'mean E[d]':>12}{'mean entropy':>16}{'row-2 argmax':>14}")
    prev_d = None
    mono = True
    for lam in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        mod = BookQGHA(d_model=d_model, n_heads=n_heads, lambda_geo=lam)
        mod.load_state_dict(attn_mod.state_dict())  # same weights for every lambda
        with torch.no_grad():
            _, a = mod(h, d_matrix=d)
        ed = float((a * d.unsqueeze(0).unsqueeze(0)).sum(-1).mean())
        ent = attention_entropy(a)
        argmax = int(a.mean(dim=(0, 1))[2].argmax())
        if prev_d is not None and ed > prev_d + 1e-9:
            mono = False
        prev_d = ed
        lines.append(f"  {lam:>12.2f}{ed:>12.6f}{ent:>16.6f}{argmax:>14}")
    lines.append(f"  E[d] non-increasing in lambda_geo: {mono}")
    lines.append("")

    # -- B.5.2: the concentration derivative dE[d]/dlambda = -Var_a(d) ----
    lam = 0.3
    mod = BookQGHA(d_model=d_model, n_heads=n_heads, lambda_geo=lam)
    mod.load_state_dict(attn_mod.state_dict())
    with torch.no_grad():
        _, a = mod(h, d_matrix=d)
        p = a.clamp_min(1e-12)
        dd = d.unsqueeze(0).unsqueeze(0)
        dmean = (p * dd).sum(-1)
        var_d = (p * (dd - dmean.unsqueeze(-1)) ** 2).sum(-1)
        H = -(p * p.log()).sum(-1)
        # base score s = logits + lambda*d  (recover the QK term)
        s = mod.scores(h)  # no d, no finsler -> pure QK
        cov = (p * (dd - dmean.unsqueeze(-1)) * (s - (p * s).sum(-1, keepdim=True))).sum(-1)
        analytic_ed = (-var_d).mean()
        analytic_h = (cov - lam * var_d).mean()
        eps = 1e-4
        mod2 = BookQGHA(d_model=d_model, n_heads=n_heads, lambda_geo=lam + eps)
        mod2.load_state_dict(attn_mod.state_dict())
        with torch.no_grad():
            _, a2 = mod2(h, d_matrix=d)
        ed1 = float((a * dd).sum(-1).mean())
        ed2 = float((a2 * dd).sum(-1).mean())
        fd_ed = (ed2 - ed1) / eps
        fd_h = (attention_entropy(a2) - attention_entropy(a)) / eps
    lines.append("B.5.2 — the concentration derivative identities")
    lines.append(f"  lambda_geo                       : {lam}")
    lines.append(f"  d E[d]/dlambda  = -Var_a(d)      : {float(analytic_ed):.6f}"
                 f"   (finite diff {fd_ed:.6f})")
    lines.append(f"  dH/dlambda = Cov_a(d,S)-l*Var_a(d): {float(analytic_h):.6f}"
                 f"   (finite diff {fd_h:.6f})")
    lines.append("")

    # -- B.5.2: the large-lambda limit -----------------------------------
    lines.append("B.5.2 — the lambda -> infinity limit (self-distance d_ii = 0)")
    mod = BookQGHA(d_model=d_model, n_heads=n_heads, lambda_geo=50.0)
    mod.load_state_dict(attn_mod.state_dict())
    with torch.no_grad():
        _, a = mod(h, d_matrix=d)
    row2 = a.mean(dim=(0, 1))[2]
    lines.append(f"  row 2 attention: {[round(float(v), 4) for v in row2]}")
    lines.append(
        f"  argmax col = {int(row2.argmax())}  |  argmin d[2,:] = {int(d[2].argmin())}"
        f" (itself, d_ii = 0)"
    )
    # mask the self-distance so the metric prefers neighbours instead
    d_off = d.clone()
    d_off.fill_diagonal_(1e6)
    mod = BookQGHA(d_model=d_model, n_heads=n_heads, lambda_geo=50.0)
    mod.load_state_dict(attn_mod.state_dict())
    with torch.no_grad():
        _, a_off = mod(h, d_matrix=d_off)
    row2_off = a_off.mean(dim=(0, 1))[2]
    lines.append(f"  self masked, row 2: {[round(float(v), 4) for v in row2_off]}")
    lines.append(
        f"  -> attends to columns {[int(c) for c in (row2_off > 1e-3).nonzero().flatten()]}"
        f" (the distance-1 neighbours)"
    )
    lines.append("")

    # -- B.5.3: directed Finsler asymmetry -------------------------------
    lines.append("B.5.3 — directed Finsler asymmetry")
    # antisymmetric bias: favour j -> i over i -> j along the chain
    B_dir = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            B_dir[i, j] = 0.5 * (j - i)  # antisymmetric
    mod = BookQGHA(d_model=d_model, n_heads=n_heads, lambda_geo=0.1)
    mod.load_state_dict(attn_mod.state_dict())
    with torch.no_grad():
        logits_b = mod.scores(h, d_matrix=d, directed_finsler=B_dir)
        logits_0 = mod.scores(h, d_matrix=d)
        _, a = mod(h, d_matrix=d, directed_finsler=B_dir)
        _, a0 = mod(h, d_matrix=d)
    # general identity: (L_ij - L_ji) - (S_ij - S_ji) = B_ij - B_ji = 2 B_ij
    lhs = (logits_b[0, 0, 0, 3] - logits_b[0, 0, 3, 0]) - \
          (logits_0[0, 0, 0, 3] - logits_0[0, 0, 3, 0])
    lines.append(f"  B antisymmetric: max|B + B^T| = "
                 f"{float((B_dir + B_dir.T).abs().max()):.1e}")
    lines.append(f"  logit asym shift (L_ij-L_ji) - (S_ij-S_ji) = {float(lhs):.6f}"
                 f"  (2 B_ij = {2.0 * B_dir[0, 3].item():.6f})")
    lines.append(
        f"  attention asym |A - A^T|: with Finsler "
        f"{float((a[0, 0] - a[0, 0].T).abs().max()):.4f}, "
        f"without {float((a0[0, 0] - a0[0, 0].T).abs().max()):.4f}"
    )
    lines.append("")

    # -- parity with the SDK ---------------------------------------------
    from geoformer.layers.attention import QuantizedGHA
    torch.manual_seed(1)
    sdk = QuantizedGHA(d_model=d_model, n_heads=n_heads, mode="ternary",
                       lambda_geo=0.1, act_bits=None)
    book = BookQGHA(d_model=d_model, n_heads=n_heads, lambda_geo=0.1)
    book.load_state_dict(sdk.state_dict())
    hh = torch.randn(2, n, d_model)
    with torch.no_grad():
        out_sdk, attn_sdk = sdk(hh, d_matrix=d)
        out_book, attn_book = book(hh, d_matrix=d)
    lines.append("Parity — production SDK QuantizedGHA (act_bits=None)")
    lines.append(
        f"  max |attn_book - attn_sdk|   = "
        f"{float((attn_book - attn_sdk).abs().max()):.3e}"
    )
    lines.append(
        f"  max |out_book  - out_sdk|    = "
        f"{float((out_book - out_sdk).abs().max()):.3e}"
    )
    lines.append("=" * 68)
    return "\n".join(lines)


if __name__ == "__main__":
    print(demo_ledger())
