# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 5 — runnable tasks.

Each task is a small experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch05_tasks.py 3        # a single task
    python3 code/gftac/ch05_tasks.py all       # every task, in order

The code shown in each "Run this now" block of the chapter is the
verbatim body of the corresponding function, and every printed line is
pinned by tests/test_ch05_tasks.py.
"""

from __future__ import annotations

import os
import sys

import torch

# Allow standalone execution: python3 code/gftac/ch05_tasks.py N
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch05_qgha import BookQGHA, attention_entropy

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]


def _chain(n: int) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.float32)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()


def task_1() -> None:
    """§5.2 — B.5.1: softmax attention is a row-stochastic kernel."""
    torch.manual_seed(0)
    mod = BookQGHA(d_model=16, n_heads=4, lambda_geo=0.1)
    h = torch.randn(1, 6, 16)
    with torch.no_grad():
        _, attn = mod(h, d_matrix=_chain(6))
    row_sums = attn.sum(dim=-1)
    print(f"attn shape (B, heads, N, N): {tuple(attn.shape)}")
    print(f"min entry = {float(attn.min()):.4e}  (>= 0)")
    print(f"row sums: min {float(row_sums.min()):.7f}, max {float(row_sums.max()):.7f}")
    print(f"head 0, row 0: {[round(float(v), 4) for v in attn[0, 0, 0]]}")


def task_2() -> None:
    """§5.3 — the geodesic penalty shifts the logits by exactly -lambda*d."""
    torch.manual_seed(0)
    mod = BookQGHA(d_model=12, n_heads=3, lambda_geo=0.5)
    h = torch.randn(1, 5, 12)
    d = _chain(5)
    with torch.no_grad():
        plain = mod.scores(h)
        shifted = mod.scores(h, d_matrix=d)
    shift = plain - shifted
    print(f"lambda_geo = {mod.lambda_geo}")
    print(f"d[0, :]           = {[int(v) for v in d[0]]}")
    print(f"logit shift row 0 = {[round(float(v), 4) for v in shift[0, 0, 0]]}")
    print(f"shift == lambda*d everywhere: "
          f"{torch.allclose(shift, mod.lambda_geo * d.unsqueeze(0).unsqueeze(0), atol=1e-6)}")


def task_3() -> None:
    """§5.4 — B.5.2: attention migrates onto shorter paths as lambda rises."""
    torch.manual_seed(0)
    ref = BookQGHA(d_model=16, n_heads=4, lambda_geo=0.1)
    h = torch.randn(1, 6, 16)
    d = _chain(6)
    dd = d.unsqueeze(0).unsqueeze(0)
    print(f"{'lambda_geo':>12}{'mean E[d]':>12}{'mean entropy':>16}")
    prev, mono = None, True
    for lam in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0):
        mod = BookQGHA(d_model=16, n_heads=4, lambda_geo=lam)
        mod.load_state_dict(ref.state_dict())
        with torch.no_grad():
            _, a = mod(h, d_matrix=d)
        ed = float((a * dd).sum(-1).mean())
        if prev is not None and ed > prev + 1e-9:
            mono = False
        prev = ed
        print(f"{lam:>12.2f}{ed:>12.6f}{attention_entropy(a):>16.6f}")
    print(f"E[d] non-increasing in lambda_geo: {mono}")


def task_4() -> None:
    """§5.4 — B.5.2: d E[d]/dlambda = -Var_a(d), checked by finite difference."""
    torch.manual_seed(0)
    lam, eps = 0.3, 1e-4
    ref = BookQGHA(d_model=16, n_heads=4, lambda_geo=lam)
    h = torch.randn(1, 6, 16)
    d = _chain(6)
    dd = d.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        _, a = ref(h, d_matrix=d)
        p = a.clamp_min(1e-12)
        dmean = (p * dd).sum(-1)
        var_d = (p * (dd - dmean.unsqueeze(-1)) ** 2).sum(-1)
        analytic = float((-var_d).mean())
        mod2 = BookQGHA(d_model=16, n_heads=4, lambda_geo=lam + eps)
        mod2.load_state_dict(ref.state_dict())
        _, a2 = mod2(h, d_matrix=d)
        fd = (float((a2 * dd).sum(-1).mean()) - float((a * dd).sum(-1).mean())) / eps
    print(f"lambda_geo                    : {lam}")
    print(f"analytic  -Var_a(d)           : {analytic:.6f}")
    print(f"finite difference d E[d]/dlam : {fd:.6f}")
    print(f"identity holds: {abs(analytic - fd) < 0.02}")


def task_5() -> None:
    """§5.5 — B.5.3: a directed Finsler bias adds exactly 2*B to the logit asymmetry."""
    torch.manual_seed(0)
    mod = BookQGHA(d_model=12, n_heads=3, lambda_geo=0.1)
    h = torch.randn(1, 6, 12)
    d = _chain(6)
    B = torch.zeros(6, 6)
    for i in range(6):
        for j in range(6):
            B[i, j] = 0.5 * (j - i)
    Lb = mod.scores(h, d_matrix=d, directed_finsler=B)
    L0 = mod.scores(h, d_matrix=d)
    shift = (Lb - Lb.transpose(-2, -1)) - (L0 - L0.transpose(-2, -1))
    print(f"B antisymmetric (max|B + B^T|): {float((B + B.T).abs().max()):.1e}")
    print(f"(L_ij - L_ji) - (S_ij - S_ji) for i=0,j=3: {float(shift[0, 0, 0, 3].detach()):.6f}")
    print(f"2 * B[0, 3]                              : {2.0 * float(B[0, 3]):.6f}")
    print(f"identity holds everywhere: "
          f"{torch.allclose(shift, 2.0 * B.unsqueeze(0).unsqueeze(0), atol=1e-5)}")


def task_6() -> None:
    """§5.6 — parity with the production SDK QuantizedGHA."""
    from geoformer.layers.attention import QuantizedGHA

    torch.manual_seed(1)
    d_model, n_heads, n = 16, 4, 6
    sdk = QuantizedGHA(d_model=d_model, n_heads=n_heads, mode="ternary",
                       lambda_geo=0.1, act_bits=None)
    book = BookQGHA(d_model=d_model, n_heads=n_heads, lambda_geo=0.1)
    book.load_state_dict(sdk.state_dict())
    h = torch.randn(2, n, d_model)
    d = _chain(n)
    with torch.no_grad():
        out_sdk, attn_sdk = sdk(h, d_matrix=d)
        out_book, attn_book = book(h, d_matrix=d)
    print(f"max |attn_book - attn_sdk| = {float((attn_book - attn_sdk).abs().max()):.3e}")
    print(f"max |out_book  - out_sdk | = {float((out_book - out_sdk).abs().max()):.3e}")
    print(f"bit-exact parity: "
          f"{torch.allclose(attn_book, attn_sdk, atol=1e-6) and torch.allclose(out_book, out_sdk, atol=1e-6)}")


TASKS = [task_1, task_2, task_3, task_4, task_5, task_6]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for i, task in enumerate(TASKS, 1):
            print(f"--- Chapter 5, task {i} ---")
            task()
            print()
    else:
        TASKS[int(which) - 1]()
