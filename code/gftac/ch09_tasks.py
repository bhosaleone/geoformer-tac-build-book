# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 9 — runnable tasks.

Each task is a small experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch09_tasks.py 3        # a single task
    python3 code/gftac/ch09_tasks.py all       # every task, in order

The code shown in each "Run this now" block of the chapter is the
verbatim body of the corresponding function, and every printed line is
pinned by tests/test_ch09_tasks.py.
"""

from __future__ import annotations

import io
import math
import os
import sys

import torch

# Allow standalone execution: python3 code/gftac/ch09_tasks.py N
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch09_model import (
    BookGeoFormer,
    BookGeoFormerBlock,
    block_packed_bytes,
    block_parameter_count,
    chain_laplacian,
    encoding_collision_rate,
    quasi_crystalline_encoding,
)

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]


def task_1() -> None:
    """§9.2 — B.9.1: Quasi-crystalline positional encoding & zero collisions."""
    torch.manual_seed(0)
    N, D = 2048, 64
    enc = quasi_crystalline_encoding(N, D, basis="golden")
    c_rate = encoding_collision_rate(enc, rtol=1e-5)
    row_norms = torch.norm(enc, dim=-1)
    print(f"Quasi-crystalline encoding: N = {N},  dim = {D}")
    print(f"Row norms: min = {float(row_norms.min()):.6f}, max = {float(row_norms.max()):.6f}")
    print(f"Collision rate (rtol=1e-5) on N={N}: {c_rate * 100.0:.2f}%")
    print(f"Zero collisions: {c_rate == 0.0}")


def task_2() -> None:
    """§9.3 — B.9.2: Single block parameter count and packed byte footprint."""
    d_model = 64
    block = BookGeoFormerBlock(d_model=d_model, n_heads=4, tau_max=3)
    measured_params = sum(p.numel() for p in block.parameters())
    pred_params = block_parameter_count(d_model)
    packed_b = block_packed_bytes(d_model)
    d_ff = math.floor(8.0 / 3.0 * d_model)
    print(f"d_model = {d_model},  d_ff = {d_ff}")
    print(f"Measured parameters: {measured_params:,}")
    print(f"Formula predicted:   {pred_params:,}  (match: {measured_params == pred_params})")
    print(f"Packed footprint:    {packed_b:,} B = {packed_b / 1024.0:.2f} KiB")


def task_3() -> None:
    """§9.3 — Full model parameter accounting and BitNet b1.58 footprint."""
    torch.manual_seed(0)
    model = BookGeoFormer(
        vocab_size=256,
        num_classes=10,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=512,
        tau_max=3,
        node_encoding="quasi",
    )
    total_p, packed_kb = model.count_parameters()
    print(f"GeoFormer stack: 2 layers, d_model=64, vocab=256, classes=10")
    print(f"Total model parameters:   {total_p:,}")
    print(f"BitNet b1.58 packed size: {packed_kb:.2f} KiB")
    print(f"FP32 uncompressed size:   {(total_p * 4.0) / 1024.0:.2f} KiB")
    print(f"Compression factor:       {4.0 / 0.20:.1f}x")


def task_4() -> None:
    """§9.4 — Forward pass shape, output finiteness, and logits."""
    torch.manual_seed(0)
    model = BookGeoFormer(
        vocab_size=256,
        num_classes=10,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=512,
        node_encoding="quasi",
    )
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        out = model(x)
    print(f"Input token tensor shape:  {list(x.shape)}")
    print(f"Output logits shape:       {list(out.shape)}")
    print(f"Output all finite:         {torch.isfinite(out).all().item()}")
    print(f"Mean logit: {float(out.mean()):.4f},  std logit: {float(out.std()):.4f}")


def task_5() -> None:
    """§9.4 — B.9.3: Backward gradient flow and state_dict roundtrip."""
    torch.manual_seed(0)
    model = BookGeoFormer(
        vocab_size=256,
        num_classes=10,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=512,
        node_encoding="quasi",
    )
    x = torch.randint(0, 256, (2, 16))
    out = model(x)
    loss = out.sum()
    loss_val = float(loss.detach())
    loss.backward()
    grad_norm = float(model.head.weight.grad.norm().detach())
    print(f"Loss backward executed: loss = {loss_val:.4f}")
    print(f"Head weight grad norm: {grad_norm:.4f}  (gradient flows: {grad_norm > 0})")

    # State dict save and load roundtrip
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    loaded_state = torch.load(buf, weights_only=True)
    model2 = BookGeoFormer(
        vocab_size=256,
        num_classes=10,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=512,
        node_encoding="quasi",
    )
    model2.load_state_dict(loaded_state)
    with torch.no_grad():
        out2 = model2(x)
    match = torch.allclose(out, out2, atol=1e-6)
    print(f"Checkpoint roundtrip match: {match}")


def task_6() -> None:
    """§9.5 — Bit-exact parity with production SDK GeoFormerBlock."""
    from geoformer.models.geoformer import GeoFormerBlock as SDKBlock

    torch.manual_seed(0)
    d_model = 32
    sdk_block = SDKBlock(d_model=d_model, n_heads=4, tau_max=2, lambda_geo=0.1, lambda_dir=0.05, dropout=0.0)
    for p in (sdk_block.attn.q_proj, sdk_block.attn.k_proj, sdk_block.attn.v_proj, sdk_block.attn.out_proj,
              sdk_block.ffn.w_gate, sdk_block.ffn.w_up, sdk_block.ffn.w_down):
        p.act_bits = None

    book_block = BookGeoFormerBlock(d_model=d_model, n_heads=4, tau_max=2, lambda_geo=0.1, lambda_dir=0.05, dropout=0.0)
    book_block.load_state_dict(sdk_block.state_dict())

    test_h = torch.randn(2, 6, d_model)
    L_test = chain_laplacian(6)
    idx = torch.arange(6, dtype=torch.float32)
    d_test = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))

    with torch.no_grad():
        out_sdk = sdk_block(test_h, d_matrix=d_test, L_matrix=L_test)
        out_bk = book_block(test_h, d_matrix=d_test, L_matrix=L_test)
    max_diff = float((out_bk - out_sdk).abs().max())
    print(f"max |h_book - h_sdk| = {max_diff:.3e}")
    print(f"Bit-exact equivalence: {max_diff == 0.0}")


TASKS = [task_1, task_2, task_3, task_4, task_5, task_6]


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for i, task in enumerate(TASKS, 1):
            print(f"--- Chapter 9, task {i} ---")
            task()
            print()
    else:
        TASKS[int(which) - 1]()
