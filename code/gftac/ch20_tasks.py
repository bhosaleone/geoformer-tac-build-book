# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 20 — runnable tasks.

Each task is a self-contained experiment placed at the exact point in the chapter
where its subject is developed. Run one from the book's root:

    python3 code/gftac/ch20_tasks.py 1        # a single task
    python3 code/gftac/ch20_tasks.py all      # every task, in order

The code shown in each "Run this now" block of the chapter is the verbatim body
of the corresponding function, and every printed line is pinned by
tests/test_ch20_tasks.py.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from gftac.ch20_mcu import (
    ATmega328P_FLASH,
    ATmega328P_SRAM,
    FP_ONE,
    UNO32_ADV_PROBES,
    UNO32_TEST_2HOP,
    UNO32_TEST_3HOP,
    decode_fp,
    encode_fp,
    export_c_header,
    flash_footprint_bytes,
    float_forward_inference,
    forward_inference,
    inject_shortcut,
    isolation_activation,
    lfsr_perturb,
    make_small_model,
    make_uno32_model,
    pack_4bit,
    pack_trits,
    parse_c_header,
    quantize_4bit,
    sram_footprint_bytes,
    ternary_quantize,
    unpack_4bit,
    unpack_trits,
)

__all__ = ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"]

_W = 38  # label column width used by every ledger line below


def task_1() -> None:
    """§20.3 — Fixed-point encode/decode round-trip and its exact error bound."""
    roundtrip_ok = 0
    for q in range(FP_ONE + 1):
        if encode_fp(decode_fp(q)) == q:
            roundtrip_ok += 1

    samples = 0
    max_err = 0.0
    for k in range(25_501):
        x = k / 25_500.0
        err = abs(decode_fp(encode_fp(x)) - x)
        if err > max_err:
            max_err = err
        samples += 1

    bound = 1.0 / (2 * FP_ONE)
    print(f"Fixed-Point Representation (FP_ONE = {FP_ONE}):")
    print(f"  • {'Codes in [0, 255]:':<{_W}}{FP_ONE + 1}")
    print(f"  • {'encode(decode(q)) == q for every q:':<{_W}}{roundtrip_ok}/{FP_ONE + 1}")
    print(f"  • {'Max |decode(encode(x)) - x|:':<{_W}}{max_err:.6e}")
    print(f"  • {'Theoretical bound 1/(2*255):':<{_W}}{bound:.6e}")
    print(f"  • {'Bound holds over ' + format(samples, ',') + ' samples:':<{_W}}{max_err <= bound + 1e-12}")


def task_2() -> None:
    """§20.4 — Ternary and 4-bit lattice quantization with byte packing."""
    rng = random.Random(20)
    tile = [[rng.uniform(-1.5, 1.5) for _ in range(8)] for _ in range(8)]
    flat = [v for row in tile for v in row]

    trits, gamma = ternary_quantize(flat)
    packed = pack_trits(trits)
    restored = unpack_trits(packed, len(trits))

    lo, hi = min(flat), max(flat)
    norm = [(v - lo) / (hi - lo) for v in flat]
    codes = quantize_4bit(norm, vmax=1.0)
    packed4 = pack_4bit(codes)
    restored4 = unpack_4bit(packed4, len(codes))

    print("Ternary (1.58-bit) Quantization of an 8x8 Weight Tile:")
    print(f"  • {'Parameters:':<{_W}}{len(flat)}")
    print(f"  • {'Scale gamma = mean|w|:':<{_W}}{gamma:.6f}")
    print(f"  • {'Quantized value set:':<{_W}}{sorted(set(trits))}")
    print(f"  • {'Trits packed 5-per-byte:':<{_W}}{len(trits)} -> {len(packed)} bytes")
    print(f"  • {'Bits per parameter:':<{_W}}{len(packed) * 8 / len(trits):.4f}")
    print(f"  • {'Ternary round-trip exact:':<{_W}}{restored == trits}")
    print("4-Bit Lattice Packing (2 codes per byte):")
    print(f"  • {'4-bit codes:':<{_W}}{len(codes)} -> {len(packed4)} bytes")
    print(f"  • {'Bits per parameter:':<{_W}}{len(packed4) * 8 / len(codes):.4f}")
    print(f"  • {'4-bit round-trip exact:':<{_W}}{restored4 == codes}")


def task_3() -> None:
    """§20.5 — Fixed-point vs float reference agreement on the exported vectors."""
    model = make_uno32_model()
    two_hop = list(UNO32_TEST_2HOP)
    three_hop = list(UNO32_TEST_3HOP)

    fp2 = sum(forward_inference(model, s, 2)[0] == t for s, t in two_hop)
    fp3 = sum(forward_inference(model, s, 3)[0] == t for s, t in three_hop)
    fl2 = sum(float_forward_inference(model, s, 2)[0] == t for s, t in two_hop)
    fl3 = sum(float_forward_inference(model, s, 3)[0] == t for s, t in three_hop)

    agree = 0
    max_dev = 0.0
    for s, t in two_hop:
        pred_fp, _, h_fp = forward_inference(model, s, 2)
        pred_fl, h_fl = float_forward_inference(model, s, 2)
        agree += int(pred_fp == pred_fl)
        max_dev = max(max_dev, max(abs(a - b) for a, b in zip(h_fp, h_fl)))
    for s, t in three_hop:
        pred_fp, _, h_fp = forward_inference(model, s, 3)
        pred_fl, h_fl = float_forward_inference(model, s, 3)
        agree += int(pred_fp == pred_fl)
        max_dev = max(max_dev, max(abs(a - b) for a, b in zip(h_fp, h_fl)))

    print("Fixed-Point vs Float Reference on the UNO-32 Model (15 exported vectors):")
    print(f"  • {'Fixed-point 2-hop accuracy:':<{_W}}{fp2}/{len(two_hop)}")
    print(f"  • {'Fixed-point 3-hop accuracy:':<{_W}}{fp3}/{len(three_hop)}")
    print(f"  • {'Float reference 2-hop accuracy:':<{_W}}{fl2}/{len(two_hop)}")
    print(f"  • {'Float reference 3-hop accuracy:':<{_W}}{fl3}/{len(three_hop)}")
    print(f"  • {'Argmax agreement (fixed vs float):':<{_W}}{agree}/{len(two_hop) + len(three_hop)}")
    print(f"  • {'Max |h_fixed - h_float| (0..255):':<{_W}}{max_dev:.1f}")


def task_4() -> None:
    """§20.6 — Flash and SRAM footprint of the 32-node deployment."""
    model = make_uno32_model()
    num_vectors = len(UNO32_TEST_2HOP) + len(UNO32_TEST_3HOP) + len(UNO32_ADV_PROBES)
    flash = flash_footprint_bytes(model, num_vectors=num_vectors, program_bytes=5760)
    sram = sram_footprint_bytes(model)

    flash_pct = 100.0 * flash["total"] / ATmega328P_FLASH
    sram_pct = 100.0 * sram["total"] / ATmega328P_SRAM
    margin = ATmega328P_SRAM - sram["total"]

    print(f"ATmega328P Flash Footprint (32 KB = {ATmega328P_FLASH:,} B):")
    print(f"  • {'W_LATTICE  (32x32, 1 B/cell):':<{_W}}{flash['W_LATTICE']:,} B")
    print(f"  • {'W_IN       (16x32, 1 B/trit):':<{_W}}{flash['W_IN']:,} B")
    print(f"  • {'W_OUT      (32x16, 1 B/trit):':<{_W}}{flash['W_OUT']:,} B")
    print(f"  • {f'Test vectors ({num_vectors} src/tgt pairs):':<{_W}}{flash['test_vectors']:,} B")
    print(f"  • {'Program code & UART routines:':<{_W}}{flash['program_code']:,} B")
    print(f"  • {'Total Flash:':<{_W}}{flash['total']:,} B ({flash_pct:.2f}% of {ATmega328P_FLASH:,} B)")
    print(f"ATmega328P SRAM Footprint (2 KB = {ATmega328P_SRAM:,} B):")
    print(f"  • {'W_ram      (32x32, 1 B/cell):':<{_W}}{sram['W_ram']:,} B")
    print(f"  • {'h_curr     (32, 1 B/node):':<{_W}}{sram['h_curr']:,} B")
    print(f"  • {'h_next     (32, 1 B/node):':<{_W}}{sram['h_next']:,} B")
    print(f"  • {'h_latent   (16, 2 B/dim):':<{_W}}{sram['h_latent']:,} B")
    print(f"  • {'Total SRAM:':<{_W}}{sram['total']:,} B ({sram_pct:.2f}% of {ATmega328P_SRAM:,} B)")
    print(f"  • {'Remaining stack margin:':<{_W}}{margin:,} B ({100.0 * margin / ATmega328P_SRAM:.2f}%)")


def task_5() -> None:
    """§20.7 — PROGMEM C-header export and parse round-trip self-check."""
    model = make_small_model()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mcu_weights.h")
        export_c_header(model, path=path)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    parsed = parse_c_header(text)

    ok = (
        parsed["num_nodes"] == model.num_nodes
        and parsed["d_model"] == model.d_model
        and parsed["w_lattice"] == model.lattice
        and parsed["w_in"] == model.w_in
        and parsed["w_out"] == model.w_out
    )
    first_row = ", ".join(str(v) for v in model.lattice[0])
    print("PROGMEM C-Header Export and Parse Round-Trip (16-node triage model):")
    print(f"  • {'Header size:':<{_W}}{len(text)} bytes")
    print(f"  • {'#define MCU_NUM_NODES:':<{_W}}{parsed['num_nodes']}")
    print(f"  • {'#define MCU_D_MODEL:':<{_W}}{parsed['d_model']}")
    print(f"  • {'W_LATTICE row 0:':<{_W}}{{ {first_row} }}")
    print(f"  • {'W_LATTICE round-trips exactly:':<{_W}}{parsed['w_lattice'] == model.lattice}")
    print(f"  • {'W_IN round-trips exactly:':<{_W}}{parsed['w_in'] == model.w_in}")
    print(f"  • {'W_OUT round-trips exactly:':<{_W}}{parsed['w_out'] == model.w_out}")
    print(f"  • {'Full export/parse round-trip:':<{_W}}{ok}")


def task_6() -> None:
    """§20.8 — Adversarial probes in the fixed-point simulator."""
    model = make_uno32_model()
    two_hop = list(UNO32_TEST_2HOP)
    base_acc = sum(forward_inference(model, s, 2)[0] == t for s, t in two_hop)

    base_pred, base_conf, _ = forward_inference(model, 0, 2)
    attacked = inject_shortcut(model.lattice, 0, 29, 15)
    att_pred, att_conf, _ = forward_inference(model, 0, 1, lattice=attacked)

    noisy, perturbed = lfsr_perturb(model.lattice)
    noise_acc = sum(
        forward_inference(model, s, 2, lattice=noisy)[0] == t for s, t in two_hop
    )

    probes = list(UNO32_ADV_PROBES)
    acts = [isolation_activation(model, s, d, 3) for s, d in probes]
    safe = sum(int(a == 0) for a in acts)

    print("Adversarial Probes in the Fixed-Point Simulator:")
    print("Adversary 1 — False Shortcut Injection:")
    print(f"  • {'Baseline query 0 (tau=2):':<{_W}}Node {base_pred} ({base_conf}% confidence)")
    print(f"  • {'Injected spurious edge:':<{_W}}0 ==> 29 (+15 conductance)")
    print(f"  • {'Post-attack query 0 (tau=1):':<{_W}}Node {att_pred} ({att_conf}% confidence)")
    print(f"  • {'Attractor warping confirmed:':<{_W}}{att_pred == 29}")
    print("Adversary 2 — 20% Analog Noise Perturbation:")
    print(f"  • {'Active conductances perturbed:':<{_W}}{perturbed}")
    print(f"  • {'Post-noise 2-hop accuracy:':<{_W}}{noise_acc}/{len(two_hop)}")
    print(f"  • {'Rank order preserved:':<{_W}}{noise_acc == base_acc}")
    print("Adversary 3 — Disconnected Cluster Isolation:")
    for (s, d), a in zip(probes, acts):
        tag = "SAFE" if a == 0 else "LEAK"
        print(f"  • Probe {s:>2} -> {d:>2} activation: {a:>3}/255 [{tag}]")
    print(f"  • {'Verified isolated:':<{_W}}{safe}/{len(probes)}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        print("Running all Chapter 20 tasks:\n")
        task_1()
        print("\n" + "-" * 60 + "\n")
        task_2()
        print("\n" + "-" * 60 + "\n")
        task_3()
        print("\n" + "-" * 60 + "\n")
        task_4()
        print("\n" + "-" * 60 + "\n")
        task_5()
        print("\n" + "-" * 60 + "\n")
        task_6()
    else:
        task_num = sys.argv[1]
        fn = globals().get(f"task_{task_num}")
        if fn is None:
            print(f"Unknown task: {task_num}")
            sys.exit(1)
        fn()


if __name__ == "__main__":
    main()
