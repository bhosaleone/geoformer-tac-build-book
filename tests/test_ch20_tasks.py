# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch20_tasks.py — Pins every printed line of every Chapter 20 task.

Each test runs the task, captures stdout, and compares it byte-for-byte against
the exact transcript shown in the chapter. No line can drift.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from gftac import ch20_tasks


def _capture(fn) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue().rstrip("\n")

EXPECTED_TASK_1 = (
    'Fixed-Point Representation (FP_ONE = 255):'
    "\n"
    '  • Codes in [0, 255]:                    256'
    "\n"
    '  • encode(decode(q)) == q for every q:   256/256'
    "\n"
    '  • Max |decode(encode(x)) - x|:          1.960784e-03'
    "\n"
    '  • Theoretical bound 1/(2*255):          1.960784e-03'
    "\n"
    '  • Bound holds over 25,501 samples:      True'
)


def test_task_1():
    assert _capture(ch20_tasks.task_1) == EXPECTED_TASK_1


EXPECTED_TASK_2 = (
    'Ternary (1.58-bit) Quantization of an 8x8 Weight Tile:'
    "\n"
    '  • Parameters:                           64'
    "\n"
    '  • Scale gamma = mean|w|:                0.776204'
    "\n"
    '  • Quantized value set:                  [-1, 0, 1]'
    "\n"
    '  • Trits packed 5-per-byte:              64 -> 13 bytes'
    "\n"
    '  • Bits per parameter:                   1.6250'
    "\n"
    '  • Ternary round-trip exact:             True'
    "\n"
    '4-Bit Lattice Packing (2 codes per byte):'
    "\n"
    '  • 4-bit codes:                          64 -> 32 bytes'
    "\n"
    '  • Bits per parameter:                   4.0000'
    "\n"
    '  • 4-bit round-trip exact:               True'
)


def test_task_2():
    assert _capture(ch20_tasks.task_2) == EXPECTED_TASK_2


EXPECTED_TASK_3 = (
    'Fixed-Point vs Float Reference on the UNO-32 Model (15 exported vectors):'
    "\n"
    '  • Fixed-point 2-hop accuracy:           8/9'
    "\n"
    '  • Fixed-point 3-hop accuracy:           6/6'
    "\n"
    '  • Float reference 2-hop accuracy:       8/9'
    "\n"
    '  • Float reference 3-hop accuracy:       6/6'
    "\n"
    '  • Argmax agreement (fixed vs float):    15/15'
    "\n"
    '  • Max |h_fixed - h_float| (0..255):     1.0'
)


def test_task_3():
    assert _capture(ch20_tasks.task_3) == EXPECTED_TASK_3


EXPECTED_TASK_4 = (
    'ATmega328P Flash Footprint (32 KB = 32,768 B):'
    "\n"
    '  • W_LATTICE  (32x32, 1 B/cell):         1,024 B'
    "\n"
    '  • W_IN       (16x32, 1 B/trit):         512 B'
    "\n"
    '  • W_OUT      (32x16, 1 B/trit):         512 B'
    "\n"
    '  • Test vectors (19 src/tgt pairs):      38 B'
    "\n"
    '  • Program code & UART routines:         5,760 B'
    "\n"
    '  • Total Flash:                          7,846 B (23.94% of 32,768 B)'
    "\n"
    'ATmega328P SRAM Footprint (2 KB = 2,048 B):'
    "\n"
    '  • W_ram      (32x32, 1 B/cell):         1,024 B'
    "\n"
    '  • h_curr     (32, 1 B/node):            32 B'
    "\n"
    '  • h_next     (32, 1 B/node):            32 B'
    "\n"
    '  • h_latent   (16, 2 B/dim):             32 B'
    "\n"
    '  • Total SRAM:                           1,120 B (54.69% of 2,048 B)'
    "\n"
    '  • Remaining stack margin:               928 B (45.31%)'
)


def test_task_4():
    assert _capture(ch20_tasks.task_4) == EXPECTED_TASK_4


EXPECTED_TASK_5 = (
    'PROGMEM C-Header Export and Parse Round-Trip (16-node triage model):'
    "\n"
    '  • Header size:                          3004 bytes'
    "\n"
    '  • #define MCU_NUM_NODES:                16'
    "\n"
    '  • #define MCU_D_MODEL:                  8'
    "\n"
    '  • W_LATTICE row 0:                      { 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }'
    "\n"
    '  • W_LATTICE round-trips exactly:        True'
    "\n"
    '  • W_IN round-trips exactly:             True'
    "\n"
    '  • W_OUT round-trips exactly:            True'
    "\n"
    '  • Full export/parse round-trip:         True'
)


def test_task_5():
    assert _capture(ch20_tasks.task_5) == EXPECTED_TASK_5


EXPECTED_TASK_6 = (
    'Adversarial Probes in the Fixed-Point Simulator:'
    "\n"
    'Adversary 1 — False Shortcut Injection:'
    "\n"
    '  • Baseline query 0 (tau=2):             Node 2 (48% confidence)'
    "\n"
    '  • Injected spurious edge:               0 ==> 29 (+15 conductance)'
    "\n"
    '  • Post-attack query 0 (tau=1):          Node 29 (56% confidence)'
    "\n"
    '  • Attractor warping confirmed:          True'
    "\n"
    'Adversary 2 — 20% Analog Noise Perturbation:'
    "\n"
    '  • Active conductances perturbed:        8'
    "\n"
    '  • Post-noise 2-hop accuracy:            8/9'
    "\n"
    '  • Rank order preserved:                 True'
    "\n"
    'Adversary 3 — Disconnected Cluster Isolation:'
    "\n"
    '  • Probe  0 ->  9 activation:   0/255 [SAFE]'
    "\n"
    '  • Probe  0 -> 22 activation:   0/255 [SAFE]'
    "\n"
    '  • Probe 20 ->  3 activation:   0/255 [SAFE]'
    "\n"
    '  • Probe 26 -> 15 activation:   2/255 [LEAK]'
    "\n"
    '  • Verified isolated:                    3/4'
)


def test_task_6():
    assert _capture(ch20_tasks.task_6) == EXPECTED_TASK_6
