# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ch20_mcu.py — Unit tests for Chapter 20 (MCU deployment).

Pins the fixed-point arithmetic (FP_ONE = 255), the discrete row-normalized
diffusion kernel of ``main.c``, ternary/4-bit packing, the PROGMEM C-header
round-trip, the Flash/SRAM accountant, and the firmware's adversarial probes.
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest

from gftac.ch20_mcu import (
    ATmega328P_FLASH,
    ATmega328P_SRAM,
    FP_ONE,
    MCUModel,
    decode_fp,
    encode_fp,
    export_c_header,
    flash_footprint_bytes,
    float_diffusion_step,
    float_forward_inference,
    forward_inference,
    inject_shortcut,
    isolation_activation,
    lattice_from_edges,
    lfsr_perturb,
    make_small_model,
    make_uno32_model,
    pack_4bit,
    pack_trits,
    parse_c_header,
    qtac_diffusion_step,
    quantize_4bit,
    sram_footprint_bytes,
    ternary_quantize,
    unpack_4bit,
    unpack_trits,
    UNO32_TEST_2HOP,
    UNO32_TEST_3HOP,
    UNO32_ADV_PROBES,
)


# ----------------------------------------------------------------------
# B.20.1 — fixed-point representation
# ----------------------------------------------------------------------

def test_fp_constants():
    assert FP_ONE == 255


def test_fp_roundtrip_all_codes():
    for q in range(FP_ONE + 1):
        assert encode_fp(decode_fp(q)) == q


def test_fp_error_bound():
    bound = 1.0 / (2 * FP_ONE)
    max_err = 0.0
    for k in range(25_501):
        x = k / 25_500.0
        max_err = max(max_err, abs(decode_fp(encode_fp(x)) - x))
    assert max_err <= bound + 1e-12
    # The bound is tight: it is attained (to floating-point precision).
    assert abs(max_err - bound) < 1e-15


def test_fp_clamps_out_of_range():
    assert encode_fp(-0.5) == 0
    assert encode_fp(2.0) == FP_ONE


# ----------------------------------------------------------------------
# the discrete diffusion kernel
# ----------------------------------------------------------------------

def test_diffusion_point_mass_advances_one_hop():
    W = lattice_from_edges(3, [(0, 1, 10)])
    assert qtac_diffusion_step(W, [255, 0, 0]) == [0, 255, 0]


def test_diffusion_even_split_truncates():
    # 255 * floor(10*255/20) // 255 = floor(255*127/255) = 127.
    W = lattice_from_edges(3, [(0, 1, 10), (0, 2, 10)])
    assert qtac_diffusion_step(W, [255, 0, 0]) == [0, 127, 127]


def test_diffusion_preserves_zero_without_outgoing():
    W = lattice_from_edges(2, [(0, 1, 10)])
    assert qtac_diffusion_step(W, [0, 255]) == [0, 0]


def test_transition_truncation_bound():
    # |P - P_hat| < 1/255 for every entry (Theorem B.20.2's per-step term).
    W = lattice_from_edges(4, [(0, 1, 3), (0, 2, 7), (1, 3, 5), (2, 3, 11)])
    for i in range(4):
        rs = sum(W[i])
        if rs == 0:
            continue
        for j in range(4):
            p = W[i][j] / rs
            p_hat = (W[i][j] * FP_ONE) // rs / FP_ONE
            assert 0.0 <= p - p_hat < 1.0 / FP_ONE


def test_accumulated_error_bound_after_k_steps():
    # Theorem B.20.2: ||h^(k) - h_hat^(k)||_inf <= k * ||h^(0)||_1 / 255.
    W = lattice_from_edges(6, [(0, 1, 4), (0, 2, 9), (1, 3, 7), (2, 4, 2), (3, 5, 8), (4, 5, 6)])
    h_int = [0] * 6
    h_int[0] = FP_ONE
    h_float = [0.0] * 6
    h_float[0] = float(FP_ONE)
    h0_norm = float(FP_ONE)
    for k in range(1, 9):
        h_int = qtac_diffusion_step(W, h_int)
        h_float = float_diffusion_step(W, h_float)
        err = max(abs(a - b) for a, b in zip(h_int, h_float))
        assert err <= k * h0_norm / FP_ONE + 1e-9


# ----------------------------------------------------------------------
# the exported UNO-32 model
# ----------------------------------------------------------------------

def test_uno32_model_structure():
    m = make_uno32_model()
    assert m.num_nodes == 32 and m.d_model == 16
    assert len(m.lattice) == 32 and len(m.lattice[0]) == 32
    nnz = sum(1 for row in m.lattice for v in row if v > 0)
    assert nnz == 24  # 24 arcs in the firmware lattice
    assert all(v in (-1, 0, 1) for row in m.w_in for v in row)
    assert all(v in (-1, 0, 1) for row in m.w_out for v in row)


def test_forward_matches_firmware_accuracy():
    m = make_uno32_model()
    p2 = sum(forward_inference(m, s, 2)[0] == t for s, t in UNO32_TEST_2HOP)
    p3 = sum(forward_inference(m, s, 3)[0] == t for s, t in UNO32_TEST_3HOP)
    assert p2 == 8  # 8/9: the symmetric 1-to-N branch of node 20 is a known tie
    assert p3 == 6  # 6/6
    # The documented failure is exactly the 20 -> 25 branch.
    assert forward_inference(m, 20, 2)[0] == 22


def test_fixed_and_float_agree_on_argmax():
    m = make_uno32_model()
    agree = 0
    for s, t in list(UNO32_TEST_2HOP) + list(UNO32_TEST_3HOP):
        tau = 2 if (s, t) in UNO32_TEST_2HOP else 3
        agree += int(forward_inference(m, s, tau)[0] == float_forward_inference(m, s, tau)[0])
    assert agree == 15


def test_fixed_float_activation_deviation_is_one_unit():
    m = make_uno32_model()
    max_dev = 0.0
    for s, _ in list(UNO32_TEST_2HOP) + list(UNO32_TEST_3HOP):
        tau = 2 if (s, _) in UNO32_TEST_2HOP else 3
        _, _, hf = forward_inference(m, s, tau)
        _, hl = float_forward_inference(m, s, tau)
        max_dev = max(max_dev, max(abs(a - b) for a, b in zip(hf, hl)))
    assert max_dev == 1.0


# ----------------------------------------------------------------------
# B.20.3 — packing
# ----------------------------------------------------------------------

def test_ternary_quantize_range_and_scale():
    trits, gamma = ternary_quantize([0.2, -0.05, 0.9, -1.1, 0.03, 0.44])
    assert set(trits) <= {-1, 0, 1}
    assert gamma > 0
    # Scale invariance: quantizing c*w gives the same trits for c > 0.
    trits2, _ = ternary_quantize([2.0 * v for v in [0.2, -0.05, 0.9, -1.1, 0.03, 0.44]])
    assert trits == trits2


def test_pack_trits_roundtrip_including_partial():
    for trits in ([1, 0, -1], [1, -1, 0, 1, -1, 1, 1, 0, -1, 1, 0, -1, -1, 1, 1, 1]):
        packed = pack_trits(trits)
        assert len(packed) == (len(trits) + 4) // 5
        assert unpack_trits(packed, len(trits)) == list(trits)


def test_pack_trits_rejects_bad_values():
    with pytest.raises(ValueError):
        pack_trits([0, 1, 2])


def test_4bit_roundtrip_including_odd_length():
    codes = [0, 1, 15, 8, 7, 3, 14, 9, 2]
    packed = pack_4bit(codes)
    assert len(packed) == (len(codes) + 1) // 2
    assert unpack_4bit(packed, len(codes)) == codes


def test_quantize_4bit_bounds():
    codes = quantize_4bit([-0.3, 0.0, 0.5, 1.0, 1.4], vmax=1.0)
    assert codes == [0, 0, 8, 15, 15]


# ----------------------------------------------------------------------
# C-header export / parse
# ----------------------------------------------------------------------

def test_export_parse_roundtrip(tmp_path):
    m = make_small_model()
    path = os.path.join(tmp_path, "mcu_weights.h")
    text = export_c_header(m, path=path)
    assert os.path.exists(path)
    parsed = parse_c_header(text)
    assert parsed["num_nodes"] == m.num_nodes
    assert parsed["d_model"] == m.d_model
    assert parsed["w_lattice"] == m.lattice
    assert parsed["w_in"] == m.w_in
    assert parsed["w_out"] == m.w_out


def test_export_uno32_roundtrip(tmp_path):
    m = make_uno32_model()
    text = export_c_header(m)
    parsed = parse_c_header(text)
    assert parsed["w_lattice"] == m.lattice
    assert parsed["w_in"] == m.w_in
    assert parsed["w_out"] == m.w_out


def test_export_contains_progmem_markers():
    text = export_c_header(make_small_model())
    assert "PROGMEM" in text
    assert "#include <avr/pgmspace.h>" in text
    assert "MCU_NUM_NODES 16" in text


# ----------------------------------------------------------------------
# footprint accountant
# ----------------------------------------------------------------------

def test_flash_footprint_totals():
    m = make_uno32_model()
    flash = flash_footprint_bytes(m, num_vectors=19, program_bytes=5760)
    assert flash["W_LATTICE"] == 1024
    assert flash["W_IN"] == 512
    assert flash["W_OUT"] == 512
    assert flash["test_vectors"] == 38
    assert flash["total"] == 7846
    assert flash["total"] < ATmega328P_FLASH


def test_sram_footprint_totals():
    m = make_uno32_model()
    sram = sram_footprint_bytes(m)
    assert sram == {"W_ram": 1024, "h_curr": 32, "h_next": 32, "h_latent": 32, "total": 1120}
    assert sram["total"] < ATmega328P_SRAM


# ----------------------------------------------------------------------
# adversarial probes
# ----------------------------------------------------------------------

def test_shortcut_injection_redirects_attractor():
    m = make_uno32_model()
    base_pred, _, _ = forward_inference(m, 0, 2)
    assert base_pred == 2
    attacked = inject_shortcut(m.lattice, 0, 29, 15)
    att_pred, att_conf, _ = forward_inference(m, 0, 1, lattice=attacked)
    assert att_pred == 29 and att_conf == 56
    # The original model is untouched.
    assert m.lattice[0][29] == 0


def test_lfsr_perturbation_matches_firmware():
    m = make_uno32_model()
    noisy, perturbed = lfsr_perturb(m.lattice)
    assert perturbed == 8
    # Each perturbation moves exactly one active cell by +/-1.
    changed = sum(
        1 for i in range(32) for j in range(32) if noisy[i][j] != m.lattice[i][j]
    )
    assert changed == 8
    # No cell leaves the non-negative range.
    assert all(v >= 0 for row in noisy for v in row)


def test_noise_preserves_rank_order():
    m = make_uno32_model()
    noisy, _ = lfsr_perturb(m.lattice)
    base = sum(forward_inference(m, s, 2)[0] == t for s, t in UNO32_TEST_2HOP)
    noisy_acc = sum(forward_inference(m, s, 2, lattice=noisy)[0] == t for s, t in UNO32_TEST_2HOP)
    assert base == noisy_acc == 8


def test_isolation_probes():
    m = make_uno32_model()
    acts = [isolation_activation(m, s, d, 3) for s, d in UNO32_ADV_PROBES]
    assert acts == [0, 0, 0, 2]
    assert sum(1 for a in acts if a == 0) == 3
