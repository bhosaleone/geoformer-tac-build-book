# SPDX-License-Identifier: AGPL-3.0-only
"""
Chapter 21 — The Memristor Crossbar Energy Model (MODELED).

An analytical 1T1R Ohm/KCL model of the plastic conductance lattice executing a
two-stage vector-matrix multiplication in the analog domain. The module is a
*model*, not silicon: every number it emits is MODELED from published 1T1R
ReRAM parameters and an analytical peripheral-overhead factor.

Contents
--------
* ``CrossbarConfig``            — physical parameters (HRS/LRS, 4-bit states,
                                  read voltage, pulse width, wire resistance,
                                  drift exponent, peripheral factor).
* ``energy_law_nj``             — the analytical energy law of Theorem B.21.1.
* ``drift_factor``              — the conductance drift law of Theorem B.21.6.
* ``MemristorCrossbar``         — program / read / audit the array; its
                                  ``compute_dynamic_energy_per_inference`` is
                                  byte-for-byte the SDK's formula.
* ``generate_tier4_chains``     — the deterministic Tier-4 graph generator.
* ``NoiseResilienceSimulator``  — seeded Monte-Carlo routing under device noise.
* ``CrossbarEnergyAuditor``     — the same baselines as the SDK's
                                  ``NanojouleEnergyAuditor``.

Author: Atmabhan Pandit / Shrikant Bhosale (Eklavya Project)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CrossbarConfig",
    "energy_law_nj",
    "drift_factor",
    "MemristorCrossbar",
    "generate_tier4_chains",
    "NoiseResilienceSimulator",
    "CrossbarEnergyAuditor",
    "sensitivity_table",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CrossbarConfig:
    """Physical parameters of a 1T1R crossbar (all MODELED)."""

    n_inputs: int = 64
    n_outputs: int = 64
    g_min: float = 10e-6      # 10 uS  (High-Resistance State, HRS)
    g_max: float = 150e-6     # 150 uS (Low-Resistance State, LRS)
    b_levels: int = 16        # 4-bit quantization (16 discrete states)
    v_read: float = 0.2       # 200 mV read pulse
    t_pulse: float = 10e-9    # 10 ns pulse width
    r_wire: float = 3.2       # 3.2 Ohm metal wire resistance per cell
    drift_exp: float = 0.045  # conductance drift exponent nu
    peripheral_factor: float = 1.6   # 60% DAC/ADC/sample-and-hold overhead
    n_stages: int = 2         # two sequential MVM stages per 2-hop route

    @property
    def delta_g(self) -> float:
        """Conductance step between adjacent 4-bit levels (Siemens)."""
        return (self.g_max - self.g_min) / (self.b_levels - 1)


# ---------------------------------------------------------------------------
# The analytical laws (Theorems B.21.1 and B.21.4)
# ---------------------------------------------------------------------------

def energy_law_nj(
    total_conductance: float,
    v_read: float,
    t_pulse: float,
    n_stages: int = 2,
    peripheral_factor: float = 1.6,
) -> float:
    """
    Theorem B.21.1 — the dynamic energy law.

        E = n_stages * peripheral_factor * V_read^2 * (sum_ij G_ij) * t_pulse

    returned in nanojoules. One stage is the Joule heating of a read pulse of
    amplitude ``v_read`` and duration ``t_pulse`` across every cell: each cell
    dissipates V^2 * G_ij * t_pulse, and Ohm/KCL sum them.

    The arithmetic order is fixed to the SDK's so the two agree bit-for-bit.
    """
    e_stage = (v_read ** 2) * total_conductance * t_pulse  # Joules, one stage
    total_joules = (e_stage * n_stages) * peripheral_factor  # stages + overhead
    return total_joules * 1e9  # nJ


def drift_factor(t_seconds: float, t_0: float = 1.0, nu: float = 0.045) -> float:
    """
    Theorem B.21.6 — the conductance drift law G(t) = G_0 (t / t_0)^(-nu).

    Returns the dimensionless multiplier applied to every conductance. Time is
    clamped at ``t_0`` so that the factor never exceeds one.
    """
    return (max(1.0, t_seconds) / t_0) ** (-nu)


# ---------------------------------------------------------------------------
# The crossbar
# ---------------------------------------------------------------------------

class MemristorCrossbar:
    """
    Analytical model of a 2D passive 1T1R memristor crossbar holding the plastic
    conductance lattice W in the physical analog domain.

    All conductances are in Siemens, all energies in nanojoules unless the
    method name says otherwise.
    """

    def __init__(self, config: Optional[CrossbarConfig] = None):
        self.config = config or CrossbarConfig()
        c = self.config
        self.delta_g = c.delta_g
        self.G_ideal = np.full((c.n_outputs, c.n_inputs), c.g_min)
        self.G_actual = np.copy(self.G_ideal)

    # -- programming --------------------------------------------------------

    def quantize_levels(self, W_norm: np.ndarray) -> np.ndarray:
        """Map normalized conductance W in [0, 1] to the nearest of ``b_levels`` states."""
        c = self.config
        return np.clip(np.round(np.asarray(W_norm, dtype=float) * (c.b_levels - 1)),
                       0, c.b_levels - 1)

    def program_conductance_matrix(
        self,
        W_norm: np.ndarray,
        noise_std: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Program W into physical cells: 4-bit discretization followed by
        multiplicative cycle-to-cycle programming variation of relative scale
        ``noise_std``, clipped to the device window [0.5 g_min, 1.5 g_max].
        """
        c = self.config
        levels = self.quantize_levels(W_norm)
        self.G_ideal = c.g_min + levels * self.delta_g
        if noise_std <= 0.0:
            self.G_actual = np.copy(self.G_ideal)
            return
        if rng is None:
            rng = np.random
        variation = rng.normal(loc=0.0, scale=noise_std, size=self.G_ideal.shape)
        self.G_actual = np.clip(self.G_ideal * (1.0 + variation), c.g_min * 0.5, c.g_max * 1.5)

    # -- physical read ------------------------------------------------------

    def total_conductance(self) -> float:
        """sum_ij G_ij, the only array functional the energy depends on (Thm B.21.2)."""
        return float(np.sum(self.G_actual))

    def parasitic_attenuation(self) -> float:
        """
        First-order IR-drop attenuation of the distributed R-G ladder:
        1 / (1 + 0.5 * R_wire * N_in * mean(G)).
        """
        c = self.config
        return 1.0 / (1.0 + 0.5 * c.r_wire * c.n_inputs * float(np.mean(self.G_actual)))

    def compute_analog_vmm(
        self,
        v_in: np.ndarray,
        apply_parasitics: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Ohm's law and Kirchhoff's current law: I_out = G_actual @ v_in, with wire
        IR-drop attenuation and Johnson-Nyquist / shot read noise (0.5% of peak).
        """
        v_in = np.asarray(v_in, dtype=float)
        i_ideal = np.dot(self.G_actual, v_in)
        if not apply_parasitics:
            return i_ideal
        i_actual = i_ideal * self.parasitic_attenuation()
        if rng is None:
            rng = np.random
        read_noise = rng.normal(loc=0.0, scale=0.005 * np.max(i_actual + 1e-12),
                                size=i_actual.shape)
        return np.maximum(0.0, i_actual + read_noise)

    # -- drift --------------------------------------------------------------

    def apply_drift(self, t_seconds: float = 1000.0, t_0: float = 1.0) -> float:
        """Apply G <- G * (t / t_0)^(-nu); return the multiplier."""
        f = drift_factor(t_seconds, t_0, self.config.drift_exp)
        self.G_actual = self.G_actual * f
        return f

    # -- energy -------------------------------------------------------------

    def compute_dynamic_energy_per_inference(self) -> float:
        """Dynamic energy of one 2-hop inference, in nJ (SDK-identical formula)."""
        c = self.config
        return energy_law_nj(
            self.total_conductance(), c.v_read, c.t_pulse,
            c.n_stages, c.peripheral_factor,
        )

    def compute_dynamic_energy_pj(self) -> float:
        """The same figure in picojoules."""
        return self.compute_dynamic_energy_per_inference() * 1000.0


# ---------------------------------------------------------------------------
# Deterministic Tier-4 graph and Monte-Carlo noise resilience
# ---------------------------------------------------------------------------

def generate_tier4_chains(
    num_entities: int = 64,
    num_chains: int = 40,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Reproduce the Tier-4 benchmark's directed-chain generator exactly.

    Returns ``(one_hop, two_hop)`` where ``one_hop`` is the shuffled list of
    single-hop facts (source, target) shown in training and ``two_hop`` is the
    list of held-out (source, grandparent-target) queries. Chains are disjoint
    in their edges, so no fact is trained twice.
    """
    rng = random.Random(seed)
    chains: List[Tuple[int, int, int]] = []
    used_edges = set()
    attempts = 0
    while len(chains) < num_chains and attempts < 10000:
        attempts += 1
        a = rng.randint(0, num_entities - 1)
        b = rng.randint(0, num_entities - 1)
        c = rng.randint(0, num_entities - 1)
        if len({a, b, c}) == 3 and (a, b) not in used_edges and (b, c) not in used_edges:
            chains.append((a, b, c))
            used_edges.add((a, b))
            used_edges.add((b, c))

    one_hop: List[Tuple[int, int]] = []
    for a, b, c in chains:
        one_hop.append((a, b))
        one_hop.append((b, c))
    two_hop = [(a, c) for a, b, c in chains]
    rng.shuffle(one_hop)
    return one_hop, two_hop


class NoiseResilienceSimulator:
    """
    Monte-Carlo evaluation of analog 2-hop routing under programming noise.

    Definition of *accuracy* (pinned): for each held-out two-hop query
    (source -> target) we program the saturated-highway conductance matrix,
    run the two physical MVM stages of the emulator, and count a hit iff the
    argmax of the second-stage output currents equals the target. Accuracy is
    the mean hit rate over the query set, the five canonical seeds, and
    ``trials`` independent programming-noise realizations. The RNG stream for a
    (seed, trial) pair is ``default_rng(seed * 10007 + trial * 101 + 5)``; the
    constants are fixed stream-separation integers and the offset 5 is the
    experiment's declared stream offset.
    """

    SEEDS: Tuple[int, ...] = (42, 101, 2024, 777, 999)
    NOISE_LEVELS: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20)
    STREAM_OFFSET: int = 5

    def __init__(
        self,
        config: Optional[CrossbarConfig] = None,
        seeds: Sequence[int] = SEEDS,
        trials: int = 5,
        noise_levels: Sequence[float] = NOISE_LEVELS,
        num_entities: int = 64,
        num_chains: int = 40,
    ):
        self.config = config or CrossbarConfig(n_inputs=num_entities,
                                               n_outputs=num_entities)
        self.seeds = tuple(seeds)
        self.trials = int(trials)
        self.noise_levels = tuple(noise_levels)
        self.num_entities = int(num_entities)
        self.num_chains = int(num_chains)

    def build_highways(self, seed: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Saturated highway matrix W[target, source] = 1 and the held-out queries."""
        one_hop, two_hop = generate_tier4_chains(
            self.num_entities, self.num_chains, seed)
        W = np.zeros((self.num_entities, self.num_entities))
        for s, t in one_hop:
            W[t, s] = 1.0
        return W, two_hop

    def route_2hop(
        self,
        crossbar: MemristorCrossbar,
        src: int,
        rng: np.random.Generator,
    ) -> int:
        """Two physical VMM stages with analog sample-and-hold normalization."""
        c = crossbar.config
        v_in = np.zeros(c.n_inputs)
        v_in[src] = c.v_read
        i_hop1 = crossbar.compute_analog_vmm(v_in, rng=rng)
        m = float(np.max(i_hop1))
        if m > 1e-12:
            v_in2 = (i_hop1 / m) * c.v_read
        else:
            v_in2 = np.zeros(c.n_inputs)
        i_hop2 = crossbar.compute_analog_vmm(v_in2, rng=rng)
        return int(np.argmax(i_hop2))

    def run(self) -> List[Tuple[float, float, float]]:
        """Return ``(noise, mean_accuracy_pct, std_accuracy_pct)`` per noise level."""
        results: List[Tuple[float, float, float]] = []
        for noise in self.noise_levels:
            accs: List[float] = []
            for seed in self.seeds:
                W, two_hop = self.build_highways(seed)
                for trial in range(self.trials):
                    rng = np.random.default_rng(
                        seed * 10007 + trial * 101 + self.STREAM_OFFSET)
                    cb = MemristorCrossbar(self.config)
                    cb.program_conductance_matrix(W, noise_std=noise, rng=rng)
                    hits = sum(
                        self.route_2hop(cb, s, rng) == t for s, t in two_hop
                    )
                    accs.append(100.0 * hits / max(1, len(two_hop)))
            results.append((float(noise), float(np.mean(accs)), float(np.std(accs))))
        return results


# ---------------------------------------------------------------------------
# Energy auditor (same baselines as the SDK's NanojouleEnergyAuditor)
# ---------------------------------------------------------------------------

class CrossbarEnergyAuditor:
    """
    Compares the modeled crossbar energy against digital von Neumann baselines.

    The baseline table is identical to ``geoformer.hardware.energy``'s
    ``NanojouleEnergyAuditor.BASELINES``; the reduction ratio is
    ``baseline_nj / max(memristor_nj, 1e-12)``.
    """

    BASELINES: Dict[str, Dict[str, object]] = {
        "nvidia_gpu_fp32": {"name": "NVIDIA Server GPU (FP32)", "energy_nj": 4200.0, "precision": "FP32"},
        "nvidia_gpu_fp16": {"name": "NVIDIA Tensor Core GPU (FP16)", "energy_nj": 1100.0, "precision": "FP16"},
        "google_edge_tpu": {"name": "Google Edge TPU (INT8)", "energy_nj": 145.0, "precision": "INT8"},
        "arm_cortex_m55": {"name": "ARM Cortex-M55 NPU (INT8)", "energy_nj": 65.0, "precision": "INT8"},
    }

    @staticmethod
    def audit(crossbar: MemristorCrossbar) -> Dict[str, Dict[str, float]]:
        """Dynamic energy for the crossbar plus the reduction against every baseline."""
        memristor_nj = crossbar.compute_dynamic_energy_per_inference()
        memristor_pj = memristor_nj * 1000.0
        comparison: Dict[str, Dict[str, float]] = {}
        for key, base in CrossbarEnergyAuditor.BASELINES.items():
            base_nj = float(base["energy_nj"])
            comparison[key] = {
                "name": str(base["name"]),
                "baseline_nj": base_nj,
                "memristor_nj": memristor_nj,
                "memristor_pj": memristor_pj,
                "reduction_ratio": base_nj / max(memristor_nj, 1e-12),
            }
        return comparison


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_table(
    config: Optional[CrossbarConfig] = None,
    perturbations: Sequence[float] = (0.9, 1.1),
    W_norm_value: float = 0.5,
) -> List[Tuple[str, float, float, float, float]]:
    """
    Vary one physical parameter at a time and recompute the 443.9 pJ energy.

    Returns rows ``(parameter, perturbation, energy_nj, energy_pj, gpu_ratio)``
    for a uniform ``W_norm = W_norm_value`` program with no programming noise.
    """
    config = config or CrossbarConfig()
    rows: List[Tuple[str, float, float, float, float]] = []
    gpu_nj = float(CrossbarEnergyAuditor.BASELINES["nvidia_gpu_fp32"]["energy_nj"])

    def energy_for(cfg: CrossbarConfig) -> float:
        cb = MemristorCrossbar(cfg)
        cb.program_conductance_matrix(np.full((cfg.n_outputs, cfg.n_inputs), W_norm_value),
                                      noise_std=0.0)
        return cb.compute_dynamic_energy_per_inference()

    # The identity row: the unperturbed model.
    base_nj = energy_for(config)
    rows.append(("baseline", 1.0, base_nj, base_nj * 1000.0, gpu_nj / base_nj))

    for param in ("g_min", "g_max", "v_read", "t_pulse"):
        for p in perturbations:
            kwargs = dict(config.__dict__)
            kwargs[param] = getattr(config, param) * p
            cfg = CrossbarConfig(**kwargs)
            e_nj = energy_for(cfg)
            rows.append((param, float(p), e_nj, e_nj * 1000.0, gpu_nj / e_nj))
    return rows
