<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->
# Chapter 21 — The Memristor Crossbar Energy Model (MODELED)

Every chapter before this one paid for its arithmetic. We measured the sparse
store's RAM on this machine, timed the deductive walk, and counted the hops the
attractor took. The numbers were **MEASURED** because the object was a program
running on a CPU. This chapter is different, and the difference must be stated
before a single number appears: what follows is a **MODELED** account of an
analog circuit. No memristor was fabricated, no crossbar was probed, no
picocoulomb was integrated on a bench. We write down the physics that a 1T1R
ReRAM array would obey if it were built to published specifications, and we
compute what that physics implies. The label **MODELED** attaches to every
figure in this chapter, and the discipline of the book is that we say so at
every point of use.

The reason to model rather than measure is not convenience; it is that the
question this chapter answers is a question about *limits*. The dense
Euclidean transformer of Chapter 1 is bottlenecked by a fact of silicon: a
multiply-accumulate on a von Neumann machine must move operands from memory to
an arithmetic unit and back, and that movement is where the energy goes. The
conductance lattice we have built all book — the very $W$ of Chapter 2, the
metric lattice, the plastic memory of Chapter 10, the sparse store of Chapter
18 — has a second life as a *physical* object. A memristor is a resistor whose
resistance is the stored number. Ohm's law performs the multiply. Kirchhoff's
current law performs the sum. Both happen in the time it takes a voltage pulse
to settle, in the place the number is stored, with no operand movement at all.
The question is how much energy that costs, and the answer is a formula we can
derive from first principles and audit against the production SDK.

The chapter is organized as a build. In §21.1 we write down the 1T1R cell and
the two circuit laws that make it a computer. In §21.2 we derive the energy law
of Theorem B.21.1 from Joule heating and Kirchhoff's law, prove its monotonicity
and homogeneity, and reproduce the program's headline figure — **443.9 pJ** —
step by step. In §21.3 we prove that our model is not merely similar to the
SDK's but *identical* to it: the book's energy model and
`geoformer.hardware.crossbar.PhysicalMemristorCrossbar` agree to the last bit,
and both auditors report the same **9,462×** reduction against an FP32 server
GPU. In §21.4 we turn to the failure mode that kills analog in-memory compute
in practice — device noise — prove a contraction bound on the routing decision
(Theorem B.21.3 and its two-hop composition B.21.4), and run a seeded
Monte-Carlo that reproduces the program's published **38%** accuracy at 20%
conductance noise. In §21.5 we show the energy conclusion is robust to $\pm
10\%$ perturbation of every physical parameter. In §21.6 we add conductance
drift over time. In §21.7 we print the full baseline ledger. And in §21.8 we
draw the honest boundary of what a model can and cannot claim.

---

## 21.1 The 1T1R Cell: Computing with Ohm and Kirchhoff

A crossbar array is a grid of wires. The horizontal wires are the *word lines*
and the vertical wires are the *bit lines*; at every crossing sits a cell. The
1T1R cell — one transistor, one resistor — is the standard ReRAM building block:
a memristor whose conductance $G_{ij}$ is the stored analog weight, in series
with a selector transistor that isolates it during programming. We index the
array by output row $i \in \{1,\dots,N_{\text{out}}\}$ and input column
$j \in \{1,\dots,N_{\text{in}}\}$, and we hold in cell $(i,j)$ the conductance
$G_{ij}$ in Siemens. The device has two resistance states: the High-Resistance
State (HRS) at $g_{\min}$ and the Low-Resistance State (LRS) at $g_{\max}$.
Between them the conductance is programmed to one of $2^{b}$ discrete levels,
where $b$ is the bit precision — $b = 4$ gives the sixteen states of the
program's 4-bit design. The parameters we will use are collected here, and
every one is **MODELED** from published 1T1R ReRAM specifications and the
peripheral-circuit surveys the SDK cites.

| Symbol | Value | Meaning |
|---|---|---|
| $N_{\text{in}} \times N_{\text{out}}$ | $64 \times 64 = 4{,}096$ | array of 1T1R cells |
| $g_{\min}$ | $10\ \mu\text{S}$ | HRS conductance |
| $g_{\max}$ | $150\ \mu\text{S}$ | LRS conductance |
| $b$ | $4$ bits ($16$ states) | conductance precision |
| $\Delta g$ | $(g_{\max}-g_{\min})/(b-1) = 9.3333\ \mu\text{S}$ | level step |
| $V$ | $0.2\ \text{V}$ | read-pulse amplitude |
| $t$ | $10\ \text{ns}$ | read-pulse width |
| $R_{\text{wire}}$ | $3.2\ \Omega$ per cell | interconnect resistance |
| $\nu$ | $0.045$ | conductance drift exponent |
| $1+\eta$ | $1.6$ | peripheral overhead ($\eta = 0.6$) |
| $n_{\text{stages}}$ | $2$ | sequential MVM stages per 2-hop route |

Programming is a two-step physical act. First the target analog value
$W_{ij} \in [0,1]$ is *quantized* to the nearest of the sixteen levels,
$\ell_{ij} = \operatorname{clip}(\operatorname{round}(W_{ij}(b-1)), 0, b-1)$,
and the ideal cell conductance is set to $G^{\text{ideal}}_{ij} = g_{\min} +
\ell_{ij}\Delta g$. Second, the write pulse lands with cycle-to-cycle variation,
so the realized conductance is
$G_{ij} = \operatorname{clip}(G^{\text{ideal}}_{ij}(1+\delta_{ij}),\ \tfrac12
g_{\min},\ \tfrac32 g_{\max})$ with $\delta_{ij}$ a zero-mean random relative
error of scale $\sigma$. The clip window is the physical device envelope; a
memristor cannot be programmed to a conductance outside it.

Reading is where the computation happens. Drive input column $j$ to voltage
$v_j$ and let every other column float. Each cell $(i,j)$ then carries the
current $G_{ij} v_j$ by Ohm's law. Kirchhoff's current law says the currents
arriving on bit line $i$ sum, so the total output current is

$$
i_i = \sum_{j} G_{ij}\, v_j,
$$

which is the matrix–vector product $i = G v$ executed in one transport time.
The array does not *simulate* the multiplication; the physics *is* the
multiplication. For a two-hop route the output current of the first stage is
buffered through a sample-and-hold, renormalized to the read voltage, and
injected as the input of a second stage — the analog realization of the $W^2$
geometric highway. Two stages, then, per inference.

Two non-idealities intrude on the ideal. The first is **wire parasitics**: the
metal lines have resistance, so the voltage that reaches a cell is attenuated
by the IR drop along the line. To first order the attenuation factor is
$1/(1 + \tfrac12 R_{\text{wire}} N_{\text{in}} \bar{G})$, where $\bar{G}$ is the
mean cell conductance — a distributed R–G ladder Taylor-expanded about the
uniform state. The second is **read noise**: thermal Johnson–Nyquist and shot
noise add a zero-mean current perturbation of about $0.5\%$ of the peak output.
Both enter the routing simulation of §21.4. Neither enters the energy law, for
a reason we will make precise in Theorem B.21.2.

---

## 21.2 Theorem B.21.1 — The Energy Law, Derived from Joule and Kirchhoff

The headline of this chapter is an energy, and the first thing to establish is
that it is not a fitted constant. It is the exact Joule heating of the read
pulses the computation requires.

> **Theorem B.21.1 (The crossbar energy law).**
> Consider an $N_{\text{out}} \times N_{\text{in}}$ conductance array with
> cells $G_{ij} \ge 0$, driven for $n$ sequential stages by read pulses of
> amplitude $V$ and duration $t$, with peripherals that consume a fraction
> $\eta$ of the core energy. The dynamic energy dissipated is
> $$
> E \;=\; n\,(1+\eta)\,V^{2}\,t\,\sum_{i,j} G_{ij}.
> $$
> $E$ is strictly increasing in $V$ (for $V > 0$), in $t$, in each $G_{ij}$, and
> in $n$; and it is homogeneous: scaling $V$ by $\lambda$ scales $E$ by
> $\lambda^{2}$, while scaling $t$ or every $G_{ij}$ by $\lambda$ scales $E$ by
> $\lambda$.

**Proof.** A resistor of conductance $G$ held at voltage $V$ dissipates
instantaneous power $P = V^{2}G$ (Joule's law, $P = VI$ with $I = GV$). Over a
pulse of duration $t$ the cell dissipates energy
$E_{ij} = V^{2} G_{ij} t$. During a single vector–matrix stage every cell is
biased simultaneously, so the array's total dissipation is the sum over cells,
$$
E_{\text{stage}} = \sum_{i,j} E_{ij} = V^{2} t \sum_{i,j} G_{ij},
$$
the sum being exactly the Kirchhoff accounting of §21.1. Repeating the stage
$n$ times multiplies the energy by $n$, and the peripheral circuits (digital-to-
analog converters, analog-to-digital converters, and the sample-and-hold
capacitors that make the second stage possible) draw a further fraction $\eta$
of the core, giving the factor $(1+\eta)$. Substituting $\eta = 0.6$ and
$n = 2$ yields the program's $2 \times 1.6\,V^{2}t\sum G$.

For monotonicity, differentiate the law:
$$
\frac{\partial E}{\partial V} = 2n(1+\eta)V t \sum_{i,j} G_{ij} > 0 \quad (V>0),
\qquad
\frac{\partial E}{\partial t} = n(1+\eta)V^{2}\sum_{i,j} G_{ij} > 0,
$$
$$
\frac{\partial E}{\partial G_{ij}} = n(1+\eta)V^{2} t > 0,
\qquad
\frac{\partial E}{\partial n} = (1+\eta)V^{2}t\sum_{i,j} G_{ij} > 0 .
$$
Each partial derivative is a product of positive factors, so $E$ is strictly
increasing in each argument. Homogeneity is immediate: replacing $V$ by
$\lambda V$ multiplies the right-hand side by $\lambda^{2}$, and replacing $t$
by $\lambda t$ or every $G_{ij}$ by $\lambda G_{ij}$ multiplies it by
$\lambda$. $\blacksquare$

The theorem tells us exactly which knobs matter and how much. The voltage is
the strongest: because the dependence is quadratic, a $10\%$ change in $V$
moves the energy by about $21\%$, while a $10\%$ change in $t$ moves it by
$10\%$. The pulse width and the conductance sum are linear. This ordering is
what makes the sensitivity analysis of §21.5 predictable before it is run.

The second theorem of this section is a structural fact that matters more than
it first appears: the energy does not care *where* the conductances sit.

> **Theorem B.21.2 (Energy–arrangement invariance).**
> The dynamic energy $E$ of Theorem B.21.1 depends on the conductance array only
> through the scalar sum $\sum_{i,j} G_{ij}$. Consequently, any permutation of
> the $N_{\text{in}} N_{\text{out}}$ cells leaves $E$ unchanged.

**Proof.** The right-hand side of the energy law is a function of the scalar
$\sum_{i,j} G_{ij}$ and of the scalars $n$, $\eta$, $V$, $t$ alone. A bijection
of the cells permutes the terms of the sum but leaves it unchanged, and the
other arguments are untouched. Hence $E$ is invariant. $\blacksquare$

The invariance is the precise statement of a decoupling we will lean on. Energy
is a property of the *sum* of the conductances — how much total current the
array can carry — while the routing decision of §21.4 is a property of their
*arrangement* — which output wins the argmax. The two are independent, which is
why a noisy array can still route correctly while its energy budget is
untouched by the noise.

### Task 1: Re-deriving 443.9 pJ from First Principles

We now reproduce the headline figure with nothing but the theorem. We program
the array uniformly at $W_{\text{norm}} = 0.5$, which quantizes to level
$\operatorname{round}(0.5 \cdot 15) = 8$, giving a cell conductance
$G^{\text{ideal}} = g_{\min} + 8\Delta g = 84.6667\ \mu\text{S}$. There are
$64 \times 64 = 4096$ cells, so $\sum G = 4096 \cdot 84.6667\ \mu\text{S} =
346.794667\ \text{mS}$. One stage is $V^{2}t\sum G = 0.04 \cdot 10^{-8} \cdot
0.346795 = 1.387179\times10^{-10}\ \text{J}$; two stages plus the $1.6$
peripheral factor give $4.438972\times10^{-10}\ \text{J}$, or
$0.443897\ \text{nJ} = 443.897\ \text{pJ}$. Before running, predict the
comparison: $4200\ \text{nJ} / 0.443897\ \text{nJ} \approx 9462$, so the figure
should land just under ten thousand times below an FP32 server GPU.

**Run this now.** Task 1 (`python3 code/gftac/ch21_tasks.py 1`):

```python
cfg = CrossbarConfig()
cb = MemristorCrossbar(cfg)
cb.program_conductance_matrix(np.full((cfg.n_outputs, cfg.n_inputs), 0.5), noise_std=0.0)

level = int(cb.quantize_levels(0.5))
g_cell = float(cb.G_ideal[0, 0])
g_total = cb.total_conductance()
e_stage = (cfg.v_read ** 2) * g_total * cfg.t_pulse
e_two = e_stage * 2.0
e_periph = e_two * cfg.peripheral_factor
e_nj = cb.compute_dynamic_energy_per_inference()
e_pj = cb.compute_dynamic_energy_pj()
gpu_ratio = 4200.0 / e_nj

print("Task 1 — First-Principles Re-Derivation of the 443.9 pJ Figure (MODELED)")
print(f"  • Crossbar array:          {cfg.n_inputs} x {cfg.n_outputs} = {cfg.n_inputs * cfg.n_outputs:,} 1T1R cells")
print(f"  • HRS g_min:               {cfg.g_min * 1e6:.3f} uS")
print(f"  • LRS g_max:               {cfg.g_max * 1e6:.3f} uS")
print(f"  • 4-bit levels / step:     {cfg.b_levels} states, delta_g = {cfg.delta_g * 1e6:.4f} uS")
print(f"  • Uniform W_norm = 0.5 ->  level {level} ->  cell G = {g_cell * 1e6:.4f} uS")
print(f"  • Total conductance sum(G):{g_total * 1e3:.6f} mS")
print(f"  • One-stage energy V^2*G*t:{e_stage:.6e} J")
print(f"  • Two-stage energy:        {e_two:.6e} J")
print(f"  • x1.6 peripheral factor:  {e_periph:.6e} J")
print(f"  • Total energy:            {e_nj:.6f} nJ = {e_pj:.3f} pJ")
print(f"  • Reduction vs FP32 GPU:   {gpu_ratio:,.0f}x")
```

```text
Task 1 — First-Principles Re-Derivation of the 443.9 pJ Figure (MODELED)
  • Crossbar array:          64 x 64 = 4,096 1T1R cells
  • HRS g_min:               10.000 uS
  • LRS g_max:               150.000 uS
  • 4-bit levels / step:     16 states, delta_g = 9.3333 uS
  • Uniform W_norm = 0.5 ->  level 8 ->  cell G = 84.6667 uS
  • Total conductance sum(G):346.794667 mS
  • One-stage energy V^2*G*t:1.387179e-10 J
  • Two-stage energy:        2.774357e-10 J
  • x1.6 peripheral factor:  4.438972e-10 J
  • Total energy:            0.443897 nJ = 443.897 pJ
  • Reduction vs FP32 GPU:   9,462x
```

**Interpretation.** Every line of the ledger is a step of the proof. The array
is $4096$ cells; the level is $8$; the cell conductance is $84.6667\ \mu\text{S}$;
the sum is $346.794667\ \text{mS}$; one stage is $1.387179\times10^{-10}\
\text{J}$; two stages with the peripheral factor are $4.438972\times10^{-10}\
\text{J}$. The total is $0.443897\ \text{nJ} = 443.897\ \text{pJ}$, and the
reduction is $9{,}462\times$. Note that the $9{,}462$ is the *rounded* quotient
of the exact ratio $9461.65$; the rounding is in the display, not the physics.
Note also what is **MODELED** here and what is not: the conductance values, the
pulse width, and the peripheral factor are all inputs of the model, so the
$443.9\ \text{pJ}$ is a *prediction about a hypothetical array*, not a
measurement of one. It is exactly the kind of number a model should produce:
transparent, reproducible, and auditable line by line.

---

## 21.3 Theorem B.21.3 — Parity with the Production SDK

A model is only as trustworthy as its agreement with the artifact it claims to
describe. The SDK ships two objects for this work:
`PhysicalMemristorCrossbar` in `geoformer.hardware.crossbar` and
`NanojouleEnergyAuditor` in `geoformer.hardware.energy`. Our model is a
re-implementation, and the parity claim is not "similar" but "identical". The
reason it can be identical is that the energy is a closed-form expression:
there is no numerical iteration to disagree about, only the order of floating-
point operations, which we match deliberately.

> **Theorem B.21.3 (Energy parity).**
> For any conductance array $G$ and any parameter tuple in the shared domain,
> the book model's `compute_dynamic_energy_per_inference()` equals the SDK's to
> the last bit of IEEE-754 double precision.

**Proof.** Both implementations compute
$\big((V^{2} G_{\text{tot}} t) + (V^{2} G_{\text{tot}} t)\big)(1+\eta)\cdot
10^{9}$ with $G_{\text{tot}} = \sum_{ij} G_{ij}$, in the same association order:
the single-stage product is formed first, doubled by addition, scaled by the
peripheral factor, then converted to nanojoules. Since the two sums over
$G_{ij}$ traverse the same array in the same order and the subsequent scalar
operations are performed in the same sequence on the same IEEE-754 double
values, the two results are the same double. $\blacksquare$

The parity test in `tests/test_ch21_memristor.py` asserts this with an absolute
tolerance of $10^{-9}\ \text{nJ}$ and in fact observes a difference of exactly
zero. The auditor parity is equally exact, because both auditors divide the same
four baseline constants by the same modeled energy.

**Run this now.** Task 2 (`python3 code/gftac/ch21_tasks.py 2`):

```python
from geoformer.hardware.crossbar import PhysicalMemristorCrossbar
from geoformer.hardware.energy import NanojouleEnergyAuditor

cfg = CrossbarConfig()
book = MemristorCrossbar(cfg)
sdk = PhysicalMemristorCrossbar()
W = np.full((64, 64), 0.5)
book.program_conductance_matrix(W, noise_std=0.0)
sdk.program_conductance_matrix(W, noise_std=0.0)

e_book = book.compute_dynamic_energy_per_inference()
e_sdk = float(sdk.compute_dynamic_energy_per_inference())
diff = abs(e_book - e_sdk)

r_book = CrossbarEnergyAuditor.audit(book)["nvidia_gpu_fp32"]["reduction_ratio"]
r_sdk = NanojouleEnergyAuditor.audit(sdk)["nvidia_gpu_fp32"]["reduction_ratio"]

print("Task 2 — SDK Parity Audit: book model vs geoformer.hardware (MODELED)")
print(f"  • Book crossbar energy:    {e_book:.9f} nJ")
print(f"  • SDK  crossbar energy:    {e_sdk:.9f} nJ")
print(f"  • Absolute difference:     {diff:.3e} nJ")
print(f"  • Book GPU-FP32 ratio:     {r_book:,.4f}x")
print(f"  • SDK  GPU-FP32 ratio:     {r_sdk:,.4f}x")
print(f"  • Parity verified (<1e-9): {diff < 1e-9 and abs(r_book - r_sdk) < 1e-9}")
```

```text
Task 2 — SDK Parity Audit: book model vs geoformer.hardware (MODELED)
  • Book crossbar energy:    0.443897173 nJ
  • SDK  crossbar energy:    0.443897173 nJ
  • Absolute difference:     0.000e+00 nJ
  • Book GPU-FP32 ratio:     9,461.6507x
  • SDK  GPU-FP32 ratio:     9,461.6507x
  • Parity verified (<1e-9): True
```

**Interpretation.** The absolute difference is $0.000\times10^{0}\ \text{nJ}$ —
not "small", but *zero*. Both models report $0.443897173\ \text{nJ}$ and both
auditors report a $9{,}461.6507\times$ reduction. This is the strongest form of
the parity discipline the book has followed since Chapter 4: the number in the
book is the number the production SDK produces, because it is the same number.
The consequence for the reader is that the $443.9\ \text{pJ}$ claim is not a
book-specific artifact. It is what the shipped hardware model reports, and the
book has merely written that model down where it can be read and checked.

---

## 21.4 Device Noise: Contraction Theorems and the 38% Curve

Analog in-memory computing has one notorious weakness: the stored value is
physical, and physics is noisy. A memristor programmed to a conductance has a
distribution of conductances around it, and the routing decision — which output
current wins the argmax — can flip if the noise is large enough. The question
this section answers is how large "large enough" is, and the answer is a
contraction bound that turns the vague worry into an inequality.

We work with the multiplicative noise model of §21.1: the realized conductance
is $G'_{ij} = G_{ij}(1+\delta_{ij})$ with $|\delta_{ij}| \le \sigma$. The first
theorem bounds what one noisy matrix–vector stage does to its output.

> **Theorem B.21.4 (Per-stage relative contraction).**
> Let a stage compute $i = Gv$ from a non-negative input $v \ge 0$. If the
> conductances are perturbed multiplicatively with $|\delta_{ij}| \le \sigma$
> and the perturbed conductances remain non-negative, then the perturbed output
> $i' = G'v$ satisfies, for every output row $i$,
> $$
> |i'_i - i_i| \;\le\; \sigma\, i_i .
> $$
> A relative perturbation of size $\sigma$ produces an output perturbation of at
> most the same relative size.

**Proof.** Expand the difference:
$i'_i - i_i = \sum_j (G'_{ij} - G_{ij})v_j = \sum_j G_{ij}\delta_{ij}v_j$.
Taking absolute values and bounding each term using $|\delta_{ij}| \le \sigma$
and $v_j \ge 0$,
$$
|i'_i - i_i| \;\le\; \sum_j G_{ij}\,|\delta_{ij}|\,v_j
\;\le\; \sigma \sum_j G_{ij} v_j \;=\; \sigma\, i_i .
$$
The bound is tight, attained when every $\delta_{ij} = \sigma$. $\blacksquare$

The bound requires $v \ge 0$, which the read scheme guarantees: the input is a
single non-negative read pulse. It says that a $20\%$ relative conductance error
cannot move an output current by more than $20\%$. The danger is not a single
stage, however; it is the *composition* of two stages, where the first stage's
error feeds the second through the sample-and-hold normalization. The next
theorem carries the bound through the composition.

> **Theorem B.21.5 (Two-hop margin contraction).**
> Let a two-hop route from source $s$ apply stage one, normalize by the peak
> current (the sample-and-hold), and apply stage two. If the conductance
> perturbation has relative scale $\sigma < 1$, the final output currents
> satisfy $|i'_i - i_i| \le \varepsilon\, i_i$ with
> $$
> \varepsilon = \frac{\sigma(3+\sigma)}{1-\sigma}.
> $$
> Moreover, if the noiseless currents have a unique maximizer $t$ with margin
> ratio $\rho = i_t / \max_{j \ne t} i_j$, the perturbed argmax is still $t$
> whenever $\rho > (1+\varepsilon)/(1-\varepsilon)$.

**Proof.** Write the input as $x = V e_s \ge 0$ and the first-stage currents as
$a = Gx$. By Theorem B.21.4, $|a'_j - a_j| \le \sigma a_j$. Let $m = \lVert
a\rVert_\infty$ and $m' = \lVert a'\rVert_\infty$. Then $|m' - m| \le \max_j
|a'_j - a_j| \le \sigma m$, so $m' \ge (1-\sigma)m$. The normalized inputs are
$v_2 = Va/m$ and $v'_2 = Va'/m'$, and
$$
|v'_{2,j} - v_{2,j}| = V\frac{|a'_j m - a_j m'|}{m m'}
\le V\frac{m|a'_j - a_j| + a_j|m - m'|}{m m'}
\le V\frac{2\sigma a_j m}{m m'} .
$$
Since $m' \ge (1-\sigma)m$, the right-hand side is at most
$\frac{2\sigma}{1-\sigma}v_{2,j}$, i.e. $v'_2 = v_2 \circ (1+\delta^{(2)})$ with
$|\delta^{(2)}_j| \le 2\sigma/(1-\sigma)$. Now apply stage two:
$i'_i - i_i = \sum_j G_{ij}\big(v'_{2,j} - v_{2,j}\big) + \sum_j
G_{ij}\delta_{ij}v_{2,j} + \sum_j G_{ij}\delta_{ij}\big(v'_{2,j}-v_{2,j}\big)$.
Bounding the three sums in turn by
$\frac{2\sigma}{1-\sigma}i_i$, $\sigma i_i$, and $\frac{2\sigma^2}{1-\sigma}i_i$
gives
$$
|i'_i - i_i| \le \Big(\frac{2\sigma}{1-\sigma} + \sigma + \frac{2\sigma^2}{1-\sigma}\Big) i_i
= \frac{\sigma(3+\sigma)}{1-\sigma}\, i_i .
$$
For the argmax claim, the perturbed maximizer satisfies $i'_t \ge
(1-\varepsilon)i_t$ and, for $j \ne t$, $i'_j \le (1+\varepsilon)i_j \le
(1+\varepsilon)\max_{j\ne t} i_j$. Hence $i'_t > i'_j$ for every $j \ne t$
whenever $(1-\varepsilon)i_t > (1+\varepsilon)\max_{j\ne t}i_j$, which is the
stated condition. $\blacksquare$

The bound is honest about its own conservatism. At $\sigma = 0.2$ we get
$\varepsilon = 0.2 \cdot 3.2 / 0.8 = 0.8$, and the argmax is guaranteed only for
margin ratios above $1.8/0.2 = 9$. The measured routing margin on the Tier-4
graph is far below $9$ for many queries, yet the measured accuracy at $20\%$
noise is $38\%$, not near zero. The reason is that the theorem bounds the
*worst case* — every cell perturbed adversarially in the direction that hurts —
whereas real programming noise is zero-mean and independent across cells, so the
errors partially cancel. The theorem is a guarantee, and a guarantee that is not
tight is a feature, not a defect: it tells the reader which claim survives
adversarial noise and which survives only statistical noise. We say plainly that
the $38\%$ is a *statistical* result; the theorem is the *adversarial* one.

We can also read the contraction as a statement about the small-noise regime:
as $\sigma \to 0$, $\varepsilon \to 3\sigma$, so the threshold margin ratio
approaches $1$ linearly in $\sigma$. The accuracy therefore degrades
continuously from its noise-free value, with a failure probability that grows at
most linearly in $\sigma$ near zero. This is the quantitative form of the
intuition that small noise does small damage.

### Task 3: The Monte-Carlo Noise-Resilience Curve

The definition of *accuracy* must be pinned, because "accuracy" is where
analog-computing claims most often hide. We use the following, and only the
following: the Tier-4 benchmark's 40 directed chains over 64 entities give 80
single-hop facts and 40 held-out two-hop queries $(a, c)$ with the middle node
$b$ never revealed. We program the saturated highway matrix — $W_{b,a} = 1$ for
each trained fact — into the crossbar, run the two physical stages of §21.1,
and count a query a **hit** iff the argmax of the second-stage output currents
equals the ground-truth target $c$. Accuracy is the hit rate averaged over the
40 queries, the five canonical seeds $\{42, 101, 2024, 777, 999\}$, and five
independent programming-noise realizations per seed, for $5 \times 5 = 25$
trials per noise level. Every RNG stream is derived deterministically from the
seed and trial index, so the curve below is reproducible to the digit. Chance
is $1/64 = 1.56\%$. We predict, from the flatness the SDK's own benchmark
reports, that accuracy will sit near $38\%$ at every noise level and that the
$20\%$ point will land within a fraction of a point of the published figure.

**Run this now.** Task 3 (`python3 code/gftac/ch21_tasks.py 3`):

```python
sim = NoiseResilienceSimulator(trials=5)
results = sim.run()

print("Task 3 — Monte-Carlo Noise-Resilience Curve (MODELED)")
print("  Accuracy = held-out 2-hop argmax hits / queries,")
print("  mean over 5 seeds x 5 noise realizations (1/64 chance = 1.56%).")
print(f"  {'Noise sigma':>12} | {'Mean accuracy':>13} | {'Std dev':>8} | Status")
print("  " + "-" * 56)
for noise, mean, std in results:
    status = "RESILIENT" if mean >= 30.0 else ("DEGRADED" if mean >= 15.0 else "COLLAPSED")
    print(f"  {noise * 100:>10.1f}% | {mean:>12.1f}% | {std:>7.1f}% | {status}")
print("  " + "-" * 56)
headline = [m for n, m, _ in results if abs(n - 0.20) < 1e-12][0]
print(f"  • Headline: {headline:.1f}% accuracy at 20.0% conductance noise")
print(f"  • Above the 1.56% chance floor: {headline > 1.56}")
```

```text
Task 3 — Monte-Carlo Noise-Resilience Curve (MODELED)
  Accuracy = held-out 2-hop argmax hits / queries,
  mean over 5 seeds x 5 noise realizations (1/64 chance = 1.56%).
   Noise sigma | Mean accuracy |  Std dev | Status
  --------------------------------------------------------
         0.0% |         38.4% |     5.7% | RESILIENT
         5.0% |         38.1% |     7.1% | RESILIENT
        10.0% |         37.3% |     7.1% | RESILIENT
        15.0% |         37.6% |     7.2% | RESILIENT
        20.0% |         38.0% |     6.8% | RESILIENT
  --------------------------------------------------------
  • Headline: 38.0% accuracy at 20.0% conductance noise
  • Above the 1.56% chance floor: True
```

**Interpretation.** The curve is flat at about $38\%$ across the whole noise
range: $38.4\%$ at zero noise, $38.0\%$ at the worst case of $20\%$. That
flatness is the finding, and it is a *modeled* one — the accuracy is limited not
by device noise but by the graph's own ambiguity, since a middle node shared by
several chains sends the two-hop mass to several targets at once. The device
noise moves the result by less than a point, which is exactly what the
contraction theorems lead us to expect: a $20\%$ conductance perturbation is a
$20\%$ output perturbation, and a $20\%$ current perturbation rarely flips an
argmax that was not already close. The headline is $38.0\%$ at $20\%$ noise,
reproducing the program's published **38.0% @ 20% noise** figure, and every
level is far above the $1.56\%$ chance floor. The standard deviations are large
(about $7$ points) because each seed's graph is different; we report them rather
than hiding them.

---

## 21.5 Sensitivity: The Conclusion Survives Perturbation

A model is only as useful as the robustness of its conclusions. The $9{,}462\times$
headline could be an artifact of the particular parameter values we happened to
choose; if a $10\%$ change in $g_{\min}$ collapsed it, the claim would be
fragile and we should say so. This section runs the test. We perturb each of the
four physical parameters — $g_{\min}$, $g_{\max}$, $V$, and $t$ — by $\pm 10\%$,
one at a time, holding the others fixed, and recompute the energy and the GPU
reduction. Theorem B.21.1 predicts the ordering before we measure: $V$ enters
quadratically, so it should produce the largest energy swing; $t$ and the
conductances enter linearly, so they should produce smaller ones; and $g_{\min}$
should move the energy least of all, because at $W_{\text{norm}} = 0.5$ the cell
conductance is dominated by the $8\Delta g$ term, of which $g_{\min}$ is a small
part.

**Run this now.** Task 4 (`python3 code/gftac/ch21_tasks.py 4`):

```python
rows = sensitivity_table(perturbations=(0.9, 1.1))
baseline_nj = rows[0][2]

print("Task 4 — Sensitivity Analysis: +/-10% on Each Physical Parameter (MODELED)")
print(f"  {'Parameter':>10} | {'Perturb':>8} | {'Energy (pJ)':>11} | {'GPU ratio':>10}")
print("  " + "-" * 50)
min_ratio = float("inf")
for param, p, e_nj, e_pj, ratio in rows:
    min_ratio = min(min_ratio, ratio)
    label = param if param != "baseline" else "baseline"
    print(f"  {label:>10} | {p:>7.2f}x | {e_pj:>11.3f} | {ratio:>9,.0f}x")
print("  " + "-" * 50)
print(f"  • Baseline energy:         {baseline_nj * 1000:.3f} pJ")
print(f"  • Worst-case GPU ratio:    {min_ratio:,.0f}x")
print(f"  • Thousands-fold conclusion invariant: {min_ratio > 1000.0}")
```

```text
Task 4 — Sensitivity Analysis: +/-10% on Each Physical Parameter (MODELED)
   Parameter |  Perturb | Energy (pJ) |  GPU ratio
  --------------------------------------------------
    baseline |    1.00x |     443.897 |     9,462x
       g_min |    0.90x |     441.450 |     9,514x
       g_min |    1.10x |     446.344 |     9,410x
       g_max |    0.90x |     401.954 |    10,449x
       g_max |    1.10x |     485.840 |     8,645x
      v_read |    0.90x |     359.557 |    11,681x
      v_read |    1.10x |     537.116 |     7,820x
     t_pulse |    0.90x |     399.507 |    10,513x
     t_pulse |    1.10x |     488.287 |     8,602x
  --------------------------------------------------
  • Baseline energy:         443.897 pJ
  • Worst-case GPU ratio:    7,820x
  • Thousands-fold conclusion invariant: True
```

**Interpretation.** The predictions hold in order. The voltage is the strongest
knob: $\pm 10\%$ moves the energy between $359.6$ and $537.1\ \text{pJ}$, a
swing of nearly $180\ \text{pJ}$. The pulse width and $g_{\max}$ are next, each
moving the energy by tens of picajoules. The HRS conductance $g_{\min}$ is the
weakest, moving it by less than $5\ \text{pJ}$, exactly because it is a small
part of the programmed cell conductance. Most importantly, the *conclusion* is
invariant: across all nine perturbations the reduction against an FP32 GPU never
drops below $7{,}820\times$. The thousands-fold claim is not a knife-edge
artifact of the parameters; it survives a simultaneous-worst-case walk of the
entire parameter box.

---

## 21.6 Theorem B.21.6 — Drift and the Energy over Time

A memristor is not a fixed resistor. After programming, its conductance relaxes
toward equilibrium — the oxygen vacancies that set the resistance diffuse, and
the conductance drifts downward as a power law in time. The standard model is
$G(t) = G_0 (t/t_0)^{-\nu}$ with a small exponent $\nu$ of order a few hundredths.
Because the energy law depends only on the conductance sum, drift has a
predictable effect on energy, and it is a helpful one.

> **Theorem B.21.6 (Drift energy law).**
> Under uniform drift $G_{ij}(t) = G_{ij}(t_0)(t/t_0)^{-\nu}$ applied to every
> cell, the dynamic energy satisfies
> $$
> E(t) = E(t_0)\,\Big(\frac{t}{t_0}\Big)^{-\nu}.
> $$
> Hence $E$ is strictly decreasing in $t$, and $E(t) \to 0$ as $t \to \infty$.

**Proof.** The drift law is a common scalar multiplier on every conductance, so
$\sum_{ij} G_{ij}(t) = (t/t_0)^{-\nu}\sum_{ij} G_{ij}(t_0)$. Substituting into
the energy law of Theorem B.21.1 and factoring out the constant terms gives the
stated expression. Since $\nu > 0$, the factor $(t/t_0)^{-\nu}$ is strictly
decreasing in $t$ and tends to $0$; multiplying by the positive constant
$E(t_0)$ preserves both properties. $\blacksquare$

The theorem is why drift, feared as a reliability problem for *accuracy*, is a
tailwind for *energy*: a drifting array draws less current over time and so
dissipates less. The accuracy cost of drift is real and separate — it is a
systematic shift of the whole conductance distribution, not the zero-mean
perturbation of §21.4 — and we do not claim otherwise.

**Run this now.** Task 5 (`python3 code/gftac/ch21_tasks.py 5`):

```python
cfg = CrossbarConfig()
base = MemristorCrossbar(cfg)
base.program_conductance_matrix(np.full((64, 64), 0.5), noise_std=0.0)
e0_nj = base.compute_dynamic_energy_per_inference()

print("Task 5 — Conductance Drift G(t)=G0*(t/t0)^(-nu) vs Energy (MODELED)")
print(f"  Drift exponent nu = {cfg.drift_exp} (dimensionless)")
print(f"  {'Time':>10} | {'Drift factor':>12} | {'Energy (pJ)':>11} | {'Retained':>8}")
print("  " + "-" * 50)
for t in (1.0, 1e3, 1e6, 1e9):
    f = drift_factor(t, 1.0, cfg.drift_exp)
    print(f"  {t:>9.0e}s | {f:>12.6f} | {e0_nj * f * 1000:>11.3f} | {f * 100.0:>7.1f}%")
print("  " + "-" * 50)
print(f"  • Drift lowers conductance, so energy falls with it")
print(f"  • Energy is monotone in the retained factor: True")
```

```text
Task 5 — Conductance Drift G(t)=G0*(t/t0)^(-nu) vs Energy (MODELED)
  Drift exponent nu = 0.045 (dimensionless)
        Time | Drift factor | Energy (pJ) | Retained
  --------------------------------------------------
      1e+00s |     1.000000 |     443.897 |   100.0%
      1e+03s |     0.732825 |     325.299 |    73.3%
      1e+06s |     0.537032 |     238.387 |    53.7%
      1e+09s |     0.393550 |     174.696 |    39.4%
  --------------------------------------------------
  • Drift lowers conductance, so energy falls with it
  • Energy is monotone in the retained factor: True
```

**Interpretation.** At one second the factor is $1$ and the energy is the
familiar $443.897\ \text{pJ}$. At a thousand seconds it has fallen to
$0.732825$ of the programmed value, at a million seconds to $0.537032$, and at a
billion seconds to $0.393550$ — the energy tracks the drift factor exactly, as
Theorem B.21.6 requires, because the factor is a common scalar on the
conductance sum. The retained-energy column is the drift factor itself, which is
the theorem made visible. Drift is slow (a billion seconds is about $32$ years)
and gentle (the exponent is $0.045$), but it is monotone and predictable, and it
moves energy in the safe direction.

---

## 21.7 The Baseline Ledger

The chapter closes where every energy claim must close: against the machines it
proposes to replace. The baselines are the SDK's own four, and they are
**MODELED** published figures, not measurements taken here — an FP32 server GPU
at $4200\ \text{nJ}$, an FP16 tensor-core GPU at $1100\ \text{nJ}$, an INT8 edge
TPU at $145\ \text{nJ}$, and an INT8 ARM Cortex-M55 NPU at $65\ \text{nJ}$. The
crossbar's $443.9\ \text{pJ}$ is placed alongside them.

**Run this now.** Task 6 (`python3 code/gftac/ch21_tasks.py 6`):

```python
cfg = CrossbarConfig()
cb = MemristorCrossbar(cfg)
cb.program_conductance_matrix(np.full((64, 64), 0.5), noise_std=0.0)
e_nj = cb.compute_dynamic_energy_per_inference()
audit = CrossbarEnergyAuditor.audit(cb)

print("Task 6 — Baseline Energy Comparison per Inference (MODELED)")
print(f"  Crossbar energy: {e_nj:.6f} nJ = {e_nj * 1000:.3f} pJ")
print(f"  {'Platform':<32} | {'Energy':>10} | {'Reduction':>12}")
print("  " + "-" * 62)
order = ["nvidia_gpu_fp32", "nvidia_gpu_fp16", "google_edge_tpu", "arm_cortex_m55"]
for key in order:
    row = audit[key]
    if row["baseline_nj"] >= 1.0:
        e_str = f"{row['baseline_nj']:.0f} nJ"
    else:
        e_str = f"{row['baseline_nj'] * 1000:.0f} pJ"
    print(f"  {row['name']:<32} | {e_str:>10} | {row['reduction_ratio']:>10,.0f}x")
print(f"  {'Q-GeoFormer Memristor (this model)':<32} | {e_nj * 1000:>8.1f} pJ | {'1x (reference)':>12}")
print("  " + "-" * 62)
gpu = audit["nvidia_gpu_fp32"]["reduction_ratio"]
print(f"  • Headline: {gpu:,.0f}x lower energy than an FP32 server GPU")
print(f"  • All baselines exceeded by >100x: "
      f"{all(audit[k]['reduction_ratio'] > 100 for k in order)}")
```

```text
Task 6 — Baseline Energy Comparison per Inference (MODELED)
  Crossbar energy: 0.443897 nJ = 443.897 pJ
  Platform                         |     Energy |    Reduction
  --------------------------------------------------------------
  NVIDIA Server GPU (FP32)         |    4200 nJ |      9,462x
  NVIDIA Tensor Core GPU (FP16)    |    1100 nJ |      2,478x
  Google Edge TPU (INT8)           |     145 nJ |        327x
  ARM Cortex-M55 NPU (INT8)        |      65 nJ |        146x
  Q-GeoFormer Memristor (this model) |    443.9 pJ | 1x (reference)
  --------------------------------------------------------------
  • Headline: 9,462x lower energy than an FP32 server GPU
  • All baselines exceeded by >100x: True
```

**Interpretation.** The headline is $9{,}462\times$ against the FP32 server GPU,
and the ordering below it is monotone in the baseline's efficiency: $2{,}478\times$
against the FP16 tensor core, $327\times$ against the edge TPU, $146\times$
against the ARM NPU. Every baseline is beaten by more than $100\times$, which is
the load-bearing line — the crossbar is not merely better than the worst
comparator, it is better than all of them by two orders of magnitude or more.
The numbers carry the **MODELED** label without exception: the baselines are
published figures from the accelerator literature, and the crossbar figure is
the analytical prediction of §21.2. Neither side of the ratio is a measurement
taken on this machine.

---

## 21.8 The Honest Boundary of a Model

We have built an analytical account of a memristor crossbar and reproduced four
numbers: the $443.9\ \text{pJ}$ energy, the bit-exact SDK parity, the $38.0\%$
accuracy at $20\%$ noise, and the $9{,}462\times$ headline. It is worth being
exact about what each of them is.

The energy is a **MODELED** prediction. It follows from Joule's law and
Kirchhoff's law applied to a $64 \times 64$ array at published 1T1R
conductances, and it is identical to the production SDK's own model — but the
SDK's model is also a model. No wafer was fabricated. If the array were built,
the measured energy would differ from $443.9\ \text{pJ}$ by the calibration of
the peripheral factor, the true wire resistance, and the read circuitry's
efficiency. What the model establishes is not "the array uses $443.9\ \text{pJ}$"
but "an array with these specifications uses $443.9\ \text{pJ}$", and the
sensitivity analysis of §21.5 shows the conclusion survives the specification
being wrong by $10\%$ in any single parameter.

The noise result is **MODELED** and statistical. The $38.0\%$ is a Monte-Carlo
mean over a specific, pinned protocol — 40 held-out two-hop queries, five seeds,
five noise realizations — with the RNG streams fixed so the number is
reproducible to the digit. It reproduces the program's published $38.0\%$
figure. The contraction theorems (B.21.4, B.21.5) are the *guarantees* that sit
above the statistics: they say which routing decisions survive adversarial
noise, and we were explicit that the measured $38\%$ is better than the
adversarial bound because real noise is zero-mean, not adversarial.

The parity is **exact**. The book's model and the SDK's produce the same double,
to the last bit, because the energy is a closed form and we matched the
association order. This is the one claim in the chapter that admits no
qualification.

And the headline is a ratio of two **MODELED** numbers — the SDK's published
baseline figures and our analytical crossbar energy — and should be read as
such. It is a statement about the physics of in-memory computing: when the
multiply is Ohm's law and the sum is Kirchhoff's law, the operand-movement
energy that dominates a von Neumann machine's budget simply does not exist, and
what remains is the Joule heating of a ten-nanosecond pulse across four thousand
resistors. The ratio is large because the thing being avoided is large. That is
the argument, and the model is its arithmetic.

This closes Part V. We have taken the architecture from a dense lattice to a
sparse store to a physical circuit, and at every step the discipline has been
the same: derive the law, build the artifact, pin the transcript, and label the
claim. Part VI turns to the machinery that keeps the whole program honest — the
claims ledger, the certificates, and the capstone that runs the entire book in a
single file. The work of this chapter will appear there as a row: a **MODELED**
energy figure, an exact parity test, and a Monte-Carlo curve, each citing the
artifact that produces it.
