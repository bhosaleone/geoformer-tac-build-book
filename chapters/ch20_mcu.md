<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->
# Chapter 20 — MCU Deployment: GeoFormer on an Arduino Uno

Every machine we have reasoned on so far had a floating-point unit. The
lattices, the attractors, the solitons, and the hundred-thousand-entity memory
all lived in a world where `0.1` is a real number and multiplication is a
single instruction. This chapter removes that assumption, because the whole
argument of the book is that geometry is a *physical* quantity, and a physical
quantity should survive being carried onto the cheapest silicon that can hold
it. The target is an Atmel ATmega328P: eight bits, sixteen megahertz, two
kilobytes of SRAM, thirty-two kilobytes of Flash, and no floating-point unit at
all.

We have to say what we are not doing, plainly, before we do anything. The
firmware we reproduce here — `firmware/uno_nano_geoformer/main.c` — runs a
**discrete, row-normalized directed diffusion kernel**, not the multi-layer
GeoFormer foundation stack. There is no attention, no SwiGLU, no Cayley
projector, and no training loop on the chip. What is on the chip is the
smallest object that still deserves the name of the architecture: a conductance
lattice, a ternary projection in and out, and the iteration
$\mathbf{h}_{t+1} = \mathbf{h}_t \mathbf{P}$ run in integer arithmetic until an
attractor settles. The published benchmark dossier is explicit that this
validates "a frugal, discrete directed graph diffusion kernel executing natively
on bare-metal 8-bit microcontrollers", and that it "does not prove that the full
multi-layer GeoFormer foundation stack is active on the Uno." We will hold
ourselves to that sentence for the rest of the chapter.

Within that boundary the chapter is complete. We extract the exact integer
arithmetic the firmware uses, prove the two error bounds that make it
trustworthy, rebuild the model's weights and test vectors in the book so that
every number is reproducible without the repo, measure the fixed-point pipeline
against a float reference, account for every byte of Flash and SRAM, export a
PROGMEM C header and parse it back, and run the firmware's own adversarial
probes in simulation. The hardware latencies and energies we quote at the end
are **MODELED** from the published dossier; no board was attached for this
chapter. Everything else — the bounds, the agreement rate, the footprints, the
probe outcomes — is **MEASURED** on this machine and pinned by
`tests/test_ch20_mcu.py` and `tests/test_ch20_tasks.py`.

---

## 20.1 What Actually Runs on the Uno

The kernel is short enough to state completely. The firmware holds a
conductance lattice $W \in \mathbb{Z}_{\ge 0}^{N \times N}$ — for the exported
model $N = 32$ — and a state vector $h \in \{0,\dots,255\}^{N}$. One relaxation
step is a normalized push of mass along the directed arcs:

$$
\mathbf{h}_{t+1}[j]
= \left\lfloor
  \frac{1}{255}
  \sum_{i=0}^{N-1} \mathbf{h}_t[i]\,
  \left\lfloor \frac{W[i][j]\cdot 255}{\sum_{k} W[i][k]} \right\rfloor
\right\rfloor .
$$

Two conventions are load-bearing. First, $1.0$ is represented by the byte
`255`, not `256`; the firmware calls this constant `FP_ONE`. Second, every
division truncates toward zero — the two `floor` operators above are C integer
divisions. The inner floor computes a transition probability on the byte scale,
and the outer floor rescales the accumulated mass back to the state scale. The
structure is chosen so the accumulator never overflows an eight-bit register,
which is the reason the whole thing fits on the chip at all.

The full forward pass adds two ternary projections around the diffusion. The
input projection is a lookup, $h_{\text{latent}}[d] = W_{\text{in}}[d][\text{src}]$
with $W_{\text{in}} \in \{-1,0,+1\}^{d \times N}$; the output projection forms
an initial distribution $h_{\text{init}}[n] = \operatorname{clamp}(\sum_d
h_{\text{latent}}[d]\,W_{\text{out}}[n][d],\,0,\,255)$ with
$W_{\text{out}} \in \{-1,0,+1\}^{N \times d}$; and the source node is then given
a full pulse, $h[\text{src}] = 255$. The chip runs $\tau$ diffusion steps and
reads out the argmax. The two projections are ternary because
Chapter 4 established that a trit costs $1.6$ bits and that the effective weight
$w_q\gamma$ preserves the signal; here they are also what makes the input map a
table lookup rather than a multiply.

The comparison that matters for honesty is the one between the workstation model
and the embedded kernel. It is a table of genuine distinctions, and the
paragraphs that follow argue it through rather than restate it:

| Property | Full GeoFormer stack | Embedded MCU kernel |
| :--- | :--- | :--- |
| Substrate | Workstation / server / edge SBC | ATmega328P, 8-bit @ 16 MHz |
| Parameters | 249,665 (48.76 KiB packed) | 2,048 weights (2.0 KiB Flash) |
| Attention | Multi-head geodesic highway attention | Ternary projection → node pulse |
| Relaxation | Continuous ODE Laplacian heat diffusion | Discrete row-normalized Markov diffusion |
| Numerics | 1.58-bit ternary + 4-bit crossbar emulation | 1.58-bit Flash lookups + 8-bit fixed point |
| Learning | Two-timescale gradient + Hebbian plasticity | Static Flash prior + SRAM conductance warping |

The right-hand column is the entire subject of this chapter. It is a directed
graph diffusion engine with ternary I/O and integer dynamics. It is not a
foundation model, and we will not pretend it is.

---

## 20.2 Fixed-Point Arithmetic at `FP_ONE = 255`

The first thing to get right is the representation, because every subsequent
bound is a statement about it. The firmware stores a number $x \in [0,1]$ as a
byte $q = \operatorname{round}(255x)$, and reads it back as $q/255$. The choice
of $255$ rather than $256$ is deliberate: $256$ would make the scale a power of
two and the round-trip exact, but it would also let the maximum code overflow an
eight-bit register under accumulation, and the firmware's author chose the
headroom. The price of that choice is a representation error, and the price is
small and exactly computable.

> **Theorem B.20.1 (Fixed-point representation error).**
> Let $Q(x) = \operatorname{clip}(\operatorname{round}(255x), 0, 255)$ for
> $x \in \mathbb{R}$, and let $D(q) = q/255$ for $q \in \{0,\dots,255\}$. Then
> for every $x \in [0,1]$,
> $$
> \bigl| D(Q(x)) - x \bigr| \le \frac{1}{2\cdot 255} = \frac{1}{510},
> $$
> and the bound is attained at $x = (2m+1)/510$ for each $m = 0,\dots,254$.

**Proof.** For $x \in [0,1]$ we have $255x \in [0,255]$, so the clamp in $Q$ is
inactive and $Q(x) = \operatorname{round}(255x)$. Write $r = Q(x)$, an integer
in $\{0,\dots,255\}$. The defining property of rounding to the nearest integer is
$|r - 255x| \le \tfrac12$. Dividing by $255$,
$$
\bigl| D(Q(x)) - x \bigr| = \left| \frac{r}{255} - x \right|
= \frac{|r - 255x|}{255} \le \frac{1}{2\cdot 255}.
$$
For attainment, take $x = (2m+1)/510$ with $0 \le m \le 254$, so that
$255x = (2m+1)/2 = m + \tfrac12$ and rounding half up gives $r = m+1$. Then
$$
\left| \frac{m+1}{255} - \frac{2m+1}{510} \right|
= \frac{|2(m+1) - (2m+1)|}{510} = \frac{1}{510}.
$$
Hence the bound is tight on the whole interval. $\blacksquare$

The theorem says the representation loses at most one half-step of the byte
scale, and it says exactly where the loss is worst. It is a statement about a
single value; the next two theorems carry it through the dynamics and through
the storage format.

---

## 20.3 Task 1 — The Round-Trip and Its Error Bound

We test Theorem B.20.1 the way it will be used. First the code round-trip: every
byte must decode and re-encode to itself. Then the real-number sweep: we sample
$25{,}501$ points across $[0,1]$ — dense enough to land on every attainment
point $x = (2m+1)/510$, since these are the grid points with $k = (2m+1)\cdot
50$ — and record the worst error. The prediction, in prose before the
measurement, is that the code round-trip is exact for all $256$ codes and that
the worst real-number error is exactly $1/510 = 0.001960784\ldots$, the
theorem's bound, met rather than merely approached.

**Run this now.** Task 1 (`python3 code/gftac/ch20_tasks.py 1`):

```python
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
```

```text
Fixed-Point Representation (FP_ONE = 255):
  • Codes in [0, 255]:                    256
  • encode(decode(q)) == q for every q:   256/256
  • Max |decode(encode(x)) - x|:          1.960784e-03
  • Theoretical bound 1/(2*255):          1.960784e-03
  • Bound holds over 25,501 samples:      True
```

**Interpretation.** The code round-trip is exact for all $256$ codes: `255` is
a bijection on the byte range, so no code is lost. The measured worst error is
$1.960784\times 10^{-3}$, which is $1/510$ to six significant figures and equal
to the theoretical bound; the theorem is not merely satisfied, it is attained.
The final line compares the measured error against the bound with a $10^{-12}$
slack, because the subtraction `decode(encode(x)) - x` carries a rounding of
about one part in $10^{17}$ and the exact mathematical equality can sit one unit
in the last place on the wrong side. The slack is twelve orders of magnitude
below the error it certifies, so it changes no conclusion; it exists only so the
comparison tests the theorem rather than the floating-point unit. This is the
per-value guarantee the firmware runs on.

---

## 20.4 Ternary and 4-Bit Packing

The firmware's Flash holds two kinds of numbers: ternary projection weights in
$\{-1,0,+1\}$, and 4-bit conductance codes in $\{0,\dots,15\}$. Neither is
stored one-per-byte in the natural encoding, and the packing is where the
"1.58-bit" and "4-bit" claims become byte counts. The ternary scheme is the one
Chapter 4 built: five trits to a byte, because $3^5 = 243 \le 255$. The 4-bit
scheme is the obvious one: two codes to a byte, low nibble and high nibble.

> **Theorem B.20.3 (Packing rates and the MCU footprint law).**
> (a) The map $(t_0,\dots,t_4) \mapsto \sum_{k=0}^{4}(t_k+1)3^k$ is injective
> from $\{-1,0,+1\}^5$ into $[0,255]$, so $n$ trits occupy exactly
> $\lceil n/5\rceil$ bytes, an asymptotic cost of $8/5 = 1.6$ bits per
> parameter. (b) The map $(c_0,c_1) \mapsto c_0 + 16 c_1$ is injective from
> $\{0,\dots,15\}^2$ into $[0,255]$, so $n$ codes occupy exactly
> $\lceil n/2\rceil$ bytes, i.e. $4$ bits per parameter. (c) The Flash image of
> an $N$-node, $d$-dimensional model with $V$ test-vector bytes and $C$ bytes of
> program code occupies
> $$
> \mathrm{Flash} = N^2 + 2Nd + V + C \ \text{bytes},
> $$
> and the SRAM working set occupies
> $$
> \mathrm{SRAM} = N^2 + 2N + 2d \ \text{bytes}.
> $$

**Proof.** (a) Each digit $d_k = t_k + 1$ lies in $\{0,1,2\}$, and
$$
\sum_{k=0}^{4} d_k 3^k \le 2\sum_{k=0}^{4} 3^k = 2\cdot\frac{3^5-1}{3-1}
= 3^5 - 1 = 242 < 256,
$$
so the value is a byte. The base-3 expansion of an integer is unique, so the map
is injective; $\lceil n/5\rceil$ bytes therefore suffice and are necessary, and
the per-parameter cost is $8\lceil n/5\rceil/n \to 8/5$. (b) The value
$c_0 + 16c_1 \le 15 + 16\cdot 15 = 255$ is a byte, and the two nibbles are
recovered by $c_0 = b \bmod 16$, $c_1 = \lfloor b/16 \rfloor$, so the map is
injective; the cost is $8\lceil n/2\rceil/n \to 4$. (c) The lattice is an
$N\times N$ array of bytes, the input projection is $d \times N$ and the output
projection is $N \times d$, giving $N^2 + 2Nd$ bytes of weights; the test
vectors add $V$ and the code adds $C$. The SRAM statement counts the working
lattice $N^2$, the two state buffers $N + N$, and the `int16_t` latent vector
$2d$. $\blacksquare$

For the exported model, $N = 32$ and $d = 16$, so the weights alone are
$1024 + 512 + 512 = 2048$ bytes and the SRAM working set is
$1024 + 32 + 32 + 32 = 1120$ bytes. Task 2 demonstrates the packing on a small
tile; Task 4 applies the law to the whole deployment.

---

## 20.5 Task 2 — Quantizing and Packing a Weight Tile

We take an $8 \times 8$ tile of pseudo-random weights in $[-1.5,1.5]$,
deterministically drawn from a seeded generator, and run both quantizers over
it. The ternary quantizer is BitNet b1.58: the scale is $\gamma = \operatorname{mean}|w|$,
and each weight becomes $w_q = \operatorname{clip}(\operatorname{round}(w/\gamma),-1,1)$.
The 4-bit quantizer rescales the tile to $[0,1]$ and rounds to a code in
$\{0,\dots,15\}$. The prediction is that the ternary set is exactly
$\{-1,0,1\}$, that $64$ trits pack into $\lceil 64/5\rceil = 13$ bytes, that
$64$ codes pack into $32$ bytes, and that both packings round-trip exactly.

**Run this now.** Task 2 (`python3 code/gftac/ch20_tasks.py 2`):

```python
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
```

```text
Ternary (1.58-bit) Quantization of an 8x8 Weight Tile:
  • Parameters:                           64
  • Scale gamma = mean|w|:                0.776204
  • Quantized value set:                  [-1, 0, 1]
  • Trits packed 5-per-byte:              64 -> 13 bytes
  • Bits per parameter:                   1.6250
  • Ternary round-trip exact:             True
4-Bit Lattice Packing (2 codes per byte):
  • 4-bit codes:                          64 -> 32 bytes
  • Bits per parameter:                   4.0000
  • 4-bit round-trip exact:               True
```

**Interpretation.** The measured scale is $\gamma = 0.776204$ for this tile,
and the quantized value set is exactly $\{-1,0,1\}$ — the trit domain, with no
weight escaping the clip. The $64$ trits occupy $13$ bytes, matching
$\lceil 64/5\rceil$, and the measured cost is $13\cdot 8/64 = 1.6250$ bits per
parameter, the finite-size version of the theorem's $8/5 = 1.6$ limit (the
quarter-bit difference is the padding in the last byte). The 4-bit codes occupy
$32$ bytes at exactly $4.0000$ bits per parameter, and both round-trips are
exact. Notice what this task does not do: it does not claim the ternary
quantization is *accurate*, only that it is *lossless to store*. The accuracy
question is the next two sections.

---

## 20.6 The Discrete Diffusion Kernel and Its Error

We now prove the bound that makes the integer pipeline trustworthy against the
float pipeline. The exact operator is the row-stochastic $P[i][j] = W[i][j]/r_i$
with $r_i = \sum_k W[i][k]$. The fixed-point operator the firmware actually runs
is $\hat P[i][j] = \lfloor 255\,P[i][j]\rfloor/255$, and the state is carried on
the byte scale. The question is how far the integer state can drift from the
float state after $k$ steps.

> **Theorem B.20.2 (Accumulated diffusion error).**
> Let $W \in \mathbb{Z}_{\ge 0}^{N\times N}$, let $r_i = \sum_k W[i][k]$, and for
> $r_i > 0$ define $P[i][j] = W[i][j]/r_i$ and
> $\hat P[i][j] = \lfloor 255\,P[i][j]\rfloor / 255$. Fix a non-negative initial
> state $h^{(0)} \in \mathbb{R}_{\ge 0}^N$ and run
> $h^{(t+1)} = P h^{(t)}$ and $\hat h^{(t+1)} = \hat P \hat h^{(t)}$ from
> $\hat h^{(0)} = h^{(0)}$. Then for every $k \ge 0$,
> $$
> \bigl\| h^{(k)} - \hat h^{(k)} \bigr\|_\infty
> \le \frac{k}{255}\,\bigl\| h^{(0)} \bigr\|_1 .
> $$

**Proof.** Write $e^{(t)} = h^{(t)} - \hat h^{(t)}$ and $E = P - \hat P$. Both
$P$ and $\hat P$ are non-negative: $P$ by construction, and $\hat P$ because it
is a floor of a non-negative quantity. For a row with $r_i > 0$,
$\hat P[i][j] = \lfloor 255 P[i][j] \rfloor / 255$ satisfies
$$
0 \le E[i][j] = P[i][j] - \hat P[i][j]
= \frac{255 P[i][j] - \lfloor 255 P[i][j]\rfloor}{255} < \frac{1}{255},
$$
and for a row with $r_i = 0$ both operators are zero, so $E[i][j] = 0$.

Subtracting the two recurrences,
$$
e^{(t+1)} = P h^{(t)} - \hat P \hat h^{(t)}
= P\bigl(h^{(t)} - \hat h^{(t)}\bigr) + (P - \hat P)\hat h^{(t)}
= P e^{(t)} + E \hat h^{(t)} .
$$
Since $e^{(0)} = 0$, unrolling the recurrence gives
$$
e^{(k)} = \sum_{t=0}^{k-1} P^{\,k-1-t} E \hat h^{(t)} .
$$
Now $P$ is non-negative with every row sum equal to $1$ (when $r_i > 0$) or
$0$ (when $r_i = 0$), so for any vector $v$,
$\|P v\|_\infty = \max_i |\sum_j P[i][j] v_j| \le \max_i \sum_j P[i][j]\|v\|_\infty
\le \|v\|_\infty$. Taking sup-norms in the unrolled sum,
$$
\bigl\| e^{(k)} \bigr\|_\infty \le \sum_{t=0}^{k-1} \bigl\| E \hat h^{(t)} \bigr\|_\infty .
$$
For each $t$,
$$
\bigl(E \hat h^{(t)}\bigr)_i = \sum_j E[i][j]\hat h^{(t)}_j
\le \frac{1}{255}\sum_j \hat h^{(t)}_j = \frac{1}{255}\bigl\|\hat h^{(t)}\bigr\|_1 ,
$$
because the entries of $\hat h^{(t)}$ are non-negative. Finally $\hat P$ is
sub-stochastic — every row sum is at most $1$ — so mass never increases,
$\|\hat h^{(t+1)}\|_1 = \sum_i \sum_j \hat P[i][j]\hat h^{(t)}_j \le \sum_j
\hat h^{(t)}_j = \|\hat h^{(t)}\|_1$, hence
$\|\hat h^{(t)}\|_1 \le \|\hat h^{(0)}\|_1 = \|h^{(0)}\|_1$. Substituting,
$$
\bigl\| e^{(k)} \bigr\|_\infty
\le \sum_{t=0}^{k-1} \frac{1}{255}\bigl\| h^{(0)} \bigr\|_1
= \frac{k}{255}\,\bigl\| h^{(0)} \bigr\|_1 . \qquad \blacksquare
$$

Two readings of the bound are worth stating. In *raw* byte units, a normalized
point mass has $\|h^{(0)}\|_1 = 255$, so the sup-norm deviation after $k$ hops
is at most $k$ — one byte per hop, no more. In *probability* units, dividing by
$255$, the deviation is at most $k/255$: the accumulated error grows at most
linearly in the hop count, at the rate of the single-value truncation bound
$1/255$ established in the proof. This is the discrete analogue of the Grönwall
estimate of Chapter 3, and like it, it is a worst-case envelope. The next task
measures the actual deviation and finds it far inside the envelope.

---

## 20.7 Task 3 — Fixed Point Against Float

The test is the exported model's own vectors. We run the $32$-node UNO-32 model
on the nine two-hop and six three-hop queries from `mcu_weights.h`, once through
the integer `forward_inference` and once through the float reference, and compare
both the argmax decisions and the activation vectors. The prediction is that the
two pipelines agree on every argmax — the deviation bound of Theorem B.20.2
predicts at most a byte or two of drift, far below the margins that decide these
queries — and that the fixed-point accuracy reproduces the firmware's own
published figures: $8/9$ at two hops and $6/6$ at three.

**Run this now.** Task 3 (`python3 code/gftac/ch20_tasks.py 3`):

```python
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
```

```text
Fixed-Point vs Float Reference on the UNO-32 Model (15 exported vectors):
  • Fixed-point 2-hop accuracy:           8/9
  • Fixed-point 3-hop accuracy:           6/6
  • Float reference 2-hop accuracy:       8/9
  • Float reference 3-hop accuracy:       6/6
  • Argmax agreement (fixed vs float):    15/15
  • Max |h_fixed - h_float| (0..255):     1.0
```

**Interpretation.** The fixed-point pipeline scores $8/9$ at two hops and $6/6$
at three, exactly the firmware's published $88.89\%$ and $100\%$ — the integer
simulator is the firmware, not an approximation of it. The float reference
scores the same, and the two agree on all fifteen argmax decisions. The maximum
activation deviation is exactly $1.0$ in the raw $0$–$255$ scale: one byte, the
smallest non-zero discrepancy the representation admits, and comfortably inside
the $k$-byte envelope of Theorem B.20.2 for $k \in \{2,3\}$.

The single failure is worth naming, because it is not a numerical failure at all.
Query $20 \to 25$ fails in *both* pipelines, and so does the same query on the
physical board. Node $20$ has two outgoing arcs of equal conductance, to $21$ and
to $24$, so the diffusion splits its mass evenly and the two-hop mass lands
equally on nodes $22$ and $25$. The argmax must then pick one, and it picks the
lower index, $22$. This is a property of the single-label top-1 metric under
$1$-to-$N$ branching, not a defect the fixed point introduced; the float
reference reproduces it exactly, which is the honest reading.

---

## 20.8 The Memory Budget

The footprint law of Theorem B.20.3(c) is the reason the deployment is possible.
For the $32$-node model the weight image is $2048$ bytes of Flash and the working
set is $1120$ bytes of SRAM, and both sit inside the ATmega328P's $32{,}768$-byte
Flash and $2048$-byte SRAM with room to spare. The SRAM number is the one that
matters, because SRAM is the scarce resource: the working lattice alone consumes
half of it, and the remaining margin is what the call stack and the UART
interrupts must live in. The dossier's own audit reports $1120$ bytes of static
SRAM and calls the remaining $928$ bytes the "stack margin"; the accountant
reproduces that decomposition byte for byte.

---

## 20.9 Task 4 — Flash and SRAM

We instantiate the accountant on the UNO-32 model with the actual exported
vector count — $9 + 6 + 4 = 19$ source/target pairs — and the audited
$5{,}760$-byte program size. The prediction is a Flash total of
$1024 + 512 + 512 + 38 + 5760 = 7846$ bytes, about $23.94\%$ of Flash, and an
SRAM total of $1120$ bytes, about $54.69\%$, leaving $928$ bytes of stack.

**Run this now.** Task 4 (`python3 code/gftac/ch20_tasks.py 4`):

```python
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
```

```text
ATmega328P Flash Footprint (32 KB = 32,768 B):
  • W_LATTICE  (32x32, 1 B/cell):         1,024 B
  • W_IN       (16x32, 1 B/trit):         512 B
  • W_OUT      (32x16, 1 B/trit):         512 B
  • Test vectors (19 src/tgt pairs):      38 B
  • Program code & UART routines:         5,760 B
  • Total Flash:                          7,846 B (23.94% of 32,768 B)
ATmega328P SRAM Footprint (2 KB = 2,048 B):
  • W_ram      (32x32, 1 B/cell):         1,024 B
  • h_curr     (32, 1 B/node):            32 B
  • h_next     (32, 1 B/node):            32 B
  • h_latent   (16, 2 B/dim):             32 B
  • Total SRAM:                           1,120 B (54.69% of 2,048 B)
  • Remaining stack margin:               928 B (45.31%)
```

**Interpretation.** The Flash total is $7{,}846$ bytes, $23.94\%$ of the
$32$-kilobyte program memory, and the SRAM total is $1{,}120$ bytes, $54.69\%$
of the two kilobytes, leaving $928$ bytes — $45.31\%$ — for the stack. The
working lattice dominates both budgets: it is exactly half of SRAM, and it is
the single allocation that would prevent scaling past roughly $N = 45$ (where
$N^2 = 2025$ bytes exhausts SRAM). That ceiling is a property of the *dense*
lattice, and it is precisely the wall Chapter 18 removed: the sparse store is
what takes this architecture from thirty-two entities to a hundred thousand.

One honest accounting note. The published dossier's audit reports a Flash total
of $7{,}856$ bytes, ten more than the model above, because the dossier counts
$48$ bytes of test-query storage where we count the $38$ bytes of payload the
vectors actually occupy; the $10$-byte difference is the count macros and
banner-table constants that the accountant does not model. Both totals are
$24.0\%$ of Flash to the precision that matters, and the discrepancy is
reported rather than hidden.

---

## 20.10 Exporting PROGMEM Weights

An embedded model is not deployed until it is a header file. The exporter turns
an `MCUModel` into text in the exact shape of `mcu_weights.h`: a guard macro, an
`#include <avr/pgmspace.h>`, the two `#define`s for the dimensions, and three
`PROGMEM` arrays — `W_LATTICE`, `W_IN`, and `W_OUT` — each with a trailing
comment naming the row. The `PROGMEM` qualifier is what places the arrays in
Flash rather than SRAM; without it the compiler would copy them into the scarce
data memory at startup and the budget above would collapse.

The exporter is paired with a parser, and the parser is the self-check: whatever
the exporter writes, the parser must read back into arrays identical to the
model's. This is not ceremony. A C header is a serialization format like any
other, and the only way to know it is lossless is to round-trip it. We
demonstrate on a smaller, independent model — the $16$-node clinical-triage graph
hand-carved in `uno_nano_geoformer.ino`, with $d = 8$ and a deterministic ternary
projection — so that the round-trip is exercised on a model whose arrays we can
also read by eye.

---

## 20.11 Task 5 — The C-Header Round-Trip

We export the $16$-node triage model to a temporary header, read the file back
from disk, parse it, and compare every array. The prediction is that the header
is a few kilobytes of text, that the `#define`s carry the dimensions through,
and that all three arrays round-trip exactly.

**Run this now.** Task 5 (`python3 code/gftac/ch20_tasks.py 5`):

```python
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
```

```text
PROGMEM C-Header Export and Parse Round-Trip (16-node triage model):
  • Header size:                          3004 bytes
  • #define MCU_NUM_NODES:                16
  • #define MCU_D_MODEL:                  8
  • W_LATTICE row 0:                      { 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
  • W_LATTICE round-trips exactly:        True
  • W_IN round-trips exactly:             True
  • W_OUT round-trips exactly:            True
  • Full export/parse round-trip:         True
```

**Interpretation.** The emitted header is $3004$ bytes of text for a model whose
binary footprint is $256 + 128 + 128 = 512$ bytes of weights; the expansion is
the price of a human-readable, diffable format, and it is paid in the
development repo, not in the chip — the compiler packs the initializers into the
$512$ bytes the accountant predicted. The first lattice row reads
`{ 0, 8, 0, ... }`, the arc $0 \to 1$ of the triage graph with conductance $8$,
which is exactly the `carve_geodesic(0, 1, 8)` call in the `.ino`. All three
arrays round-trip exactly, and the `#define`s carry $N = 16$ and $d = 8$ through
the parse. The serializer is lossless on both the sparse lattice and the dense
ternary projections.

---

## 20.12 Adversarial Probes

A model that only works on its training queries is a lookup table. The firmware
ships four stress tests, and three of them are reproduced here in the simulator:
a false shortcut injection that writes a spurious high-conductance arc into SRAM,
a noise perturbation that nudges active conductances, and an isolation probe that
checks whether mass leaks into structurally disconnected components.

The shortcut test is the interesting one, because it exercises the dynamic
plasticity the architecture is for. Node $0$'s baseline two-hop query lands on
node $2$. We then write $W[0][29] = 15$ directly into the working lattice and
re-run at one hop. The prediction is that the injected arc dominates the row
sum, the transition probability to node $29$ becomes large, and the attractor
moves. This is not a failure of the model; it is the model working as designed —
increasing an edge weight contracts effective resistance and reroutes flow, which
is the in-context adaptation of Chapter 10 made visible on a microcontroller.

The noise test perturbs active conductances using the firmware's exact 16-bit
Galois LFSR, tap mask `0xB400`, iterating row-major and nudging a cell by $\pm 1$
when the low two bits of the register are zero. The prediction is that the graph's
signal-to-noise margin — every active arc carries $W = 10$, far above a $\pm 1$
nudge — preserves the top-1 rank order, so the two-hop accuracy is unchanged.
The isolation probe runs a query and reads the activation of a node that has no
path from the source; the prediction is zero activation, up to the integer
truncation the dossier already flagged.

---

## 20.13 Task 6 — Noise, Shortcuts, and Isolation

We run all three probes against the fixed-point simulator. The prediction is a
baseline prediction of node $2$ at $48\%$; a post-attack prediction of node $29$
at $56\%$; eight perturbed conductances with the two-hop accuracy unchanged at
$8/9$; and three of four isolation probes at exactly zero activation, with the
fourth showing the small truncation trace the dossier reports.

**Run this now.** Task 6 (`python3 code/gftac/ch20_tasks.py 6`):

```python
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
```

```text
Adversarial Probes in the Fixed-Point Simulator:
Adversary 1 — False Shortcut Injection:
  • Baseline query 0 (tau=2):             Node 2 (48% confidence)
  • Injected spurious edge:               0 ==> 29 (+15 conductance)
  • Post-attack query 0 (tau=1):          Node 29 (56% confidence)
  • Attractor warping confirmed:          True
Adversary 2 — 20% Analog Noise Perturbation:
  • Active conductances perturbed:        8
  • Post-noise 2-hop accuracy:            8/9
  • Rank order preserved:                 True
Adversary 3 — Disconnected Cluster Isolation:
  • Probe  0 ->  9 activation:   0/255 [SAFE]
  • Probe  0 -> 22 activation:   0/255 [SAFE]
  • Probe 20 ->  3 activation:   0/255 [SAFE]
  • Probe 26 -> 15 activation:   2/255 [LEAK]
  • Verified isolated:                    3/4
```

**Interpretation.** The shortcut injection moves the attractor exactly as
predicted: baseline node $2$ at $48\%$, post-attack node $29$ at $56\%$, matching
the dossier's "signal dynamically redirected to Node 29" at $56\%$ confidence.
The mechanism is Theorem B.20.2 seen from the other side — writing $W[0][29] =
15$ raises $P[0][29]$ above the incumbent, and the argmax follows the largest
transition probability. The noise adversary perturbs eight of the twenty-four
active conductances and leaves the two-hop accuracy at $8/9$: the $\pm 1$ nudge
is an order of magnitude below the $W = 10$ signal, so the rank order is
preserved. The isolation probes return three exact zeros and one trace of
$2/255$ on the $26 \to 15$ probe — not a structural connection, since nodes $26$
and $15$ share no path, but the integer division's truncation remainder bleeding
a couple of units across the accumulation. That is precisely the artifact the
dossier flags as needing clamping in a production safety system, and reproducing
it here is the point: the simulator is faithful enough to reproduce the
firmware's *imperfections*, not just its successes.

---

## 20.14 What the Silicon Measured (MODELED)

The numbers in this section are **MODELED** from the published benchmark
dossier; they were measured on a physical ATmega328P in an earlier audit and are
quoted here as the hardware context for the simulation. We did not attach a
board for this chapter, and we label them accordingly.

On silicon, the kernel ran the nine two-hop queries in an average of $6{,}606\
\mu s$ and the six three-hop queries in $8{,}908\ \mu s$, with accuracies of
$88.89\%$ and $100\%$ — the same accuracies the simulator reproduces. A single
diffusion step took $47.0\ \mu s$, so the $10{,}000$-cycle memory hammer
completed in $470.36\ \text{ms}$ with zero stack corruption. At a $5\text{ V}$
rail and roughly $10\ \text{mA}$ of core current the ATmega328P dissipates
$50\ \text{mW}$, so a two-hop query costs about $330\ \mu J$ of core energy. The
comparison the dossier draws — $50\ \text{mW}$ of operating power against a
$300\ \text{W}$ server GPU, a factor of $6{,}000$ — is a statement about power,
not energy per query: the eight-bit chip is thousands of times slower per query
because it serializes every multiply. The energy-per-query win belongs to the
analog memristor crossbar of the next chapter, not to the digital
microcontroller, and the dossier is careful to say so. We repeat the caution
here: the $443.9\ \text{pJ}$ figure that appears later in the book is a modeled
analog device, not this chip.

The scaling ceiling is likewise **MODELED** from the footprint law. At $N = 32$
the dense working lattice is $1{,}024$ bytes and the deployment is comfortable;
at $N = 45$ it is $2{,}025$ bytes and SRAM is exhausted; at $N = 64$ the lattice
alone is twice the chip's memory. The lesson is the one Chapter 18 taught in a
different register: the dense lattice scales as $N^2$, and the way past $N = 32$
on a microcontroller is either a larger chip or the sparse store, not a cleverer
integer trick.

---

## 20.15 Conclusion: The Kernel on the Die

We have taken the smallest honest object the architecture contains — the
discrete row-normalized diffusion $\mathbf{h}_{t+1} = \mathbf{h}_t\mathbf{P}$ —
and reproduced it exactly on the terms of the silicon it runs on. Theorem
B.20.1 fixed the per-value representation error at $1/510$ and showed the bound
is attained; Theorem B.20.2 carried it through the dynamics to a linear
accumulated bound of $k$ bytes after $k$ hops; Theorem B.20.3 turned the packing
into the footprint law. The measurements followed the proofs: the round-trip is
exact on all $256$ codes, the worst representation error is exactly $1/510$, the
ternary and 4-bit packings are lossless at $1.6250$ and $4.0000$ bits per
parameter, the fixed-point pipeline agrees with the float reference on all
fifteen exported vectors while reproducing the firmware's $8/9$ and $6/6$, the
deployment fits in $7{,}846$ bytes of Flash and $1{,}120$ bytes of SRAM, the
C-header round-trips exactly, and the three adversarial probes behave as the
firmware's dossier reports — including the $2/255$ truncation leak the simulator
reproduces rather than hides.

What we have *not* done is equally important to state. The Uno is not running
the foundation stack. It is running a directed graph diffusion kernel with
ternary I/O and eight-bit dynamics, and the only reason it deserves the
GeoFormer name is that the kernel is the same operator the larger model relaxes,
compiled down until it fits. That is the honest claim, and it is a real one:
the geometry survives the trip to the cheapest silicon in the book.

The next chapter takes the other road. Instead of shrinking the digital
arithmetic until it fits an eight-bit register, it asks what happens when the
arithmetic stops being digital at all — when the conductance lattice is a
physical crossbar of memristive devices and the multiply is Ohm's law. That is a
**MODELED** chapter through and through, and it begins where this one ends: with
a conductance, a voltage, and the energy it takes to let physics do the
multiply.
