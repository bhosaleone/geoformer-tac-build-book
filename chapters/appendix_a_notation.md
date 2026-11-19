# Appendix A — Notation and Symbols

The book reuses a small alphabet in many chapters, and a few symbols carry a
different meaning in different parts. This appendix fixes each symbol once, says
where it is introduced, and — where a symbol is overloaded — says so explicitly,
because an overloaded symbol silently read the wrong way is the most common way
to misread a proof.

The symbols fall into four families. The **geometry** family describes the
metric lattice and its linear algebra. The **dynamics** family describes
propagation, relaxation, and sharpening. The **quantization** family describes
how continuous numbers become storable ones. The **evidence** family describes
how claims are labeled and verified.

### Geometry

| Symbol | Meaning | Introduced |
|---|---|---|
| $N$, $n$ | number of entities (nodes) in the lattice | Ch 2 |
| $W$ | conductance matrix, $W \\in \\mathbb{R}^{n \\times n}$, $W \\ge 0$, $\\operatorname{diag}(W) = 0$ | Ch 2 |
| $W_{ij}$ | conductance of the directed edge $i \\to j$ | Ch 2 |
| $w_{\\max}$ | maximum conductance of a level (default $1.0$) | Ch 2 |
| $b$ | quantization bit-width of the lattice (default $4$) | Ch 2 |
| $\\Delta w$ | quantization step, $\\Delta w = w_{\\max} / (2^{b} - 1)$ | Ch 2 |
| $\\varepsilon$ | epsilon guard (default $10^{-8}$): the additive term that keeps resistance finite and quantization stable | Ch 2 |
| $d_{ij}$ | geodesic resistance of edge $i \\to j$, $d_{ij} = 1/(W_{ij} + \\varepsilon)$ | Ch 2 |
| $D$ | resistance matrix, $D = 1/(W + \\varepsilon)$ elementwise | Ch 2 |
| $L$ | graph Laplacian, $L = \\operatorname{diag}(W \\mathbf{1}) - W$ | Ch 2 |
| $L_{\\text{dir}}$, $P_{\\text{dir}}$ | directed Laplacian and directed (row-normalized) transition kernel | Ch 5 |
| $\\lambda_2$ | Fiedler value: the second-smallest eigenvalue of $L$; the algebraic connectivity | Ch 2 |
| $\\lambda_{\\max}$ | spectral radius of the Q-TAC relaxation operator | Ch 7 |

### Dynamics

| Symbol | Meaning | Introduced |
|---|---|---|
| $z$ | propagation state vector over the $N$ entities | Ch 3 |
| $h$ | hidden activation state (model tensors) | Ch 9 |
| $\\gamma$ | soliton sharpening exponent applied after each propagation step | Ch 16 |
| $\\gamma_{\\text{crit}}$ | critical sharpening exponent separating sub- and super-critical propagation | Ch 16 |
| $\\rho$ | contraction ratio of the sharpened propagation operator | Ch 16 |
| $\\tau$ | relaxation step count, or halting time of an autonomous propagation | Ch 3, Ch 7 |
| $\\tau^{*}$ | emergent halting depth: the hop at which a propagation halts of its own accord | Ch 17 |
| $\\alpha_0$ | signal-to-decoy ratio (path separability); also the name of the control parameter studied in Ch 12 | Ch 12, Ch 16 |
| $d_{\\text{avg}}$ | average degree, i.e. the number of decoy branches per highway node | Ch 16 |
| $\\kappa$ | **overloaded:** the condition number $\\lambda_{\\max}/\\lambda_{\\min}$ of the relaxation operator (Ch 7), *and* the trajectory-closure index in $[0,1]$ (Ch 10) | Ch 7, Ch 10 |

### Quantization

| Symbol | Meaning | Introduced |
|---|---|---|
| $\\gamma$ (scale) | **overloaded:** the per-tensor scale $\\operatorname{mean}|w| + \\varepsilon$ of BitNet b1.58 quantization | Ch 4 |
| trit | a value in $\\{-1, 0, +1\\}$; five pack losslessly into one byte | Ch 4 |
| $K(S)$ | Kolmogorov description length of the S₅ Cayley projector $S$ | Ch 8 |
| $S_5$ | the symmetric group on five symbols; the projector's symmetry group | Ch 8 |
| `FP_ONE` | fixed-point scale of the MCU export, $255$ (an 0.8 fixed-point, *not* 16-bit Q8.8) | Ch 20 |

### Evidence

| Symbol | Meaning | Introduced |
|---|---|---|
| $B.c.n$ | a Book theorem: the $n$-th theorem of Chapter $c$ | Ch 0 |
| Q-1, Q-2, Q-3 | the research program's theorem series that the book's $B.2.1$, $B.2.3$, $B.16.1$ cross-reference | Ch 0 |
| MEASURED | a number computed or simulated on this machine, deterministic under seed | Ch 0 |
| MODELED | a number produced by an analytical or physical model, not by silicon | Ch 0 |
| RESEARCH DEMO | an illustrative application carrying no domain claim | Ch 0 |
| `«…»` | a volatile span in a transcript: a run-dependent value the checker matches by pattern | Ch 0 |
| pJ | picojoule, $10^{-12}$ J | Ch 21 |

Two overloads are worth stating twice. **$\\gamma$** is the sharpening exponent
in the dynamics family and the per-tensor quantization scale in the
quantization family; context always disambiguates, and the two never appear in
the same equation. **$\\kappa$** is the condition number in Chapter 7 and the
trajectory-closure index in Chapter 10; the condition number is always written
with the operator it belongs to, and the closure index always with a trajectory.
**$\\varepsilon$** is always the $10^{-8}$ guard unless a chapter explicitly
redefines it, which none does.
