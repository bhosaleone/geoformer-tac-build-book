<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->
# Chapter 23 — What the Z3 Certificates Do and Don't Say

The program ships certificates. In `TAC/src/tac/proofs/z3_verifier.py` five
theorems are discharged by Microsoft Z3; in
`geoformer/tests/test_z3_invariants.py` three more — the geodesic contraction
Q-1, the Fiedler preservation Q-2, the soliton amplitude bound Q-3 — are
discharged the same way; and the quantization monograph in
`geoformer/z3_proofs_quantization/` ships a parallel certificate over six
quantized-propagation theorems.
These are not marketing. Each is a real machine-checked refutation, produced by
a real solver, over a real decision procedure. They are also *narrower* than
their titles suggest, and this chapter is the audit that says exactly how much
narrower.

There are two failure modes when a project leans on formal certificates, and
they pull in opposite directions. The first is to treat a certificate as
decoration — to print the word "verified" and stop reading. The second is to
mistrust the whole enterprise because a certificate did not prove something it
never claimed. The honest path is in between, and it is mechanical: open the
certificate, read its premises, identify the structure it quantifies over,
and then ask separately whether the code that runs lives in that structure.
The answer, for every certificate in this repository, is that it lives in the
real numbers, while the code lives in binary floating point. That gap is not a
flaw in the proofs. It is the boundary of what they say, and stating it
precisely is the work of §23.2.

The chapter then does something the certificates do not do on their own. A
universally quantified proof over $\mathbb{R}$ and a finite set of float
evaluations are complementary evidence, and we can supply the second kind
mechanically. We sample graph ensembles and parameter draws, evaluate each
invariant's *precondition ⇒ conclusion* numerically, run the same check in FP32
and FP64, and report where the two precisions agree and where they must not.
This is **property-based randomized verification**, and it turns a certificate
that says "true for all reals" into a certificate that says "true for all reals,
*and* true on these $N$ sampled floats, to within a stated confidence."

Everything in this chapter is **MEASURED** on this machine and pinned by
`tests/test_ch23_z3_certificates.py` and `tests/test_ch23_tasks.py`. The three
theorems the chapter proves itself — the soundness of the negated-goal method
(B.23.1), the floating-point limitation that method cannot see (B.23.2), and the
statistical meaning of a clean sample (B.23.3) — are proved in full below.

---

## 23.1 The Shape of a Certificate

A proof of an implication $T \Rightarrow G$ does not begin by asserting $T$ and
$G$. It begins by asserting $T$ and $\neg G$, and showing that the combination
is impossible. This is the *negated-goal* method, and it is the shape every Z3
certificate in the repository takes. The solver is handed a query that asks, in
effect: *can you exhibit a world in which the premises hold and the conclusion
fails?* If the answer is `unsat` — unsatisfiable — then no such world exists,
and the implication holds.

The method is sound, and the soundness is worth stating and proving because it
is the entire license for reading the word "verified" in the repository.

> **Theorem B.23.1 (Soundness of the negated-goal certificate).**
> Let $T$ and $G$ be formulas in a first-order language interpreted in a
> structure $\mathfrak{M}$, and suppose a sound solver reports that the
> conjunction $T \wedge \neg G$ is unsatisfiable. Then every model of $T$
> satisfies $G$; that is, $\mathfrak{M} \models T \Rightarrow G$.

**Proof.** Suppose, for contradiction, that there is a model
$\mathfrak{N}$ with $\mathfrak{N} \models T$ and
$\mathfrak{N} \not\models G$. Interpreting $\neg G$ in $\mathfrak{N}$ then gives
$\mathfrak{N} \models \neg G$, so
$\mathfrak{N} \models T \wedge \neg G$. The conjunction is therefore
satisfiable — a model exists — which contradicts the solver's report that it is
unsatisfiable. Hence no model of $T$ falsifies $G$, and every model of $T$
satisfies $G$. $\blacksquare$

Two hypotheses carry the theorem, and neither is checked by the certificate
itself. The first is that the solver is *sound*: every `unsat` it prints is
genuinely unsatisfiable. For the arithmetic these certificates use — integer
and real polynomial constraints with comparisons — Z3 decides the question with
complete procedures (Presburger arithmetic and quantifier-free real closed
fields), so a sound `unsat` is exactly what it prints. The second hypothesis is
that the *encoding* is faithful: the Z3 formula must denote the same relation
the mathematics denotes. A proof is only as trustworthy as its translation, and
the translation is a few lines of Python that a reader can audit — which is why
Task 3 prints the premises of every certificate rather than hiding them behind a
function call.

The repository's certificates come in two shapes, and the difference is the
first thing an honest reading must notice. Four of the five TAC theorems and all
three GeoFormer invariants are genuine negated-goal refutations: assert the
negation, get `unsat`. But TAC-5 is different. The shipped
`verify_theorem_5_s5_representation_invariant` asserts the arithmetic equalities
$|S_5| = 120$, $\sum_i d_i^2 = 120$, and $120 + 4^2 + 1 = 137$, and then checks
for `sat`. That is a *consistency witness* — it exhibits a model of the stated
facts — not a refutation of the negation. It is true and useful, but it is a
different kind of object, and calling it a proof of a universally quantified
statement would be an overreach. In this chapter we record what the repository
does, and then we perform the proof-shaped upgrade: assert
$\neg(120{=}120 \wedge 120{=}120 \wedge 137{=}137)$ and obtain `unsat`. Both
columns appear in the task below, because the point is not to replace the
shipped certificate but to read it accurately.

We begin by reproducing the five TAC certificates and the three GeoFormer
invariants, recording for each the negated-goal status and the shipped method.

**Run this now.** Task 1 (`python3 code/gftac/ch23_tasks.py 1`):

```python
records, elapsed = _time_proofs(run_tac_theorems)

print("TAC Z3 Certificate — five theorems:")
for r in records:
    print(
        f"  • {r.proof_id}  {r.name:<36} status={r.status:<5} "
        f"proved={str(r.proved):<5} repo={r.repo_method}"
    )
print(f"  • All five goals proved:  {all(r.proved for r in records)}")
print(f"  • Z3 wall time:           {elapsed:.4f} s (run-dependent)")
```

```text
TAC Z3 Certificate — five theorems:
  • TAC-1  Lyapunov free-energy dissipation     status=unsat proved=True  repo=negated-goal
  • TAC-2  Dimension-independent relaxation     status=unsat proved=True  repo=negated-goal
  • TAC-3  Bistable double-well soft-spin       status=unsat proved=True  repo=negated-goal
  • TAC-4  Exact-cover ground state             status=unsat proved=True  repo=negated-goal
  • TAC-5  S5 representation invariant (137)    status=unsat proved=True  repo=SAT-witness
  • All five goals proved:  True
  • Z3 wall time:           «0.0375» s (run-dependent)
```

**Interpretation.** Every row reads `status=unsat`: each negated goal is
unsatisfiable, so each implication holds over the reals. The `repo=` column is
the disclosure. Four rows say `negated-goal`, meaning the shipped verifier and
our upgrade agree on the shape of the argument. The fifth row says
`SAT-witness`, and that is the honesty line: TAC-5 as shipped asks the solver for
a model, not for a refutation. Our `status=unsat` on that row is the
proof-shaped upgrade of §23.1, and it is a genuine refutation of the arithmetic
identity. The wall time is reported wrapped in «...» because it is a property of
the machine, not of the proof; the statuses beside it are exact and do not
vary. This is **MEASURED**, asserted by `test_tac_theorems_all_unsat` and
`test_tac5_shipped_as_sat_witness_and_upgraded`.

**Run this now.** Task 2 (`python3 code/gftac/ch23_tasks.py 2`):

```python
records = run_geoformer_theorems()

print("GeoFormer Z3 Certificate — Q-series invariants:")
for r in records:
    print(
        f"  • {r.proof_id}  {r.name:<36} status={r.status:<5} "
        f"proved={str(r.proved):<5} repo={r.repo_method}"
    )
print(f"  • All three goals proved:  {all(r.proved for r in records)}")
```

```text
GeoFormer Z3 Certificate — Q-series invariants:
  • Q-1  Discrete geodesic contraction        status=unsat proved=True  repo=negated-goal
  • Q-2  Quantized Fiedler preservation       status=unsat proved=True  repo=negated-goal
  • Q-3  Soliton amplitude lower bound        status=unsat proved=True  repo=negated-goal
  • All three goals proved:  True
```

**Interpretation.** The three GeoFormer invariants are all refutations of their
negations. Q-1 asserts that a positive conductance increment strictly shrinks
the edge distance $d = 1/(W + \epsilon)$; Q-2 asserts that a spectral gap pinned
at $N \cdot \Delta w$ cannot be non-positive; Q-3 asserts that the contraction
ratio $\rho = d_{\text{avg}}/\alpha_0^{\gamma}$ cannot reach one. Each premise
is stated explicitly in the source, and each `unsat` is the machine's refusal to
find a counterexample. The chapter's unit tests
(`test_geoformer_theorems_all_unsat`) assert all three. What these rows do *not*
say is the subject of the next section.

---

## 23.2 The Bounded Domain

Read the variables in those certificates. Every one is a Z3 `Real` or `Int`.
The solver reasons in exact arithmetic — in the rationals and integers, with no
rounding, no overflow, and no notion of a machine word. That is what makes the
`unsat` results trustworthy: the certificate is a theorem about
$\mathbb{R}$, and in $\mathbb{R}$ the arithmetic laws that the proofs use are
exactly true.

The code that runs, however, lives in binary floating point. The set of
binary32 numbers is not a field. Addition is not associative. The reciprocal is
not exact. An inequality that holds for all real $W$ and all real
$\delta_w > 0$ can fail for the specific floats the program actually holds,
because the program's $\delta_w$ can be smaller than the distance between
consecutive floats near $W$. This is not a subtle philosophical point; it is a
measurable, reproducible failure, and we can locate it exactly.

> **Theorem B.23.2 (Binary32 absorbs a sub-half-spacing increment).**
> Let $W \ge 1$ be a normal binary32 float with $W \in [2^e, 2^{e+1})$, and let
> $0 < \delta \le 2^{-25} W$. Then the round-to-nearest binary32 sum satisfies
> $\operatorname{fl}(W + \delta) = W$. Consequently, evaluating the geodesic
> distance as $d(x) = \operatorname{fl}\!\big(1/(x + \epsilon)\big)$ with the
> left-to-right order $\operatorname{fl}\big(\operatorname{fl}(W + \delta) +
> \epsilon\big)$, we have $\operatorname{fl}(W + \delta + \epsilon) =
> \operatorname{fl}(W + \epsilon)$, hence $d(W + \delta) = d(W)$, and the strict
> contraction $d(W + \delta) < d(W)$ asserted by Q-1 fails in binary32.

**Proof.** In the interval $[2^e, 2^{e+1})$ the binary32 numbers are spaced by
$2^{e-23}$. Round-to-nearest maps a real number $x$ to its neighbour $W$ when
$|x - W|$ is strictly below half the spacing, $2^{e-24}$. Since $W < 2^{e+1}$,

$$
2^{-25} W < 2^{-25} \cdot 2^{e+1} = 2^{e-24},
$$

so $0 < \delta \le 2^{-25} W < 2^{e-24}$ and therefore
$\operatorname{fl}(W + \delta) = W$. Now the sum $W + \delta + \epsilon$ is
evaluated left to right as $\operatorname{fl}\big(\operatorname{fl}(W +
\delta) + \epsilon\big) = \operatorname{fl}(W + \epsilon)$, because the inner
sum rounded to exactly $W$. The two arguments to the outer reciprocal are
therefore the same float, so the reciprocal rounding, being a function of its
input, produces the same value: $d(W + \delta) = d(W)$. The strict inequality
$d(W + \delta) < d(W)$ is then false, since equality is not less-than.
$\blacksquare$

Two remarks keep the theorem honest. First, it is the *strict* inequality that
breaks. The non-strict statement $d(W + \delta) \le d(W)$ survives in binary32,
because equality is permitted. If Q-1 had been stated with $\le$, the float code
would satisfy it; because it is stated with $<$, the float code can violate it.
Second, the theorem does not say the certificate is wrong. The certificate
proved the real statement, and the real statement is true. The theorem locates
the exact boundary the certificate is silent about: increments below the
half-spacing of the float grid. For binary64 the unit roundoff is $2^{-53}$, so a
$\delta = 2^{-25} W$ sits far above the half-spacing $2^{e-54}$ and contraction
proceeds normally. The asymmetry between the two precisions is not an accident
of this example; it is the width of the grid.

The disclosure is the inventory of exactly these silences. For each of the eight
certificates it prints the premises the proof assumes, the conclusion it
reaches, and the things it does not cover.

**Run this now.** Task 3 (`python3 code/gftac/ch23_tasks.py 3`):

```python
records = bounded_domain_disclosure()

print("Bounded-Domain Disclosure — what each certificate assumes and omits:")
for r in records:
    print(f"  [{r.proof_id}] {r.name}")
    print(f"      domain:   {r.domain}")
    print(f"      assumes:  {'; '.join(r.premises)}")
    print(f"      proves:   {r.conclusion}")
    print(f"      omits:    {'; '.join(r.omits)}")
print("")
print(
    f"  Disclosure summary: {len(records)} certificates over the real domain; "
    f"0 cover floating-point rounding."
)
```

```text
Bounded-Domain Disclosure — what each certificate assumes and omits:
  [TAC-1] Lyapunov free-energy dissipation
      domain:   real arithmetic (Z3 Real/Int), exact
      assumes:  kappa > 0; grad F = (g0, g1, g2) in R^3
      proves:   dF/dt = -kappa * ||grad F||^2 <= 0, with equality iff grad F = 0
      omits:    the discretized Euler-Maruyama update that actually runs; floating-point rounding of the product -kappa * ||grad F||^2; gradient dimensions N > 3
  [TAC-2] Dimension-independent relaxation
      domain:   real arithmetic (Z3 Real/Int), exact
      assumes:  kappa > 0; lambda_min > 0; ||s||^2 > 0; dissipation >= 2 * kappa * lambda_min * ||s||^2
      proves:   dissipation > 0; relaxation timescale governed only by lambda_min
      omits:    A >= lambda_min * I is taken as a premise, not derived; finite-precision evaluation of the quadratic form
  [TAC-3] Bistable double-well soft-spin
      domain:   real arithmetic (Z3 Real/Int), exact
      assumes:  s in R; V(s) = 1/4 * (s^2 - 1)^2
      proves:   stationary points exactly {-1, 0, +1}; +/-1 stable, 0 unstable; confinement for |s| > 1
      omits:    Langevin noise (the stochastic term of the relaxer); multi-dimensional soft-spin fields
  [TAC-4] Exact-cover ground state
      domain:   real arithmetic (Z3 Real/Int), exact
      assumes:  beta > 0; A = [[1], [1]] (a 2-set, 1-element micro-instance)
      proves:   F(s) = 0 iff (s0, s1) in {(1, 0), (0, 1)}
      omits:    general set systems: only the 2x1 instance is encoded; floating-point evaluation of F
  [TAC-5] S5 representation invariant (137)
      domain:   real arithmetic (Z3 Real/Int), exact
      assumes:  |S5| = 5! = 120; irrep dimensions [1, 1, 4, 4, 5, 5, 6]
      proves:   sum(d_i^2) = 120 and 120 + 4^2 + 1 = 137
      omits:    that the listed integers are the actual irreps (taken from the classification); meaning for the floating-point Cayley projector
  [Q-1] Discrete geodesic contraction
      domain:   real arithmetic (Z3 Real/Int), exact
      assumes:  W_t >= delta_w; delta_w > 0; eps > 0
      proves:   1/(W_t + delta_w + eps) < 1/(W_t + eps): the edge distance strictly contracts
      omits:    floating-point rounding: in FP32 a small delta_w is absorbed by W_t; the graph-level path distance (this certificate is about one edge)
  [Q-2] Quantized Fiedler preservation
      domain:   real arithmetic (Z3 Real/Int), exact
      assumes:  N >= 4; delta_w > 0; lambda_2_min == N * delta_w
      proves:   lambda_2_min > 0: the spectral gap never collapses
      omits:    the graph-theoretic derivation lambda_2(L) >= N * min W_ij (taken here as the definition); the FP32/FP64 Laplacian eigensolve
  [Q-3] Soliton amplitude lower bound
      domain:   real arithmetic (Z3 Real/Int), exact
      assumes:  alpha_0 >= 2; 1 <= d_avg <= 4; alpha_0^gamma >= alpha_0^2; alpha_0^gamma > d_avg
      proves:   rho = d_avg / alpha_0^gamma < 1: the distractor ratio contracts
      omits:    that the 32-hop benchmark actually satisfies these premises; the induction over hops: only the per-hop ratio is bounded

  Disclosure summary: 8 certificates over the real domain; 0 cover floating-point rounding.
```

**Interpretation.** The summary line is the chapter's thesis in one sentence:
eight certificates, all over the reals, none covering floating-point rounding.
Reading the rows tells you *which* premise each proof leans on. TAC-2 assumes
$A \succeq \lambda_{\min} I$ rather than deriving it, so the certificate proves
the consequence of the spectral gap, not the gap's existence. TAC-4 encodes a
two-set, one-element instance, so it proves the exact-cover equivalence on that
instance, not for arbitrary set systems. Q-2 takes
$\lambda_2 \ge N\,\Delta w$ as a premise and proves positivity from it, so the
graph-theoretic bound itself is not inside the certificate. Q-3 bounds the
per-hop contraction ratio but does not perform the induction over hops. None of
these are defects; each is a boundary. The rows that say "omits: floating-point"
are the ones Theorem B.23.2 makes concrete. This is **MEASURED**, asserted by
`test_disclosure_covers_all_and_none_is_float`.

---

## 23.3 Property-Based Verification of Q-1

The certificate is a theorem about $\mathbb{R}$; the program is a finite set of
float evaluations. The complement of a universal proof is not another universal
proof — it is evidence, gathered by sampling. We can, mechanically, draw a large
ensemble from the invariant's *domain*, evaluate the implication
precondition ⇒ conclusion on each draw, and count the contradictions. If the
implementation were wrong anywhere in the sampled region, the count would not be
zero. This is **property-based randomized verification**, and its design choices
are exactly the confounds it removes.

We draw conductance increments from the domain the certificate quantifies over.
Each edge samples $W \ge 1$ and $\delta_w \in (0, W]$, so the precondition
$W_t \ge \delta_w > 0$ holds by construction; $\epsilon = 10^{-8}$ is positive as
required. For each edge we compute $d_{\text{before}} = 1/(W + \epsilon)$ and
$d_{\text{after}} = 1/(W + \delta_w + \epsilon)$ twice — once with binary64
tensors, once with the same values cast down to binary32 — and test the strict
contraction $d_{\text{after}} < d_{\text{before}}$. The two verdicts are
compared edge by edge. The ensemble size is $200 \times 64 = 12{,}800$ edges,
and the seed is fixed at 23, so the run is byte-for-byte reproducible.

Then we add the probe that the certificate cannot see. A second ensemble draws
$\delta_w = 2^{-25} W$ — strictly below the binary32 half-spacing, exactly the
regime of Theorem B.23.2 — and runs the same two checks. The prediction is
precise: FP32 should fail on every edge, FP64 should pass on every edge, and the
two precisions should disagree.

**Run this now.** Task 4 (`python3 code/gftac/ch23_tasks.py 4`):

```python
res = property_check_q1()

pct32 = 100.0 * res["fp32_passed"] / res["total"]
pct64 = 100.0 * res["fp64_passed"] / res["total"]
print("Property-Based Check of Q-1 (discrete geodesic contraction):")
print(
    f"  Ensembles:                 {res['num_graphs']} random graphs × "
    f"{res['edges_per_graph']} edges = {res['total']} edges"
)
print("  Domain:                    W_t >= delta_w > 0, eps > 0")
print(f"  FP32 contractions held:    {res['fp32_passed']}/{res['total']}  ({pct32:.2f}%)")
print(f"  FP64 contractions held:    {res['fp64_passed']}/{res['total']}  ({pct64:.2f}%)")
print(f"  FP32 == FP64 on all edges: {res['agreement']}")
print("  Bounded-domain probe (delta_w = 2^-25 * W, W >= 1):")
print(
    f"    FP32 contractions held:  {res['boundary_fp32_passed']}/{res['boundary_total']}"
)
print(
    f"    FP64 contractions held:  {res['boundary_fp64_passed']}/{res['boundary_total']}"
)
print(f"    FP32 == FP64:            {res['boundary_agreement']}")
```

```text
Property-Based Check of Q-1 (discrete geodesic contraction):
  Ensembles:                 200 random graphs × 64 edges = 12800 edges
  Domain:                    W_t >= delta_w > 0, eps > 0
  FP32 contractions held:    12800/12800  (100.00%)
  FP64 contractions held:    12800/12800  (100.00%)
  FP32 == FP64 on all edges: True
  Bounded-domain probe (delta_w = 2^-25 * W, W >= 1):
    FP32 contractions held:  0/50
    FP64 contractions held:  50/50
    FP32 == FP64:            False
```

**Interpretation.** In the domain the certificate quantifies over, both
precisions contract on every one of the $12{,}800$ edges, and they agree edge by
edge. That is the property-based confirmation of Q-1: not a proof, but a
falsifiable check that came back clean, on an ensemble large enough to catch a
systematic implementation error. The probe is the other half of the story. On
increments below the binary32 half-spacing, FP32 holds $0$ of $50$ contractions
— the increment is absorbed, exactly as Theorem B.23.2 predicts — while FP64
holds all $50$, and the precisions disagree on every edge. The certificate was
never wrong; it was silent about this regime, and now the silence has a number
attached. This is **MEASURED**, asserted by
`test_property_q1_holds_and_agrees`,
`test_property_q1_boundary_probe_shows_fp32_gap`, and the direct witness
`test_b23_2_fp32_absorbs_sub_half_spacing_increment`.

---

## 23.4 Property-Based Verification of Q-2

The second invariant is the Fiedler floor of Chapter 2, Theorem B.2.3: if every
unordered pair of nodes carries symmetrized conductance at least $\Delta w$,
then the algebraic connectivity of the symmetrized Laplacian satisfies
$\lambda_2(L) \ge N\,\Delta w$. The shipped certificate Q-2 takes the inequality
$\lambda_2 \ge N\,\Delta w$ as a *premise* and proves the consequence
$\lambda_2 > 0$. The graph-theoretic bound — the part that says the premise is
achievable on a real lattice — is not in the certificate. That is precisely the
part the property check can supply.

We sample dense lattices: each of $N = 12$ nodes carries a directed conductance
to every other node, drawn uniformly from $[0.5, 8.0]$, and the symmetrized
adjacency is $A = W + W^{\mathsf T}$. The minimum off-diagonal entry of $A$ is
the lattice's $\Delta w$, so the hypothesis of B.2.3 holds by construction. We
then form the Laplacian $L = D - A$, take its second-smallest eigenvalue
$\lambda_2$, and test $\lambda_2 \ge N\,\Delta w$ — once in binary64 and once in
binary32. The check is not that the certificate is true (it is, by B.23.1); it is
that the *premise* is satisfiable and the conclusion survives, on a real
ensemble, in both precisions.

**Run this now.** Task 5 (`python3 code/gftac/ch23_tasks.py 5`):

```python
res = property_check_q2()

print("Property-Based Check of Q-2 (Fiedler floor lambda_2 >= N * delta_w):")
print(
    f"  Ensembles:                 {res['num_lattices']} random dense lattices "
    f"of N = {res['n']} nodes"
)
print("  Domain:                    every off-diagonal symmetrized conductance >= delta_w > 0")
print(f"  FP32 bound held:           {res['fp32_passed']}/{res['num_lattices']}")
print(f"  FP64 bound held:           {res['fp64_passed']}/{res['num_lattices']}")
print(f"  FP32 == FP64 on all lattices: {res['agreement']}")
print(f"  Minimum margin (lambda_2 - N*delta_w): {res['min_margin']:.3f}")
```

```text
Property-Based Check of Q-2 (Fiedler floor lambda_2 >= N * delta_w):
  Ensembles:                 50 random dense lattices of N = 12 nodes
  Domain:                    every off-diagonal symmetrized conductance >= delta_w > 0
  FP32 bound held:           50/50
  FP64 bound held:           50/50
  FP32 == FP64 on all lattices: True
  Minimum margin (lambda_2 - N*delta_w): 36.660
```

**Interpretation.** The bound holds on all $50$ lattices in both precisions,
and the two precisions agree lattice by lattice. The minimum margin —
$\lambda_2 - N\,\Delta w$, the smallest slack observed — is $36.660$, far larger
than any plausible floating-point perturbation of the eigensolve, which is why
the two precisions do not disagree here even though they can in principle. The
margin is also a measure of how much room the architecture has: the spectral gap
is not grazing its floor. Unlike the Q-1 probe, there is no adversarial
counterpart to exhibit, because the conclusion here is a non-strict inequality
with positive slack; the boundary of Theorem B.23.2 is specific to strict
inequalities and to increments below the grid. This is **MEASURED**, asserted by
`test_property_q2_holds_and_agrees`.

---

## 23.5 From Samples to a Certificate: The Statistical Bound

A clean sample is evidence, and evidence can be quantified. If the predicate
$\varphi$ is false with probability $p$ over the sampling distribution, then
observing *zero* failures in $N$ independent draws is unlikely when $p$ is
large. Inverting that observation gives a distribution-free upper bound on $p$,
and the bound is the honest way to say what a property-based pass buys.

> **Theorem B.23.3 (Zero-failure statistical bound).**
> Let $\varphi$ be a predicate on a domain $D$, let
> $p = \Pr_{x \sim D}[\neg\varphi(x)]$ be the failure probability under a fixed
> distribution on $D$, and let $F$ be the number of failures among $N$
> independent draws. Then for every $\epsilon \in (0,1)$,
> $\Pr[F = 0] \le (1-\epsilon)^N$ whenever $p > \epsilon$. Consequently, if zero
> failures are observed, then with confidence at least $1 - \delta$,
> $p \le \ln(1/\delta)/N$.

**Proof.** If $p > \epsilon$ then each draw succeeds with probability
$1 - p < 1 - \epsilon$, and the draws are independent, so
$\Pr[F = 0] = (1-p)^N < (1-\epsilon)^N$. Thus rejecting the hypothesis
$p > \epsilon$ whenever $F = 0$ has type-I error at most $(1-\epsilon)^N$.
Choosing $\epsilon = \ln(1/\delta)/N$ makes
$(1-\epsilon)^N \le e^{-N\epsilon} = \delta$, because $1 - t \le e^{-t}$ for all
real $t$. Hence with confidence $1 - \delta$ the failure probability obeys
$p \le \ln(1/\delta)/N$. $\blacksquare$

Insert the Q-1 ensemble: $N = 12{,}800$ and $\delta = 0.01$ give
$\epsilon = \ln(100)/12{,}800 \approx 3.6 \times 10^{-4}$. We may therefore say,
with $99\%$ confidence, that the failure probability of Q-1 over the sampled
domain is below $3.6 \times 10^{-4}$. The bound is distribution-free — it needs
no assumption about *how* the edges are drawn — and it is a statement about the
sampled domain, not about $\mathbb{R}$. Outside the domain it says nothing, which
is the same honesty boundary §23.2 drew.

The last task packages the whole audit — proofs, disclosure counts, property
results — into one canonical JSON document and hashes it. The hash is over a
payload with no timestamps and no wall-clock values, so it is stable from run to
run; changing any field changes the hash.

**Run this now.** Task 6 (`python3 code/gftac/ch23_tasks.py 6`):

```python
cert = build_certificate()

print("Z3 + Property-Based Verification Certificate (canonical JSON):")
print(json.dumps(cert, indent=2, sort_keys=True))
print("")
print(f"  SHA-256 of the canonical payload (this field excluded): {cert['sha256']}")
```

```text
Z3 + Property-Based Verification Certificate (canonical JSON):
{
  "all_goals_proved": true,
  "chapter": "23",
  "disclosure": {
    "certificates": 8,
    "cover_floating_point": 0,
    "over_reals": 8
  },
  "proofs": {
    "Q-1": "unsat",
    "Q-2": "unsat",
    "Q-3": "unsat",
    "TAC-1": "unsat",
    "TAC-2": "unsat",
    "TAC-3": "unsat",
    "TAC-4": "unsat",
    "TAC-5": "unsat"
  },
  "property_based": {
    "q1": {
      "agreement": true,
      "boundary_fp32_passed": 0,
      "boundary_fp64_passed": 50,
      "boundary_total": 50,
      "edges": 12800,
      "edges_per_graph": 64,
      "fp32_passed": 12800,
      "fp64_passed": 12800,
      "graphs": 200
    },
    "q2": {
      "agreement": true,
      "fp32_passed": 50,
      "fp64_passed": 50,
      "lattices": 50,
      "min_margin": 36.66,
      "nodes": 12
    }
  },
  "repo_sat_witnesses": [
    "TAC-5"
  ],
  "sha256": "eb730262d0cf0d0d4f500f207a9536396eda32591605787300a9b44027f63c1e",
  "title": "What the Z3 Certificates Do and Don't Say",
  "z3_version": "5.1.0"
}

  SHA-256 of the canonical payload (this field excluded): eb730262d0cf0d0d4f500f207a9536396eda32591605787300a9b44027f63c1e
```

**Interpretation.** The certificate is a ledger, not a proof: it records eight
`unsat` goals, names TAC-5 as the one shipped SAT witness, counts the disclosure
as eight certificates over the reals with zero covering floating point, and
carries the property-based pass counts and the boundary probe. The SHA-256
digest is over the canonical JSON of everything except the digest field itself,
so the hash is stable across runs and any edit to a count, a status, or a
margin changes it. What the hash attests is *integrity* — that the numbers you
read are the numbers that were computed — not truth. The truth is carried by the
proofs and by the tests: `test_certificate_well_formed` asserts the structure,
and `test_certificate_hash_is_stable_and_verifiable` recomputes the digest and
confirms it is unchanged on a second assembly. This is **MEASURED**.

---

## 23.6 What the Certificates Do and Don't Say

We set out to read the certificates rather than to repeat the word "verified,"
and the reading is now complete.

What they **do** say is precise and strong. Eight goals are refuted by a sound
decision procedure over exact arithmetic: the Lyapunov dissipation of TAC-1, the
dimension-independent relaxation of TAC-2, the double-well partition of TAC-3,
the exact-cover ground state of TAC-4, the $S_5$ arithmetic of TAC-5, and the
three GeoFormer invariants Q-1, Q-2, Q-3. By Theorem B.23.1, each `unsat` is a
proof that every model of the premises satisfies the conclusion, over the real
numbers. TAC-5 ships as a SAT witness and we upgraded it to a genuine
refutation; the rest ship in proof shape already.

What they **don't** say is equally precise. Every one quantifies over
$\mathbb{R}$ or $\mathbb{Z}$, and none of them says anything about the binary32
or binary64 code that runs. Theorem B.23.2 makes one instance of the gap
explicit: an increment below the binary32 half-spacing is absorbed by the
current conductance, and the strict contraction of Q-1 fails — measurably, on
every one of the $50$ probe edges. The certificates also inherit their premises
rather than deriving them: TAC-2 assumes the spectral gap, TAC-4 proves a
micro-instance, Q-2 assumes the Fiedler inequality, Q-3 bounds a ratio without
the hop induction. The disclosure ledger of Task 3 is that list, and the
"0 cover floating-point rounding" line is its summary.

What we **added** is the complementary evidence. Property-based randomized
verification confirms Q-1 on $12{,}800$ sampled edges in both precisions, and
Q-2 on $50$ sampled lattices in both precisions, with the two precisions in
agreement wherever the invariants hold and in deliberate disagreement at the
boundary Theorem B.23.2 predicts. Theorem B.23.3 converts the clean Q-1 sample
into a distribution-free statement: below $3.6 \times 10^{-4}$ failure
probability at $99\%$ confidence, over the sampled domain. And the certificate of
Task 6 binds the whole audit — proofs, disclosure, samples — to a stable
SHA-256, so the ledger cannot drift from the artifact that produced it.

The architecture's armor, then, is not a single certificate but a layered
argument: exact proofs over the model, an explicit statement of the model's
boundary, randomized checks inside that boundary, and a hash over the record.
Read that way, "machine-checkable invariant" is a claim you can hold the program
to — and, as this chapter has just done, hold to the light.
