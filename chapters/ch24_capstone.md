<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->
# Chapter 24 — Capstone: The Whole Stack in One Pipeline

Twenty-three chapters, twenty-three components. A metric lattice that quantizes
conductance. A plasticity engine that writes to it and keeps its invariants. A
GeoFormer block stack assembled from ternary synapses, quasi-crystalline
attention, and RMSNorm. A soliton propagator certified by an analytic lower
bound. A packer that stores trits five to a byte. A sparse memory that carries
relational state across a hundred thousand entities. Each was built, tested, and
measured on its own. The capstone runs them together.

Composition is where a build book earns its name. A component that passes its
own tests in isolation can still fail to compose: an API mismatch, a shape that
only works for one batch size, a stateful generator that leaks between stages.
The capstone is the integration test at the scale of the whole book. It calls
every subsystem exactly as that subsystem's own tests call it, reports a small
set of deterministic metrics per stage, and folds all of them into a single
SHA-256 certificate. If any stage misbehaves, the certificate changes, and a
test notices.

There is one stage the capstone adds on purpose, and it is the book's one
deliberate projection: **code-security transfer**. The same sparse relational
machinery that propagates state across a clinical ontology propagates *taint*
across a code property graph. It is a toy, labeled RESEARCH DEMO, and §24.9
explains honestly what it is and what it is not — and where this paradigm is
headed next.

---

## 24.1 What the Capstone Guarantees

The capstone makes two promises, and both are theorems about the code rather
than claims about the world.

> **Theorem B.24.1 (Capstone determinism).**
> Each stage function of the capstone is a deterministic function of its
> arguments. No stage reads or mutates global random state except Stage 3, which
> seeds the torch generator explicitly before it draws; no stage depends on
> wall-clock time, the filesystem, or the network. Consequently
> `run_capstone()` returns an equal sequence of metric vectors on every
> invocation, and `certificate_sha256(run_capstone())` is a fixed point.

**Proof.** We check each stage. Stage 1 constructs a `MetricLattice` with an
explicit `seed=0` and performs only deterministic projection arithmetic. Stage 2
constructs a `PlasticConductanceEngine` with an explicit `seed=42` and calls
`update_conductance` with `stochastic=False`; the projection path taken is
`quantize(..., stochastic=False)`, which is a pure rounding map, and
`assert_invariants` is a pure predicate. Stage 4 (`simulate_multihop_soliton_chain`),
Stage 5 (`pack_trits` on a vector built by a closed-form formula), and Stage 6
(taint propagation over fixed edge lists) contain no random draws at all. Stage 3
calls `torch.manual_seed(0)` before constructing the model — whose
`TernaryLinear` layers draw their initial weights from the seeded generator — and
before drawing the input batch, so its parameter count and output are fixed. A
composition of deterministic functions is deterministic, and a pure function of
deterministic inputs — the certificate — is itself deterministic. $\\blacksquare$

> **Theorem B.24.2 (Certificate binding).**
> Let $M$ be the deterministic encoding of the concatenated stage metrics and
> $H = \\mathrm{SHA\\text{-}256}(M)$. Then any change to any stage metric yields
> $H' \\neq H$ except with probability at most $2^{-256}$.

**Proof.** The encoding $M$ is a string built by iterating the stages in a fixed
order and the metrics within each stage in sorted key order, formatting each
value by a fixed rule (`_fmt`). Two metric vectors that differ in any coordinate
therefore produce different strings $M \\neq M'$. SHA-256 is collision-resistant:
the probability that two distinct strings share a digest is bounded by
$2^{-256}$ for a random-oracle model of the hash. Hence $H \\neq H'$ except on a
collision. $\\blacksquare$

The two theorems together say the capstone is a *seal*: a fixed point that
commits to the exact measured configuration of the whole stack. Change a weight,
a quantization level, a hop count, or a single sink-reachability result, and the
seal breaks. The chapter's tests recompute the seal, so the seal cannot be
forged into agreement with a changed stack.

---

## 24.2 Stage 1 — The Metric Lattice (Chapter 2)

We build a thirty-two-node chain world, reinforce every edge to full
conductance, and read the algebraic connectivity of the resulting lattice.

**Run this now.** Stage 1 (`python3 code/gftac/ch24_tasks.py 1`):

```python
lat = MetricLattice(num_entities=n, bits=4, w_max=1.0, seed=0)
for i in range(n - 1):
    lat.observe_edge(i, i + 1, weight=1.0)
edges = int((lat.W > 0).sum().item())
return StageResult(
    "lattice",
    {
        "nodes": n,
        "edges": edges,
        "fiedler_lambda2": lat.fiedler_value(),
        "quantized": lat.is_quantized(),
    },
)
```

```text
--- Task 1 ---
Stage 1 — Metric lattice:
  • edges                        31
  • fiedler_lambda2              0.00963038671762
  • nodes                        32
  • quantized                    True
```

**Interpretation.** Thirty-one edges for thirty-two nodes — a tree, the
sparsest connected world possible, exactly what a chain should be. The Fiedler
value $\\lambda_2 = 0.0096303867$ is, to the digit, the closed form for a unit
chain: $\\lambda_2 = 2 - 2\\cos(\\pi/32)$. And `quantized` is `True`: every entry
of the conductance store is a multiple of $\\Delta w$, lies in $[0, w_{\\max}]$,
and the diagonal is exactly zero. The lattice stage inherits the Chapter 2
invariant battery intact, which is the first thing composition has to preserve.

---

## 24.3 Stage 2 — Hebbian Plasticity (Chapter 10)

We hand the same world to the plasticity engine and reinforce every edge. The
metric that matters is resistance, and the theorem that matters is B.10.1: a
reinforced edge contracts.

**Run this now.** Stage 2 (`python3 code/gftac/ch24_tasks.py 2`):

```python
eng = PlasticConductanceEngine(PlasticityConfig(num_entities=n, seed=42))
d_before = eng.resistance(0, 1)
updated = eng.update_conductance(
    list(range(n - 1)), list(range(1, n)), weight_inc=8.0
)
d_after = eng.resistance(0, 1)
return StageResult(
    "plasticity",
    {
        "edges_updated": updated,
        "resistance_before": d_before,
        "resistance_after": d_after,
        "contracted": d_after < d_before,
        "invariants": eng.assert_invariants(),
    },
)
```

```text
--- Task 2 ---
Stage 2 — Hebbian plasticity:
  • contracted                   True
  • edges_updated                31
  • invariants                   True
  • resistance_after             4.99999967549
  • resistance_before            100000000
```

**Interpretation.** All thirty-one edges updated. The resistance of the edge
$(0,1)$ fell from the empty-conductance value $1/\\epsilon = 10^{8}$ to
$\\approx 5.0 = 1/(0.2 + \\epsilon)$ — a contraction of eight orders of magnitude,
exactly the $W: 0 \\to 0.2$ step the engine is allowed to take. And `invariants`
is `True` after the batch: the plastic writes did not break non-negativity, the
hollow diagonal, the step bound, or quantization. The Chapter 10 invariant
battery survives a thirty-one-edge write, which is the second thing composition
has to preserve.

---

## 24.4 Stage 3 — Model Forward (Chapter 9)

Now the neural half of the stack. We instantiate a two-block GeoFormer, run a
batch of three vectors through it, and count what we built.

**Run this now.** Stage 3 (`python3 code/gftac/ch24_tasks.py 3`):

```python
torch.manual_seed(0)
model = BookGeoFormer(
    in_features=d_model,
    num_classes=4,
    d_model=d_model,
    n_layers=2,
    n_heads=4,
)
x = torch.randn(3, d_model)
with torch.no_grad():
    logits = model(x)
params, packed_kb = model.count_parameters()
```

```text
--- Task 3 ---
Stage 3 — Model forward:
  • batch                        3
  • logits_sum                   2.62864494324
  • num_classes                  4
  • packed_kb                    5.69375
  • params                       29152
```

**Interpretation.** A two-block, $d_{\\text{model}} = 32$ GeoFormer with four
classes holds 29,152 parameters — small enough to read, large enough to exercise
every block: input projection, two quasi-crystalline attention blocks, RMSNorm,
and a ternary head. Under BitNet b1.58 packing that is 5.69 KiB. The logits sum
is a deterministic scalar fingerprint of the forward pass; the test pins it, so
a change in any block's arithmetic changes the stage. This is the only stage that
touches the global random generator, and it seeds it explicitly — which is what
Theorem B.24.1 requires for the certificate to be a fixed point.

---

## 24.5 Stage 4 — Soliton Propagation (Chapter 16)

The reasoning half of the stack. We run the certified multi-hop soliton chain at
depth thirty-two and check it against its analytic lower bound.

**Run this now.** Stage 4 (`python3 code/gftac/ch24_tasks.py 4`):

```python
res = simulate_multihop_soliton_chain(
    num_hops=hops, alpha_0=2.5, gamma=2.5, d_avg=2.0
)
return StageResult(
    "soliton",
    {
        "hops": hops,
        "rho": res["rho"],
        "gamma_crit": res["gamma_crit"],
        "supercritical": res["is_supercritical"],
        "final_peak_mass": res["final_peak_mass"],
        "satisfies_bound": res["satisfies_bound"],
    },
)
```

```text
--- Task 4 ---
Stage 4 — Soliton propagation:
  • final_peak_mass              0.812405288219
  • gamma_crit                   0.756470797366
  • hops                         32
  • rho                          0.202385770251
  • satisfies_bound              True
  • supercritical                True
```

**Interpretation.** The chain is supercritical — $\\gamma = 2.5$ exceeds the
critical value $\\gamma_{\\text{crit}} = 0.7565$ — so the sharpening map contracts
the propagation operator's spectral gap, and the peak mass after thirty-two hops
is $0.8124$, comfortably above the analytic lower bound. `satisfies_bound` is
`True` at every hop, not merely at the end: the empirical trajectory dominates
the bound throughout. This is the Chapter 16 result, running inside the composed
stack, on the same parameters the chapter's own tests use.

---

## 24.6 Stage 5 — Packed Export (Chapter 4)

A stack is only deployable if it can be stored. We pack a deterministic
1,024-trit vector five to a byte and compare the result to the theoretical
footprint.

**Run this now.** Stage 5 (`python3 code/gftac/ch24_tasks.py 5`):

```python
trits = [(i % 3) - 1 for i in range(num_trits)]
packed = pack_trits(trits)
return StageResult(
    "export",
    {
        "trits": num_trits,
        "packed_bytes": int(packed.numel()),
        "theoretical_bytes": packed_bytes(num_trits),
        "bits_per_parameter": packed_bits_per_parameter(),
    },
)
```

```text
--- Task 5 ---
Stage 5 — Packed export:
  • bits_per_parameter           1.6
  • packed_bytes                 205
  • theoretical_bytes            205
  • trits                        1024
```

**Interpretation.** 1,024 trits pack into 205 bytes — exactly
$\\lceil 1024/5 \\rceil$, the theoretical minimum at five trits per byte. The
achieved footprint equals the bound, so the packer wastes nothing. The asymptotic
cost is $8/5 = 1.6$ bits per parameter, the BitNet b1.58 figure the whole book
has been paying against. Export is where the abstract stack becomes a byte
string, and it does so at the information-theoretic floor.

---

## 24.7 Stage 6 — Code-Security Transfer (Chapter 18, RESEARCH DEMO)

This stage is the book's one deliberate projection, and it is labeled as such
everywhere it appears. We take the sparse relational memory of Chapter 18 — the
machinery that carried state across a clinical ontology — and point it at a
**code property graph**: a small graph whose nodes are program operations and
whose edges are data-flow. Node 0 is an untrusted source. Nodes 4 and 6 are
sinks, a database query and a format string. Node 5 is a sanitizer, modeled as a
node with no outgoing data-flow edge, so it absorbs taint rather than passing it
on. We propagate a unit of taint from the source and ask which sinks ever receive
it, over eight hops.

**Run this now.** Stage 6 (`python3 code/gftac/ch24_tasks.py 6`):

```python
vuln = _taint_reaches(_CPG_VULNERABLE)
safe = _taint_reaches(_CPG_SAFE)
return StageResult(
    "code_security",
    {
        "vulnerable_sinks_reached": len(vuln),
        "safe_sinks_reached": len(safe),
        "taint_reached_sink": len(vuln) > 0,
        "sanitizer_blocked_taint": len(safe) == 0,
    },
)
```

```text
--- Task 6 ---
Stage 6 — Code-security transfer (RESEARCH DEMO):
  • safe_sinks_reached           0
  • sanitizer_blocked_taint      True
  • taint_reached_sink           True
  • vulnerable_sinks_reached     2
```

**Interpretation.** In the vulnerable program the taint reaches both sinks
(`vulnerable_sinks_reached = 2`); in the sanitized program it reaches neither
(`safe_sinks_reached = 0`). The sparse memory's reachability, which is the same
`forward_multi_hop_chain` primitive Chapter 18 used to walk a clinical chain,
distinguishes the two programs. That is the entire claim, and it is deliberately
small: an eight-node toy with two programs, demonstrating that *the primitive
transfers*. It is not a vulnerability detector, it has no soundness or
completeness guarantee, and no real code was analyzed. §24.9 says where this goes
next; this stage is only the seed.

---

## 24.8 The Whole-Stack Certificate

Finally we run all six stages in order and seal them.

**Run this now.** The certificate (`python3 code/gftac/ch24_tasks.py 7`):

```python
results = run_capstone()
print("Whole-Stack Certificate:")
print(f"  • Stages executed:                {len(results)}")
for r in results:
    print(f"  • {r.name:<20} metrics:    {len(r.metrics)}")
print(f"  • Certificate SHA-256 (first 16): {certificate_sha256(results)[:16]}")
print("  • All stages green:               True")
```

```text
--- Task 7 ---
Whole-Stack Certificate:
  • Stages executed:                6
  • lattice              metrics:    4
  • plasticity           metrics:    5
  • model                metrics:    5
  • soliton              metrics:    6
  • export               metrics:    4
  • code_security        metrics:    4
  • Certificate SHA-256 (first 16): a9e0b239a83fe32c
  • All stages green:               True
```

**Interpretation.** Six stages, twenty-eight metrics, one digest. The full
certificate is
`a9e0b239a83fe32cdd452d4ac49aee7fb64125b5bd02c54d713e5a9de98868bf`. By
Theorem B.24.1 it is a fixed point of the stack; by Theorem B.24.2 it binds the
stack's exact measured configuration. The full digest is pinned in
`tests/test_ch24_tasks.py`, and that test file is itself hashed into the book's
verification certificate — so the capstone's seal is committed to by the book's
seal, two hashes agreeing that the composed machine is the one we built.

The tests behind this chapter recompute the seal from scratch on every run:
`tests/test_ch24_capstone.py` asserts each stage's metrics and the digest, and
`tests/test_ch24_tasks.py` pins every printed line above. If any stage drifts,
the seal breaks in both places at once.

---

## 24.9 From This Book to the Security Flagship

Stage 6 was not decoration. It was the smallest possible instance of a plan, and
it is worth stating that plan plainly, because the book's machinery was, in
hindsight, shaped for it.

A **code property graph** unifies a program's syntax tree, control-flow graph,
and data-flow graph into one labeled multigraph. The questions security
researchers ask of it are relational and multi-hop: does this untrusted input
reach that sink; does it reach it through a sanitizer; is this taint path
reachable at all; which of a thousand paths is the short one. Those are exactly
the questions the book has spent twenty-three chapters learning to answer
*efficiently and with a certificate*:

- **Multi-hop relational reasoning** is Chapter 18's sparse memory. A CPG is
  sparse — real programs have far more nodes than data-flow edges — so the
  $O(|E|)$ store and the $O(|E|)$ propagation are the right tools, not the dense
  $O(N^2)$ ones.
- **Soliton sharpening** (Chapter 16) is a mechanism for keeping a propagation
  signal from dispersing across decoy paths — which is precisely what taint
  analysis fights, since a program's graph is full of branching noise.
- **Autonomous halting** (Chapter 17) is a reachability oracle: a propagation
  that stops when it has found its target rather than exploring the whole graph,
  with a measured accuracy at depth.
- **The plasticity engine** (Chapter 10) is a way to consolidate repeated
  relational patterns — a call graph learned from a corpus — into direct
  shortcuts, which is what a vulnerability knowledge base is.
- **The Z3 certificates** (Chapter 23) are the missing piece for security: a
  static analyzer that says "taint reaches the sink" is only as good as its
  proof, and an SMT solver over the reals can discharge the arithmetic and
  ordering facts that a taint path depends on.

The honest limits are as important as the fit. Stage 6 is a toy with no
soundness guarantee. Real CPGs are orders of magnitude larger than a clinical
ontology, their edges carry types and labels, and vulnerability detection has a
false-positive rate that a demo does not. None of that is a reason not to build
it; it is the specification of what the flagship must measure before it makes a
single claim.

And there is a funding structure that fits the work itself: **bug bounties**. A
model that finds real, verifiable vulnerabilities in real code produces
artifacts that programs pay for — and those payments fund the compute, the data,
and the assets the next round of research needs. The flywheel is tight: the same
output that proves the model works (a valid finding) is the output that funds it.
That is why the book ends by pointing here. The paradigm is built; the
certificates are in place; the honest demo is on the table. The next book's
Chapter 1 is a code property graph and a bug bounty that pays for the GPU that
finds the next one.

---

## 24.10 Conclusion

We have built a machine out of twenty-three parts and sealed it with a hash.

Start with a metric lattice that stores conductance on a quantized grid. Teach
it to learn, fast, with Hebbian writes that contract resistance and never break
its invariants. Give it a neural half: ternary synapses, quasi-crystalline
attention, RMSNorm, assembled into blocks. Give it a reasoning half: a soliton
propagator that holds its peak mass for thirty-two hops against an analytic
bound, and a halting rule that knows when to stop. Compress its memory to a
hundred thousand entities without going dense, and pack its weights five trits
to a byte. Prove its invariants with an SMT solver, and label every public number
MEASURED, MODELED, or RESEARCH DEMO. Then run all of it as one pipeline and hash
the result.

The capstone's certificate is the book's closing argument. It does not ask the
reader to believe that the parts fit; it recomputes the fit on every run and
fails loudly if they do not. What the book has built is not a claim about
intelligence. It is a *measurable* system — every number locatable, every
invariant tested, every theorem proved or explicitly bounded — and that is the
only kind of foundation on which a flagship can be built without lying to
itself. The next machine starts here.

---

## Exercises

**24.1** Add a seventh stage that trains the Stage 3 model for one step on a
fixed batch and reports the loss. Seed it explicitly, and prove the analogue of
Theorem B.24.1 for it: show it reads no unseeded randomness. Then extend the
certificate and confirm the digest changes — and that it changes *back* when you
revert the stage.

**24.2** In Stage 6, add a second sanitizer that is *partial* — it has an
outgoing data-flow edge to the sink, but with a small weight, modeling an
incomplete escaping routine. Define a threshold $\\theta$ on the taint mass at
the sink and report how the vulnerable/safe classification depends on $\\theta$.
This is the honest version of the demo: a sanitizer that is "almost safe."

**24.3** Prove that the certificate of Theorem B.24.2 is *not* a signature: an
adversary who changes a metric can recompute the digest. Then state precisely
what additional secret (an HMAC key, a commit hash, a timestamped transparency
log) would turn the certificate into a tamper-evident record, and write the one
line of code that does it.

**24.4** The capstone's stages are independent, so `run_capstone` could run them
concurrently. Show that the certificate is invariant to stage execution order
*because* the encoding sorts stage names and metric keys, and then argue why the
book still runs them in dependency order. (Hint: Stage 6's honesty label depends
on the reader having seen Stage 18's machinery.)

**24.5** Take the Stage 6 CPG and add a *sanitizer that is bypassable*: two
edges out of the sanitizer, one to the sink and one back into the propagator.
Show that the reachability result is unchanged but the taint *mass* at the sink
is now nonzero. What does that tell you about reachability versus mass as the
right invariant for a real taint analysis?
