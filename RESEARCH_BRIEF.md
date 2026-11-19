# RESEARCH_BRIEF.md — A Ten-Page Brief for Busy Reviewers

**Read this instead of 283 pages.** Everything here is backed by the repository's tests, claims
ledger, and verification certificate. The honest boundaries are in `LIMITATIONS.md`.

---

## Positioning

This is **not** a claim to have invented a better Transformer.

The claim is narrower and more testable:

> We have developed a reproducible architecture for geometry-driven relational computation,
> with thermodynamic convergence, quantized learning, and structured memory. The complete
> construction and verification framework is public. Please try to break it.

GeoFormer & TAC is best understood as a **complementary relational substrate**: a way to make
graph topology, effective resistance, and attractor dynamics part of the computation rather than
merely an input representation. A near-term use is structured multi-hop retrieval alongside an
existing foundation model, not replacing its language-generation stack.

## The problem

Transformers use an immutable Euclidean representation space, all-to-all attention, an implicit
metric, discrete depth, and knowledge stored in weights. That contract is excellent for dense
sequence modelling and poor at four things the book calls P1–P4: double-duty representations
(routing and semantics in the same weights), topological blindness, discrete depth, and
weight-bound memory. The book's Chapter 1 builds a tiny demonstration: a one-hop-trained model
scores ~0% on unseen 2- and 3-hop queries by echoing the direct successor, while propagation
scores 100% at every depth.

## The proposed machinery

| Component | Mechanism | Book chapter |
|---|---|---|
| Metric lattice | Non-negative conductance $W$, effective resistance, Laplacian, directed kernel | 2 |
| TAC relaxation | Gradient flow on free energy, Lyapunov dissipation, Kramers escape | 3 |
| Ternary synapses | 1.58-bit weights, straight-through estimation, 5 trits per byte | 4 |
| Q-GHA | Attention scores shifted by geodesic distance: $S_{ij} - \lambda d_{ij}$ | 5 |
| Q-TAC layers | Continuous relaxation inside the block, layer-free depth | 7 |
| S₅ projector | Symmetry pooling via irreps; description-length bound | 8 |
| Two-timescale learning | Fast Hebbian writes on the lattice plus slow weights | 10 |
| Soliton sharpening | $\gamma$-power renormalization under $\alpha_0^\gamma > d_{avg}$ | 16 |
| Goal-conditioned halting | Free energy with a semantic term; autonomous $\tau^*$ | 17 |
| Sparse relational memory | CSR conductance store at 100,000 entities | 18 |
| MCU export / memristor model | Q8.8 firmware path; analytical crossbar energy | 20–21 |

The design argument is that **semantics and relational routing need not share a storage
substrate**. Conductance carries structure; ternary weights carry semantics; relaxation depth is
emergent rather than fixed.

## Evidence, with labels

Every number below is in `CLAIMS.md` with its artifact, test, and regenerating command.

| Result | Value | Label | Reading |
|---|---|---|---|
| One-hop expert on unseen multi-hop | ~0% at 2/3 hops | MEASURED | Controlled synthetic chain, Chapter 1 |
| Ternary foundation footprint | 249,665 params → 48.76 KiB | MEASURED | Packing arithmetic, Chapter 4 |
| Chebyshev schedule acceleration | up to 31.5× at k=6 | MEASURED | Relaxation simulator, Chapter 7 |
| S₅ description-length bound | ≤ 54.73 bits | MODELED | Representation-theoretic arithmetic |
| Autonomous halting | 100% at 32 hops, 5 seeds | MEASURED | Constructed un-truncated graphs, Chapter 17 |
| Sparse memory at 100k entities | 100,000× vs dense | MEASURED | Book CSR implementation, Chapter 18 |
| Sparse vs dense propagation error | $3.73\times10^{-9}$ at 8 hops | MEASURED | Chapter 18 |
| Memristor energy per inference | 443.897 pJ | MODELED | Analytical model, not silicon |
| Clinical KG integration | 128 entities, 166.6× compression | RESEARCH DEMO | Toy ontology, no clinical claim |

The claims ledger currently holds 31 rows: 25 MEASURED, 4 MODELED, 2 RESEARCH DEMO.

## What is genuinely falsifiable

These are the questions we would ask first if we were reviewing the work:

1. **Does the relational substrate help on real graphs?** Run the Chapter 17 protocol on FB15k-237
   or Wikidata with no planted highway. Predicate: if Q-GHA's advantage comes from $\alpha_0 > 1$
   planting, it should disappear.
2. **Does TAC beat an equal-compute iterative baseline?** Match FLOPs and steps. A relaxation
   loop is not free; the book does not claim it is.
3. **How does graph overhead scale?** Profile throughput, memory bandwidth, and performance per
   watt against an optimized dense baseline.
4. **Is the soliton bound practically relevant?** Test near the stated boundary
   $\alpha_0^\gamma \approx d_{avg}$ and below it, where the book predicts failure.
5. **Do the theorems add capability beyond existing geometric deep learning?** Compare with R-GCN,
   GAT, graph transformers, and resistance-based methods under matched information.

A useful result would be a negative one: a clean failure that tells us where the substrate stops
working. The book already documents several such failures (Blind-PG expansion graphs, linear
diffusion drowning/echo/dispersion, below-boundary sharpening).

## What we are not claiming

* No claim of general-purpose reasoning superiority.
* No claim that $p$-values in Chapter 14 constitute benchmark evidence — Task 4 uses supplied
  values and is labeled as a worked example.
* No claim of hardware measurement for the memristor model.
* No clinical claim for the Chapter 19 demo.
* No claim that Z3 certificates cover unbounded domains.
* No claim of novelty over the full geometric-Deep-Learning literature; that comparison is open.

## Reproduce in one command

```bash
git clone https://github.com/bhosaleone/geoformer-tac-build-book.git
cd geoformer-tac-build-book && pip install -r requirements.lock.txt && make all
python3 verify_book.py --check
```

Expected: 442 tests passed, 151 transcripts matched, 222 artifacts hashed, `all_green: true`.
Full instructions are in `REPRODUCTION.md`; the boundaries are in `LIMITATIONS.md`.

## Suggested next experiments (in priority order)

| Priority | Experiment | Why it matters |
|---|---|---|
| 1 | Strong-baseline comparison on one real KG | Tests external validity |
| 2 | Equal-compute TAC vs iterative baseline | Separates dynamics from extra FLOPs |
| 3 | Noisy / dynamic graph robustness | Tests the deployment story |
| 4 | External mathematical review of B.2.x, B.3.x, B.16.1 | Tests theoretical claims |
| 5 | Physical crossbar measurement | Upgrades MODELED to MEASURED |
| 6 | Independent replication by a third party | The credibility event that matters |

## Contact and licensing

Code: AGPL-3.0-only. Book and figures: CC BY-NC-SA 4.0. Commercial proprietary exemption:
**research@twistpool.org**. Book DOI: 10.5281/zenodo.24358478; research program DOI:
10.5281/zenodo.23730305.

We would rather have a rigorous replication than praise. Open an issue with an artifact and a
command; we will treat a falsified claim as a successful contribution.
