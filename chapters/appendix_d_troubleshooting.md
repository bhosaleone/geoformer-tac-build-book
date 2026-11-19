# Appendix D — Troubleshooting

Every failure in this appendix is one the book actually hit while being built,
and each entry gives the symptom, the cause, and the fix. They are collected
here because the same three or four causes — an import path, an unseeded
generator, a dtype, a drift between prose and artifact — account for nearly
every failure a reader will meet.

## D.1 "ModuleNotFoundError: No module named 'gftac'"

**Symptom.** Running a task or a test from the wrong directory, or importing the
package directly, fails with `ModuleNotFoundError: No module named 'gftac'`.

**Cause.** `gftac` lives at `code/gftac/`, not at the repository root. Python
does not have `code/` on its import path unless it is told to.

**Fix.** Run the tasks and tests **from the book's root**, exactly as the
chapters show: `python3 code/gftac/chNN_tasks.py N`. The task modules insert
`code/` into `sys.path` themselves, and the test suite runs from the root where
`pytest` finds the package. If you must import from a REPL, prefix with
`PYTHONPATH=code`:

```
PYTHONPATH=code python3 -c "from gftac.ch22_claims import CLAIMS; print(len(CLAIMS))"
```

## D.2 The torch sparse-invariant warning

**Symptom.** A `UserWarning: Sparse invariant checks are implicitly disabled…`
appears whenever Chapter 18's or Chapter 24's sparse memory runs.

**Cause.** PyTorch's sparse COO tensors carry index invariants (sorted indices,
no duplicates) that are *not* checked by default, because checking them costs
performance. The warning is informational, not an error.

**Fix.** Nothing. The book's sparse construction calls `.coalesce()`, which
restores the invariants, so the warning is noise. If you want it silenced, opt
in explicitly with
`torch.sparse.check_sparse_tensor_invariants.enable()` around the construction,
or filter the warning in your test config. Do **not** disable the invariants
globally and then construct sparse tensors by hand — that is the path to the
segfault the warning describes.

## D.3 A number changes between runs

**Symptom.** A task prints a different value on a second run, or a transcript
check fails with a one-line difference.

**Cause.** An unseeded random draw. The global torch generator carries state
between calls, so a module that draws without seeding inherits whatever the
previous call left behind. This is exactly the Chapter 11 task 6 failure
described in Appendix C.3.

**Fix.** Seed the generator immediately before the draw, at the top of the task
function: `torch.manual_seed(k)`. Then **pin the new value in three places**: the
task's own logic (nothing), the chapter test that asserts the printed line, and
the markdown transcript. If the value is genuinely machine-dependent, do not pin
it — wrap it in a volatile span (`«…»`) so the checker matches it by pattern.

## D.4 A transcript drifts from the artifact

**Symptom.** `python3 verify_transcripts.py` reports a mismatch: a line in the
book and the line the command prints differ.

**Cause.** Either the prose was edited without re-running the task, or the task
was changed without updating the prose. Both are drift, and both are exactly
what the checker exists to catch.

**Fix.** Run the command the book names, and paste its **verbatim** output into
the `text` block. Do not hand-edit the digits. If the value is volatile, mark
the varying span. After editing, re-run `python3 verify_transcripts.py`; a
`0 mismatch` line means the prose and the artifact agree again. Remember that
the checker pairs each `python3 code/gftac/...` mention with the **next** `text`
block, so keep the command and its output adjacent.

## D.5 FP32 and FP64 disagree at the last bits

**Symptom.** A residual, a determinant, or a symmetry check passes in one dtype
and fails in the other, or fails by a hair at a tolerance that "should" work.

**Cause.** Floating-point associativity. FP32 has machine epsilon
$2^{-24} \\approx 5.96\\times10^{-8}$ and FP64 has $2^{-52} \\approx
2.22\\times10^{-16}$; a reduction that sums thousands of terms accumulates error
at the epsilon of its dtype.

**Fix.** Choose the tolerance from the dtype, not from hope. The book's own
residuals — `2.384e-07` (FP32) and `4.441e-16` (FP64) for the Chapter 7
permutation — are pinned at exactly those sizes, because they are the expected
error, not a defect. If a check must hold across dtypes, assert it in FP64 and
assert the *scaling* in FP32.

## D.6 Eigenvector sign or eigenvalue ordering surprises

**Symptom.** A Fiedler-vector-derived quantity flips sign between runs, or two
near-degenerate eigenvalues swap order.

**Cause.** Eigenvectors are defined up to sign, and near-degenerate eigenvalues
are numerically indistinguishable. `torch.linalg.eigh` is free to return either
sign and either order.

**Fix.** Never assert on a raw signed eigenvector component. Assert on an
eigenvalue (which is sign-free) or on an absolute value or an invariant (a
graph cut, a connectivity, a ratio). This is why the book reports the Fiedler
*value* and not the Fiedler *vector*.

## D.7 Z3 is missing or a proof fails

**Symptom.** Chapter 23's tasks raise an import error for `z3`, or a goal that
should be `unsat` comes back `unknown`.

**Cause.** The `z3-solver` package is not installed, or the query is in a theory
Z3 cannot decide — nonlinear real arithmetic is decidable but expensive, and a
timeout returns `unknown` rather than `unsat`.

**Fix.** Install `z3-solver` (the reference run used 5.1.0). If a goal returns
`unknown`, it is not a proof; the book's Chapter 23 is explicit that an
`unknown` is a non-answer, and the property-based sweep is the fallback that
covers exactly the cases Z3 cannot discharge quickly. Never read `unknown` as
`unsat`.

## D.8 Stochastic rounding is nondeterministic by design

**Symptom.** A quantization test using `stochastic=True` fails intermittently.

**Cause.** Unbiased stochastic rounding draws a random bit per value by
construction; it is *supposed* to vary between runs.

**Fix.** Assert on the *statistics* of stochastic rounding (the mean, the
unbiasedness), not on individual values. Assert exact values only on the
deterministic path, `stochastic=False`. The book keeps the two paths separate
for precisely this reason.

## D.9 A single task does not run

**Symptom.** `python3 code/gftac/chNN_tasks.py 4` runs every task, or runs
nothing, instead of running task 4.

**Cause.** A task module whose `main()` does not honour the numeric argument.
Every task module in the book accepts a task number or `all`, and dispatches on
it.

**Fix.** Check the module's `main()`: it should map the integer argument to the
matching `task_N` function and print a `--- Task N ---` header before it. The
Chapter 14–17 modules were retrofitted with single-task dispatch during the
hardening pass for exactly this reason; if you extend a chapter, keep the
convention.

## D.10 Where to look first

When something fails, the order that resolves failures fastest is:

1. `python3 verify_transcripts.py` — catches prose/artifact drift in seconds.
2. `python3 -m pytest tests/test_chNN_*.py -q` — catches a broken chapter in isolation.
3. `python3 code/gftac/chNN_tasks.py N` — shows the exact line that disagrees.
4. `python3 verify_book.py` — the full gate: tests, transcripts, and the artifact manifest.

If step 4 is green, the chapter is done by the book's own definition: the
markdown is complete, the module exists, the tests pass, and the certificate
records it. That is the only definition the book accepts.
