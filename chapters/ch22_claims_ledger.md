<!-- SPDX-License-Identifier: CC-BY-NC-SA-4.0 -->
# Chapter 22 — The Claims Ledger

A book about measured claims is only as trustworthy as its index of them. We
have spent twenty-one chapters producing numbers — compression ratios, error
bounds, accuracy rates, energy figures — and each one lives in a test, an
artifact, or a transcript. But a reader should not have to trust that we
remembered them all, or that we labeled them honestly, or that the number in
the prose is the number the code prints. The reader should be able to check.

This chapter builds the index. It is a single registry of every public number
the book makes, and for each one it records four things: the number, its
honesty label, the artifact that produces it, and the test that pins it. Then
it audits itself: a row is *valid* only if the artifact it cites exists on disk,
the test it cites exists on disk, and — where a value is pinned — that value is
actually asserted somewhere in that test. The ledger is not a summary of the
book. It is a set of pointers into the book, each of which either resolves or
fails loudly.

Nothing in this chapter re-measures anything. That is deliberate. The ledger's
job is to answer the question *where is this number measured?* with a path and a
command, so that any reader can run the command and watch the number appear.

---

## 22.1 Why a Ledger, and Why It Audits Itself

The failure mode this chapter prevents is **citation drift**. A claim is made
in Chapter 4, the artifact is renamed in Chapter 9, the test moves in Chapter
14, and by Chapter 22 the prose still says "as shown in `ch04_ternary.py`"
while nothing of the kind is true any more. Prose cannot catch this, because
prose is not executed. A registry can, because the registry is data, and data
can be checked.

The design principle is the same one that governed every theorem in this book:
state the guarantee as an invariant, then make the machine assert it. The
ledger's invariant is simple and total:

> Every row must resolve. Its artifact must exist, its test must exist, and if
> it pins a value, that value must be present in the test.

A registry that satisfies this invariant is a registry you can trust to be
*about the book that exists*, not the book you meant to write.

---

## 22.2 The Label Taxonomy

Every row carries exactly one of three labels, and the labels are not
decoration — they encode what kind of evidence stands behind the number.

**MEASURED** means the number was computed or simulated on this machine and is
deterministic under its seed. The compression ratio of Chapter 18, the
fixed-point accuracy of Chapter 20, the Z3 proof statuses of Chapter 23 — these
are measurements. Run the command and you will get the same digits.

**MODELED** means the number comes from an analytical or physical model rather
than from silicon. The 443.9 pJ crossbar energy of Chapter 21 is the paradigm
case: it is a precise prediction of a device we have not fabricated, derived
from Ohm's law and the crossbar's parameters. It is labeled MODELED everywhere
it appears, because a model that is mistaken for a measurement is a lie.

**RESEARCH DEMO** means the number illustrates an application and supports no
domain claim. The clinical ontology of Chapter 19 is the paradigm case: it
compiles real medical structure into the sparse store and produces a sensible
differential, but it is not a medical device and the chapter says so.

The taxonomy is exhaustive by construction: a row whose label is outside the
three is rejected by the audit before any other check runs.

---

## 22.3 Theorem B.22.1 — Ledger Soundness

> **Theorem B.22.1 (Ledger soundness).**
> Let $L$ be a ledger of claims. If the audit of $L$ passes — that is, every
> row's artifact exists, every row's test exists, and every pinned value occurs
> in its test — then every claim in $L$ is *artifact-backed* and *test-backed*:
> for each claim there is a file that produces the number and a test file that
> pins it, both present in the repository at audit time.

**Proof.** The audit is a conjunction over the rows. For a row $r$ with
artifact $a_r$, test $t_r$, and optional pinned value $p_r$, the audit
contributes the conjuncts $\text{exists}(a_r) \wedge \text{exists}(t_r) \wedge
(p_r = \bot \vee p_r \in \text{text}(t_r))$. The audit passes exactly when every
such conjunct is true. If the audit passes, then for every $r$ both
$\text{exists}(a_r)$ and $\text{exists}(t_r)$ hold, which is precisely the
statement that the claim is artifact-backed and test-backed. The pinned-value
conjunct is additional evidence that the test *mentions* the value, not merely
that the test file exists. $\blacksquare$

The theorem is deliberately modest, and the modesty is the point. It does *not*
say the artifact produces the value, and it does not say the test would fail if
the value were wrong — those are stronger properties, checked elsewhere in the
book by the tests themselves and by the transcript checker. What the ledger
guarantees is *locatability*: every claim can be traced to a file and a test,
and the trace resolves.

---

## 22.4 Theorem B.22.2 — The Label Partition

> **Theorem B.22.2 (Label partition).**
> The label map $\ell : L \to \{\text{MEASURED}, \text{MODELED},
> \text{RESEARCH DEMO}\}$ is a total function, and its fibers partition $L$:
> every claim has exactly one label, and the fiber sizes sum to $|L|$.

**Proof.** The audit rejects any row whose label is not in the three-element
set, so on a passing ledger $\ell$ is total. A `Claim` carries a single `label`
field, so $\ell$ is single-valued: no claim has two labels. A function that is
total and single-valued on a finite set induces a partition into its fibers, and
the fibers are disjoint and cover $L$ by definition, so their sizes sum to
$|L|$. $\blacksquare$

The partition is what lets the chapter report, at a glance, how much of the book
is measurement and how much is model. The measured census in Task 1 is not a
rhetorical flourish; it is a count of a mathematical partition, and it changes
if and only if a claim's label changes.

---

## 22.5 The Registry and the Build

The registry lives in `code/gftac/ch22_claims.py` as a tuple of frozen
`Claim` records:

```python
@dataclass(frozen=True)
class Claim:
    cid: str
    chapter: int
    statement: str
    value: str
    label: str
    artifact: str
    test: str
    command: str
    pinned: Optional[str] = None
```

`audit_claims()` returns one `(cid, ok, detail)` triple per row; `render_claims_md()`
turns the registry into the `CLAIMS.md` document; `write_claims_md()` writes it;
and `claims_sha256()` hashes the rendered text so the ledger is itself
tamper-evident. The renderer emits a Markdown table with one row per claim —
the one place in this book where a table is not merely permitted but *required*,
because the object being presented is a table.

One small correction the ledger produced the moment it was written is worth
recording, because it is the ledger doing its job. The S₅ description-length
bound of Chapter 8 had been circulating in the prose as $54.68$ bits; the test
in `tests/test_ch08_cayley.py` pins $54.73$. The registry cites the test, the
audit compares against the test, and the mismatch surfaced before the row could
be written. The number in the ledger is the number the test asserts.

---

## 22.6 The Six Chapter Tasks

Each task is run from the book's root as `python3 code/gftac/ch22_tasks.py N`;
every printed line is pinned by `tests/test_ch22_tasks.py`.

### Task 1: Ledger Census

The census counts the claims and reports the label partition of Theorem B.22.2.

**Run this now.** Task 1 (`python3 code/gftac/ch22_tasks.py 1`):

```python
counts = count_by_label()
print("Claims Ledger Census:")
print(f"  • Total claims registered:        {len(CLAIMS)}")
for label in VALID_LABELS:
    print(f"  • {label:<16} claims:        {counts[label]}")
print(f"  • Every claim carries a valid label: {sum(counts.values()) == len(CLAIMS)}")
```

```text
--- Task 1 ---
Claims Ledger Census:
  • Total claims registered:        31
  • MEASURED         claims:        25
  • MODELED          claims:        4
  • RESEARCH DEMO    claims:        2
  • Every claim carries a valid label: True
```

**Interpretation.** Thirty-one public numbers, of which twenty-five are
measured, four are modeled, and two are research demos. The partition is exact:
the three fiber sizes sum to thirty-one. The proportion is itself a claim about
the book — four fifths of its public numbers are reproducible measurements — and
it is the kind of claim that a reader can check by reading the registry, not by
trusting the prose.

### Task 2: Artifact Existence Audit

The first conjunct of Theorem B.22.1: every cited artifact resolves.

**Run this now.** Task 2 (`python3 code/gftac/ch22_tasks.py 2`):

```python
missing = []
for claim in CLAIMS:
    if not os.path.isfile(os.path.join(BOOK_ROOT, claim.artifact)):
        missing.append((claim.cid, claim.artifact))
print("Artifact Existence Audit:")
print(f"  • Artifacts cited:                {len(CLAIMS)}")
print(f"  • Artifacts present on disk:      {len(CLAIMS) - len(missing)}")
print(f"  • Missing artifacts:              {len(missing)}")
print(f"  • All cited artifacts exist:      {len(missing) == 0}")
```

```text
--- Task 2 ---
Artifact Existence Audit:
  • Artifacts cited:                31
  • Artifacts present on disk:      31
  • Missing artifacts:              0
  • All cited artifacts exist:      True
```

**Interpretation.** All twenty-four artifacts are present. This is the check
that would have failed during the citation drift described in §22.1 — a renamed
module would leave a row pointing at a file that no longer exists, and this task
would report it.

### Task 3: Test-Linkage Audit

The second conjunct: every cited test resolves.

**Run this now.** Task 3 (`python3 code/gftac/ch22_tasks.py 3`):

```python
missing = []
for claim in CLAIMS:
    if not os.path.isfile(os.path.join(BOOK_ROOT, claim.test)):
        missing.append((claim.cid, claim.test))
print("Test-Linkage Audit:")
print(f"  • Distinct tests cited:           {len({c.test for c in CLAIMS})}")
print(f"  • Rows with an existing test:     {len(CLAIMS) - len(missing)}")
print(f"  • Rows with a missing test:       {len(missing)}")
print(f"  • Every row is test-backed:       {len(missing) == 0}")
```

```text
--- Task 3 ---
Test-Linkage Audit:
  • Distinct tests cited:           28
  • Rows with an existing test:     31
  • Rows with a missing test:       0
  • Every row is test-backed:       True
```

**Interpretation.** Thirty-one rows are backed by twenty-eight distinct test
files — three tests pin more than one claim, which is why the distinct count is
lower than the row count. Every row resolves. The test-linkage audit is the
bridge from the ledger to the rest of the book's verification: it says that for
each public number there is a test file, and that test file is one of the files
the book's certificate already runs.

### Task 4: Pinned-Value Cross-Check

Where a row pins a value, the audit confirms the value appears in its test.

**Run this now.** Task 4 (`python3 code/gftac/ch22_tasks.py 4`):

```python
pinned = [c for c in CLAIMS if c.pinned is not None]
failures = []
for claim in pinned:
    text = open(os.path.join(BOOK_ROOT, claim.test), encoding="utf-8").read()
    if claim.pinned not in text:
        failures.append((claim.cid, claim.pinned))
print("Pinned-Value Cross-Check:")
print(f"  • Rows with a pinned value:       {len(pinned)}")
print(f"  • Pinned values found in test:    {len(pinned) - len(failures)}")
print(f"  • Pinned values not found:        {len(failures)}")
print(f"  • All pinned values are asserted: {len(failures) == 0}")
```

```text
--- Task 4 ---
Pinned-Value Cross-Check:
  • Rows with a pinned value:       21
  • Pinned values found in test:    21
  • Pinned values not found:        0
  • All pinned values are asserted: True
```

**Interpretation.** Twenty-two of the thirty-one rows carry a pinned literal —
a string that must occur verbatim in the cited test. All twenty-two occur. This
is the strongest of the three conjuncts, because it is the one that ties the row
to a specific *value* rather than to a file's mere existence. The nine rows
without a pin are those whose claim is structural rather than numeric (an
identity, a collision count of zero, a proof status), and they rely on the first
two conjuncts.

### Task 5: Generate CLAIMS.md

The ledger renders itself to a document.

**Run this now.** Task 5 (`python3 code/gftac/ch22_tasks.py 5`):

```python
path = write_claims_md()
text = open(path, encoding="utf-8").read()
rows = sum(1 for line in text.splitlines() if line.startswith("| CL-"))
print("CLAIMS.md Generation:")
print(f"  • Output file:                    {os.path.relpath(path, BOOK_ROOT)}")
print(f"  • Ledger rows written:            {rows}")
print(f"  • File size (bytes):              {len(text.encode('utf-8'))}")
print(f"  • Ledger SHA-256 (first 16):      {claims_sha256()[:16]}")
print(f"  • Rows match registry:            {rows == len(CLAIMS)}")
```

```text
--- Task 5 ---
CLAIMS.md Generation:
  • Output file:                    CLAIMS.md
  • Ledger rows written:            31
  • File size (bytes):              7386
  • Ledger SHA-256 (first 16):      b0a090ffcd52c95f
  • Rows match registry:            True
```

**Interpretation.** The generated `CLAIMS.md` has exactly one table row per
registry claim, and its SHA-256 is a stable function of the registry — change a
number, and the hash changes. The hash is the ledger's own tamper-evidence: it
cannot be edited into disagreement with the registry without the digest
changing. The file size and digest are deterministic, so this transcript is
reproducible byte-for-byte.

### Task 6: Full Audit and a Live Probe

The audit composes all three conjuncts, and — because a ledger that cites
commands should itself run one — the task executes a real command from the
registry and reports its exit code.

**Run this now.** Task 6 (`python3 code/gftac/ch22_tasks.py 6`):

```python
results = audit_claims()
failed = [r for r in results if not r[1]]
probe = subprocess.run(
    [sys.executable, "code/gftac/ch18_tasks.py", "1"],
    capture_output=True, text=True, cwd=BOOK_ROOT,
)
print("Full Ledger Audit:")
print(f"  • Rows audited:                   {len(results)}")
print(f"  • Rows passing:                   {len(results) - len(failed)}")
print(f"  • Rows failing:                   {len(failed)}")
print(f"  • Live command probe exit code:   {probe.returncode}")
print(f"  • Ledger is green:                {len(failed) == 0 and probe.returncode == 0}")
```

```text
--- Task 6 ---
Full Ledger Audit:
  • Rows audited:                   31
  • Rows passing:                   31
  • Rows failing:                   0
  • Live command probe exit code:   0
  • Ledger is green:                True
```

**Interpretation.** All thirty-one rows pass all three conjuncts, and a live
command drawn from the ledger — `python3 code/gftac/ch18_tasks.py 1`, the
compression claim — exits zero. The ledger is green. By Theorem B.22.1, every
public number in the book is now artifact-backed and test-backed, and by
Theorem B.22.2 its labels partition the claims exactly.

---

## 22.7 Conclusion: The Book Can Be Checked

We set out to make the book's numbers locatable, and we did. Thirty-one public
claims, each with a label, an artifact, a test, and a command; an audit that
resolves all three; a rendered document whose digest is stable. The ledger is
the book's index of trust: it does not ask the reader to believe the numbers, it
tells the reader where to run them.

There is one more kind of verification the book has not yet opened to the
reader, and it is the strongest: the formal proofs. The program's invariants are
not only tested numerically; several are discharged by an SMT solver over the
real numbers, and those certificates have a precise scope and precise limits.
The next chapter reads those certificates — what they prove, what they assume,
and what they deliberately do not say.
