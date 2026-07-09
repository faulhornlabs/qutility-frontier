# Shor Period-Finding Benchmark — Implementation Status

*Last updated: 2026-07-09*

This document tracks the review/verification status of the
`ShorPeriodFindingBenchmark` implementation in
`frontier/shorbenchmark/shorbenchmark.py`. The review evidence (runnable checks,
plots) lives in `notebooks/ShorPeriodFindingBenchmark_tutorial.ipynb`, which is
being built up section by section as components are reviewed.

## Component status

| Component | Code | Reviewed / verified |
|---|---|---|
| Mersenne factor table (`_mersenne_factors.py`) | done | yes — notebook + pytest |
| Primitive polynomial generation | done | yes — notebook + pytest |
| CNOT synthesis (Patel–Markov–Hayes) | done | yes — notebook (no pytest coverage yet) |
| Circuit assembly (`_build_period_finding_circuit`) | done | **not yet** |
| Sample creation (`_create_single_sample`) | done | **not yet** |
| Evaluation (`evaluate_benchmark`, `_is_successful_bitstring`) | done | **not yet** |

### Primitive polynomial generation — reviewed

- Polynomials over GF(2) are encoded as ints (bit `i` = coefficient of `x**i`);
  arithmetic (`_poly_mul`, `_poly_mod`, `_poly_gcd`, `_poly_pow_mod`) is
  implemented with bitwise operations, with explanatory comments.
- `_is_primitive_polynomial` = Rabin irreducibility test + order test; the order
  test needs the complete factorization of `2**degree - 1`, looked up from a
  precomputed table covering degrees 1–200 (extend with
  `scripts/generate_mersenne_factor_table.py`).
- `_primitive_polynomial_for_degree(degree, sample_id)` finds a primitive
  polynomial by seeded random search — deterministic per `(degree, sample_id)`
  (the pair is the RNG seed), expected O(n log n) attempts.
- Covered by `test/test_shorbenchmark.py` and the notebook's first section
  (correctness vs. textbook examples, determinism, sample_id variation, timing
  up to degree 200, clear KeyError beyond table coverage).

### CNOT synthesis — rewritten to Patel–Markov–Hayes, reviewed

Replaced the original plain Gauss–Jordan synthesis (O(n²) CNOTs) on 2026-07-09.

- `_lower_triangularize` — PMH core (quant-ph/0302002): per column-block
  duplicate-pattern cancellation (Step A), then Gaussian elimination (Step B).
- `_pmh_cnot_schedule` — two triangular passes (matrix, then transpose of the
  remainder), assembled into a CNOT list in circuit order; tries every block
  width up to ~log2(n) and keeps the shortest schedule. Deterministic.
- `_synthesize_linear_map` — unchanged interface; now just emits the schedule as
  CX gates (or relative-phase Toffolis when a control qubit is given).

Verification (notebook section 2):

- Replaying each schedule as GF(2) row operations reproduces every controlled
  power `M**(2**k)` exactly; singular matrices raise `ValueError`.
- CNOT counts per benchmark sample (sum over all `2n + 1` controlled powers):
  ~17% below the previous implementation at n = 8, ~34% below at n = 64
  (161,867 vs 246,824).
- Compared against Qiskit's `synth_cnot_count_full_pmh` (Qiskit 2.5.0): ours is
  ~2.4× lower at n = 64. Qiskit's variant adds a greedy "back-reduce the pivot
  row" heuristic that backfires on dense matrices like these companion-matrix
  powers (an experimental variant of our code with that heuristic reproduced
  Qiskit's counts to within 0.1%); our implementation follows the paper without
  it.
- Note: the original Gauss–Jordan implementation no longer exists in the
  package; its counting logic is preserved as `gauss_jordan_cnot_count` in the
  notebook's comparison cell, which is currently its only record (the package
  predates version control coverage — see open items).

### Circuit assembly / sample creation / evaluation — code-complete, unverified

Standard QPE structure: Hadamards on a control register of `2n + offset`
qubits, target register seeded with `|0...01>`, controlled `M**(2**k)` per
control qubit, inverse QFT, measurement of the control register only.
`_is_successful_bitstring` scores a shot as success when
`Fraction(measured, 2**control_size).limit_denominator(2**n - 1)` has
denominator exactly `2**n - 1`.

**Intended behavior (confirmed 2026-07-09):** shots whose phase `s/r` has
`gcd(s, r) > 1` reduce to a smaller denominator and count as *failures*, even
though the phase estimate itself is correct. This is the deliberate success
rule; it caps the ideal success probability at roughly `phi(r)/r`.

Not yet verified end-to-end (planned as the notebook's next section):

- Ideal simulation of a small instance (e.g. n = 3 target qubits, 10 total) to
  confirm the measured phases concentrate on multiples of `1/(2**n - 1)` and
  that the success probability matches the `phi(r)/r`-level prediction.
- Bit-order consistency between the QASM emitter's bitstrings, the
  measurement mapping (control qubit `i` → classical bit `i`), the inverse
  QFT's swap convention, and `int(bitstring, 2)` in the evaluator — a silent
  failure mode only an end-to-end run can catch.

## Open items

1. **Run the pytest suite** (`test/test_shorbenchmark.py`) — not run since the
   synthesis rewrite (deferred by request; existing tests only cover the
   polynomial/Mersenne helpers, so no failures are expected).
2. **Add pytest coverage for the synthesis** — promote the notebook's
   `apply_schedule` replay check into the test suite.
3. **End-to-end simulation section** in the tutorial notebook (see above).
4. **Commit the module** — `frontier/shorbenchmark/`, the tests, and the
   tutorial notebook are still untracked in git; until committed, the notebook
   is the only record of the pre-PMH baseline.

## Environment notes

- Qiskit 2.5.0 was installed ad hoc into the project venv for the synthesis
  comparison (`pip install qiskit`). It is **not** a package dependency and
  nothing in `frontier/` imports it; only the notebook's comparison cells use
  it. Remove with `pip uninstall qiskit` if undesired.
