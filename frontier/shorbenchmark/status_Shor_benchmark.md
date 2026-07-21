# Shor Period-Finding Benchmark — Implementation Status

*Last updated: 2026-07-21*

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
| Circuit assembly (`_build_period_finding_circuit`) | done (bug fixed, see below) | yes — notebook (Aer) |
| Sample creation (`_create_single_sample`) | done | yes — notebook |
| Evaluation (`evaluate_benchmark`, `_is_successful_bitstring`) | done | yes — notebook (Aer) |

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

### Circuit assembly / sample creation / evaluation — reviewed end-to-end (2026-07-21)

Standard QPE structure: Hadamards on a control register of `2n + offset`
qubits, target register seeded with `|0...01>`, controlled `M**(2**k)` per
control qubit, inverse QFT, measurement of the control register only.
`_is_successful_bitstring` scores a shot as success when
`Fraction(measured, 2**control_size).limit_denominator(2**n - 1)` has
denominator exactly `2**n - 1`.

**Intended behavior (confirmed 2026-07-09):** shots whose phase `s/r` has
`gcd(s, r) > 1` reduce to a smaller denominator and count as *failures*, even
though the phase estimate itself is correct. This is the deliberate success
rule; it caps the ideal success probability at exactly `phi(r)/r` (QPE samples
each `s in {0..r-1}` equally on a single length-`r` orbit).

**Bug found and fixed (2026-07-21):** `_build_period_finding_circuit`
originally sized the classical register to `total_qubits` (control + target)
while measuring only the `control_register_size` control qubits. Simulators then
returned full-width bitstrings (the unmeasured target bits as leading zeros), and
`_is_successful_bitstring`'s exact-width check (`len != control_register_size`)
rejected *every* shot — the benchmark scored 0.0 at all `n` under perfect gates.
Fixed by sizing the classical register to `control_register_size`. The physics
(QPE, iQFT convention, power ordering, bit order) was already correct; only the
register width was wrong.

Verified in the notebook (sections 3–6, ideal `qiskit_aer.AerSimulator`):

- Structural checks on `_build_period_finding_circuit` / `_create_single_sample`
  (register sizes, single `X` seed, control-only measurement mapping, JSON
  sample shape, determinism per `(n, sample_id)`).
- Unitary-level isolation checks via Qiskit `Operator`: every controlled
  `M**(2**k)` equals the controlled permutation up to a single global phase
  (guards the Toffoli synthesis of the controlled map), and `_inverse_qft`
  matches the analytic inverse DFT. (Compared at `atol=1e-5` because the emitter
  rounds rotation angles to 6 decimals.)
- End-to-end: measured phases concentrate on the grid `s/(2**n - 1)`, and the
  benchmark's own `evaluate_benchmark` returns success probabilities tracking
  `phi(2**n - 1)/(2**n - 1)` for n = 3, 4, 5 (e.g. n=3: 0.82 vs 0.857 ideal).
  The small shortfall is finite control-register resolution and shrinks as
  `control_register_offset` grows. This also confirms bit-order consistency
  between the emitted QASM, the measurement mapping, the iQFT swaps, and
  `int(bitstring, 2)` in the evaluator.

**Round-trip note:** loading the emitted QASM into Qiskit needs
`target_sdk="qiskit"` (the default emitter omits `include "qelib1.inc"`, leaving
`h` etc. undefined) and `qasm2.loads(..., custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)`
(Qiskit's strict QASM2 parser does not define `swap` otherwise).

## Open items

1. **Run the pytest suite** (`test/test_shorbenchmark.py`) — not run since the
   synthesis rewrite (deferred by request; existing tests only cover the
   polynomial/Mersenne helpers, so no failures are expected).
2. **Add pytest coverage for the synthesis and the end-to-end run** — promote the
   notebook's `apply_schedule` replay check and the `phi(r)/r` success check into
   the test suite (the latter would have caught the classical-register bug).
3. ~~End-to-end simulation section~~ — done (notebook sections 3–6, 2026-07-21).
4. ~~Commit the module~~ — done (committed in `04aa40d`).

## Environment notes

- Qiskit 2.5.0 and qiskit-aer 0.17.2 were installed ad hoc into the project venv
  (`pip install qiskit qiskit-aer`) for the synthesis comparison and the
  end-to-end simulation. They are **not** package dependencies and nothing in
  `frontier/` imports them; only the notebook's comparison/simulation cells use
  them. Remove with `pip uninstall qiskit qiskit-aer` if undesired.
