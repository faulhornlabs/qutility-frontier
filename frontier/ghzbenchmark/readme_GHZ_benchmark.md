# GHZ Benchmark

## Table of Contents

- [Background](#background)
- [GHZ State Generation](#ghz-state-generation)
  - [Star-Like Fanout](#star-like-fanout)
  - [Linear / Nearest-Neighbour Fanout](#linear--nearest-neighbour-fanout)
  - [Logarithmic-Depth Fanout](#logarithmic-depth-fanout)
- [Benchmark Protocol](#benchmark-protocol)
  - [Witness (Stabilizer) Measurements](#witness-stabilizer-measurements)
  - [Shadow-Overlap Measurements](#shadow-overlap-measurements)
- [Fidelity Estimation](#fidelity-estimation)
  - [Stabilizer-Based Fidelity Witness](#stabilizer-based-fidelity-witness)
  - [Shadow-Overlap Fidelity Witness](#shadow-overlap-fidelity-witness)
- [Implementation Notes](#implementation-notes)
- [References](#references)

## Background

Genuine multipartite entanglement is a key resource for quantum algorithms that
surpass classical approaches. The GHZ benchmark targets this resource directly:
it prepares and probes $n$-qubit GHZ states

$$
|\mathrm{GHZ}_n\rangle = \frac{1}{\sqrt{2}}\big(|0\rangle^{\otimes n} + |1\rangle^{\otimes n}\big),
$$

which are stabilizer states with well-defined Pauli generators. The benchmark is
designed to

- remain **classically verifiable** (stabilizer structure),
- support **multiple hardware topologies**, and
- provide **scalable certification** via two complementary measurement schemes:
  witness-based fidelity estimation and single-shot quantum shadows.

## GHZ State Generation

The benchmark creates the GHZ resource circuit once per sample and then derives
measurement circuits from that base circuit. Three fanout patterns are
available; all begin with a Hadamard on the root qubit.

### Star-Like Fanout

- Qubit 0 controls a CNOT onto each other qubit.
- Minimal depth on fully connected hardware.
- Matches the historical GHZ preparation used on ion-trap devices.

### Linear / Nearest-Neighbour Fanout

- A CNOT ladder: qubit $0 \rightarrow 1$, $1 \rightarrow 2$, etc.
- Works on line or heavy-hex style topologies without long-range gates.

### Logarithmic-Depth Fanout

- A binary-tree fanout: controls and targets are scheduled so that the depth scales as $O(\log n)$ when parallel layers are available.
- Useful when connectivity supports simultaneous CNOTs.

## Benchmark Protocol

Each benchmark sample stores:

1. A GHZ preparation circuit following the chosen topology.
2. Measurement circuits, either witness or shadow based.
3. Metadata describing the scheme, number of measurements, and basis choices.

### Witness (Stabilizer) Measurements

1. Start from the GHZ preparation circuit.
2. Produce two measurement circuits:
   - **X basis**: append Hadamards to every qubit, then measure all qubits.
   - **Z basis**: measure all qubits directly.
3. Record the QASM plus metadata (`basis = "x"` or `"z"`).
4. Execution on hardware or simulator yields histograms
   `counts_x`, `counts_z`.
5. Use the stabilizer witness to compute a fidelity lower bound.

### Shadow-Overlap Measurements

1. Start from the GHZ preparation circuit and apply Hadamards to all qubits.
2. For each round:
   - Randomly pick two distinct qubits.
   - For each selected qubit, randomly choose a Pauli basis (X, Y, Z) and apply
     the corresponding single-qubit rotations (H for X, S†+H for Y).
   - Measure all qubits; store the `basis_map` describing the two randomized
     bases.
3. The benchmark records every circuit’s QASM and basis metadata but leaves the
   number of rounds configurable.
4. Execution returns full bitstrings (or counts) per round that feed the
   shadow-overlap estimator.

## Fidelity Estimation

### Stabilizer-Based Fidelity Witness

The GHZ stabilizer generator set is

$$
\mathcal{G} = \left\{ X^{\otimes n}, Z_1 Z_2, Z_2 Z_3, \dots, Z_{n-1} Z_n \right\}.
$$

Only the two global measurement settings described above are required:

- **X basis**: expectation $\tilde{\mu}_0 = \langle X^{\otimes n} \rangle$,
  obtained from the parity (even minus odd) of the measured bitstrings.
- **Z basis**: expectations $\tilde{\mu}_k = \langle Z_k Z_{k+1} \rangle$ for
  $k=1,\dots,n-1$ using the parity of adjacent qubits.

The empirical GHZ fidelity lower bound is

$$
\hat{F}_{\min} = \max\left(0,\ 1 - \tfrac{1}{2} \sum_{l=0}^{n-1} (1 - \tilde{\mu}_l)\right),
$$

with an equivalent form

$$
\hat{F}_{\min} = \max\left(0,\ \frac{\tilde{\mu}_0 + \sum_{k=1}^{n-1} \tilde{\mu}_k - (n-2)}{2}\right).
$$

Variance estimates follow from the Bernoulli statistics of $\pm1$
measurement outcomes, enabling one-sided certification that
$F_{\text{true}} > 1/2$ at a specified confidence.

### Shadow-Overlap Fidelity Witness

The shadow-overlap method builds a classical “shadow” for a randomly selected
pair of qubits. For each round:

1. Choose two qubits and measure the remaining $n-2$ in $Z$.
2. Measure the selected pair in random Pauli bases.
3. Construct the projector $L_{z_k}$ from the GHZ amplitudes for the observed
   post-measurement pattern, and the local operator
   $\sigma = (3|s_1\rangle\langle s_1| - I) \otimes (3|s_2\rangle\langle s_2| - I)$
   from the measured basis/outcomes.
4. The per-round estimator is $\omega = \mathrm{Tr}(L_{z_k}\sigma)$.
5. Averaging $\omega_t$ over $T$ rounds yields $\bar{\omega}$, which lower-bounds
   the GHZ fidelity. For $T = O(\varepsilon^{-2} \log(1/\delta))$, the method
   certifies fidelity $> 1/2$ with confidence $1 - \delta$ when
   $\bar{\omega} \ge 1 - \varepsilon$.

## Implementation Notes

- `GHZBenchmark` lives in `frontier.ghzbenchmark` and shares the common
  `Benchmark` API (`create_benchmark`, `add_experimental_results`,
  `evaluate_benchmark`, etc.).
- Witness mode always produces two circuits per sample; shadow mode produces
  `measurement_rounds` circuits, each annotated with its two-qubit `basis_map`.
- Helper functions exposed at the package root:
  - `evaluate_fidelity(z_counts, x_counts)` — compute $\hat{F}_{\min}$ and its
    standard deviation.
  - `certify_fidelity_gt_half(f_min, std, confidence)` — certify
    $F > 1/2$ at the requested confidence level.
  - `evaluate_shadow_overlap(measurement_outcomes, basis_maps)` — compute the
    shadow-overlap mean and error bar directly from raw shots.

## References

1. A. Kalev, A. Kyrillidis, and N. M. Linke, “Validating and Certifying
   Stabilizer States,” *Phys. Rev. A* **99**, 042337 (2019),
   [arXiv:1808.10786](https://arxiv.org/abs/1808.10786).
2. H.-Y. Huang, J. Preskill, and M. Soleimanifar, “Certifying Almost All Quantum
   States with Few Single-Qubit Measurements,” *Phys. Rev. A* **110**, 052401
   (2024), [arXiv:2404.07281](https://arxiv.org/abs/2404.07281).
