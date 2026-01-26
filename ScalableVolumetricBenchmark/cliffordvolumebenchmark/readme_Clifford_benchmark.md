# Clifford Volume Benchmark

## Table of Contents

- [Background](#background)
- [Benchmark Task](#benchmark-task)
  - [Clifford Group and Stabilizer States](#clifford-group-and-stabilizer-states)
  - [Random Clifford Instances](#random-clifford-instances)
- [Benchmark Protocol](#benchmark-protocol)
  - [Step-by-Step Procedure](#step-by-step-procedure)
- [Performance Criteria](#performance-criteria)
- [Implementation Notes](#implementation-notes)
  - [Circuit Synthesis](#circuit-synthesis)
  - [Measurement of Pauli Operators](#measurement-of-pauli-operators)
- [Benchmark Score and Interpretation](#benchmark-score-and-interpretation)
- [How to Use the Clifford Volume Benchmark](#how-to-use-the-clifford-volume-benchmark)

## Background

The Clifford Volume Benchmark (CV) is a **device-level volumetric benchmark** designed to test how reliably a quantum processor can implement **random $n$-qubit Clifford unitaries**. The benchmark focuses on **algorithmic primitives** rather than full fault-tolerant algorithms and aims to:

- remain **classically verifiable** at all tested sizes,
- be **platform independent** (no fixed connectivity or native gate assumptions),
- probe a **practically relevant** class of circuits.

The benchmark leverages the Gottesman–Knill theorem: any circuit composed of Clifford gates, stabilizer-state preparations, and Pauli-basis measurements can be **simulated efficiently on a classical computer**. This makes it possible to scale the benchmark to relatively large widths while retaining a trusted reference.

---

## Benchmark Task

### Clifford Group and Stabilizer States

The **$n$-qubit Clifford group** $\mathcal{C}(n)$ is the set of unitary operators (up to global phase) that map the $n$-qubit Pauli group $\mathcal{P}_n$ to itself under conjugation:

$$
\mathcal{C}(n)
= \left \{
U \in \mathrm{U}(2^n)
\,| \,
U P U^\dagger \in \mathcal{P}_n
\text{ for all } P \in \mathcal{P}_n
\right\} / \mathrm{U}(1).
$$

Starting from $\ket{0}^{\otimes n}$, any Clifford unitary $C \in \mathcal{C}(n)$ prepares a **stabilizer state**

$$
\rho_C
= C \, |0\rangle \langle 0 |^{\otimes n} \, C^\dagger .
$$

A stabilizer state is uniquely characterized by an abelian group $\mathcal{S}$ of $2^n$ commuting Pauli operators with eigenvalue $+1$ on that state (the *stabilizer group*). A minimal **generating set** $\mathcal{G} \subset \mathcal{S}$ contains only $n$ independent $n$-qubit Pauli operators:

$$
\rho
= |\psi \rangle \langle \psi |
= \frac{1}{2^n} \prod_{P_\ell \in \mathcal{G}} (I + P_\ell)
= \frac{1}{2^n} \sum_{P_\ell \in \mathcal{S}} P_\ell .
$$

For any Pauli operator $P$ in a full generator set of the Pauli group ($2n$ independent elements in total), its expectation value in $\rho$ is

$$
\langle P \rangle
= \operatorname{Tr}(\rho P) =
\begin{cases}
1, & P \in \mathcal{S}, \\\\[2mm]
0, & P \notin \mathcal{S}.
\end{cases}
$$

Thus, **stabilizers** have ideal expectation value **1**, while appropriately chosen **destabilizers** (generators outside the stabilizer group) have ideal expectation value **0**.

### Random Clifford Instances

For each qubit number $n$, the Clifford Volume Benchmark considers:

- an ensemble of **random $n$-qubit Clifford unitaries** $C^m$,
- each realized as a quantum circuit compiled for the target hardware,
- starting from the initial state $\ket{0}^{\otimes n}$.

For each sampled Clifford, the corresponding stabilizer group is determined. A set of **stabilizer generators** $\{ \mathcal{S}_i^m \}$ and an equal number of **non-stabilizer Pauli operators** (destabilizers) $\{ \mathcal{D}_i^m \}$ is then selected. These operators form the **probe set** used to verify whether the device implemented $C^m$ correctly.

---

## Benchmark Protocol

### Step-by-Step Procedure

The benchmark is evaluated for increasing numbers of qubits $n$. For each $n$, the following steps are performed.

1. **Select the width $n$**  
   Begin with $n = 1$ and increment by 1 until the benchmark fails or the device limit is reached.

2. **Sample random Clifford unitaries**  
   Sample $M = 10$ random $n$-qubit Cliffords $C^m$ from $\mathcal{C}(n)$.

3. **Determine stabilizer and destabilizer sets**  
   For each Clifford $C^m$:

   - Compute an $n$-element set of **stabilizer generators** $\{ \mathcal{S}_i^m \}$ for
     $\rho_C = C^m \, |0 \rangle \langle 0 |^{\otimes n} \, C^{m\dagger}$.
   - Choose $n$ additional **Pauli operators outside the stabilizer group** $\{ \mathcal{D}_i^m \}$ such that together they generate the full $n$-qubit Pauli group.

   To keep the number of distinct measurement settings manageable, the number of operators actually measured per Clifford is limited to $N=4$.

   A random subset of size $N$ is taken from stabilizers and another $N$ from destabilizers.

4. **Prepare and compile circuits**  
   For each $C^m$:

   - Build a circuit implementing $C^m$ using the native gate set and connectivity of the device.
   - Apply any allowed optimizations (architecture-aware synthesis, approximate compilation, ancillas, etc.).

5. **Run the circuits and measure Pauli operators**  
   For each selected Pauli operator $P \in \{ \mathcal{S}_i^m \} \cup \{ \mathcal{D}_i^m \}$:

   - Initialize the system in $\ket{0}^{\otimes n}$.
   - Apply the circuit for $C^m$.
   - Apply basis-change gates (single-qubit Clifford rotations) to map $P$ to a product of $Z$-operators.
   - Measure all qubits in the computational basis.
   - From the resulting bitstrings, compute the eigenvalue of $P$ and estimate $\langle P \rangle$ by repeating the procedure at least

     $$
     L = 512
     $$

     times (shots).

   This yields:
   - $\langle \mathcal{S}_i^m \rangle$ — measured stabilizer expectations,
   - $\langle \mathcal{D}_i^m \rangle$ — measured destabilizer expectations.

6. **Check success conditions**  
   For the given width $n$, the benchmark is considered successful if all measured values satisfy the criteria below.

   The benchmark is designed to detect **worst-case failures**, not just good average performance. Accordingly, for each width $n$, the device passes if:

   - all stabilizer expectation values exceed the stabilizer threshold by **at least two standard deviations**,
   - all destabilizer expectation values remain below the destabilizer threshold by **at least two standard deviations**.

   Let $\sigma_{S,i}^m$ and $\sigma_{D,i}^m$ denote the estimated standard deviations of the measured stabilizer and destabilizer expectation values, respectively. The success conditions are:

   $$
   \begin{cases}
   \displaystyle
   \min_{m,i}
   \big( \langle \mathcal{S}_i^m \rangle - 2 \sigma_{S,i}^m \big)
   \ge
   \tau_S, \\\\[3mm]
   \displaystyle
   \max_{m,i}
   \big( |\langle \mathcal{D}_i^m \rangle| + 2 \sigma_{D,i}^m \big)
   \le
   \tau_D.
   \end{cases}
   $$

   Recommended default thresholds:

   - **Stabilizers**:

     $$
     \tau_S = \frac{1}{e}
     $$

     corresponding to the standard $1/e$ benchmark threshold, with the additional requirement that measured values lie at least two standard deviations above this limit.

   - **Destabilizers**:

     $$
     \tau_D = \frac{1}{2e}
     $$

     with the requirement that measured values remain at least two standard deviations below this bound. This ensures that deviations from the ideal value 0 cannot be explained by statistical fluctuations alone.

  In addition to the worst-case conditions above, we also impose a requirement on the **average performance** over the ensemble. Specifically, that the **mean stabilizer expectation value** lies at least **five standard deviations above** the stabilizer threshold, and that the **mean destabilizer expectation value** lies at least **five standard deviations below** the destabilizer threshold.

  Let $\overline{\langle \mathcal{S} \rangle}$ and $\overline{\langle \mathcal{D} \rangle}$ denote the averages over all measured stabilizer and destabilizer expectation values at fixed width $n$, with corresponding standard deviations $\sigma_{\overline{S}}$ and $\sigma_{\overline{D}}$. The additional conditions are:

  $$
  \begin{cases}
  \displaystyle
  \overline{\langle \mathcal{S} \rangle} - 5 \sigma_{\overline{S}}
  \ge
  \tau_S, \\\\[3mm]
  \displaystyle
  \big| \overline{\langle \mathcal{D} \rangle} \big| + 5 \sigma_{\overline{D}}
  \le
  \tau_D.
  \end{cases}
  $$

  This average-performance criterion complements the worst-case tests by ensuring that the device not only avoids isolated failures but also exhibits **robust, statistically significant fidelity across the full ensemble of random Clifford instances**.


7. **Increase $n$**  
   If the benchmark passes for width $n$, repeat from step 1 with $n+1$.  
   The final score is the largest $n$ for which all criteria are satisfied for all smaller widths.

---

## Implementation Notes

### Circuit Synthesis

The benchmark is **implementation agnostic**. Any compilation strategy is allowed as long as the implemented circuit realizes the intended Clifford unitary (or a sufficiently accurate approximation, if this is part of the design).

The **benchmark score depends only on the logical width $n$** of the Clifford operator, not on the number of physical qubits or gates.

### Measurement of Pauli Operators

Most hardware platforms measure in the **computational basis** only. To measure an arbitrary Pauli string:

- For each qubit:
  - measure $Z$ directly,
  - measure $X$ via an $H$ gate followed by $Z$,
  - measure $Y$ via $S^\dagger$ then $H$ then $Z$,
  - ignore identity terms.

The eigenvalue of the full Pauli string is the product of the eigenvalues on the affected qubits. Repeating this over many shots yields an estimate of its expectation value.

---

## Benchmark Score and Interpretation

The **Clifford Volume score** is defined as:

> the largest integer $n_{\text{max}}$ such that the benchmark success conditions are satisfied for **all widths** $1 \le n \le n_{\text{max}}$.

This provides a **single-number summary** of how many qubits the device can use to reliably implement **general random Clifford unitaries**.

Because the benchmark is based on **random Clifford unitaries**, it probes the device over a wide variety of stabilizer structures and is sensitive to:

- gate errors,
- readout errors,
- crosstalk and correlated noise,
- compilation and routing overhead.

This makes the Clifford Volume Benchmark a **scalable, application-relevant volumetric benchmark** that sits naturally between simple gate-level protocols and full algorithmic benchmarks.

## How to Use the Clifford Volume Benchmark

This section explains how the Clifford Volume Benchmark codebase is intended to be used, from benchmark generation to evaluation. 

---

#### Overview of the Workflow

Using the benchmark consists of the following conceptual steps:

1. Instantiate a benchmark for a fixed number of qubits  
2. Generate benchmark samples (random Clifford circuits + measurements)  
3. Export circuits and execute them on a backend of choice  
4. Attach experimental measurement results  
5. Evaluate benchmark conditions  
6. Inspect numerical results and plots

Each step is handled by the benchmark framework and does not assume a specific quantum SDK or hardware platform.

---

### Benchmark Initialization

The Clifford Volume Benchmark is represented by the `CliffordVolumeBenchmark` class.

At initialization the user specifies:
- the number of qubits (`number_of_qubits`)
- the number of random Clifford samples (`sample_size`)
- the number of measurement shots per circuit (`shots`)
- the output circuit format and target SDK (for QASM export)

These parameters fully define the benchmark configuration.

```python
from benchmarks import CliffordVolumeBenchmark

# Configure the benchmark
benchmark = CliffordVolumeBenchmark(
    number_of_qubits=5,
    sample_size=4,   # number of random Clifford instances
    shots=1024,       # shots per circuit (used for error bars and thresholds)
    format="qasm2",
    target_sdk="qiskit",
)
```


---

### Benchmark Generation

Calling the benchmark generation step produces the full benchmark dataset in memory and optionally saves it as a JSON file.

During this step, the benchmark:
- samples random Clifford operators using a stabilizer tableau representation
- synthesizes each Clifford into a quantum circuit
- identifies stabilizer and destabilizer Pauli operators
- selects a fixed number of operators to measure
- creates one measurement circuit per selected operator
- exports all circuits to QASM

Each benchmark sample corresponds to a single random Clifford operator and contains multiple measurement circuits.
```python
# Generate samples (and, by default, auto-save a JSON file under .benchmarks/)
bench.create_benchmark()
```
After generation, the benchmark object contains:
- `bench.samples`: list of samples (each sample is one random Clifford instance)
- `bench.samples[*]["circuits"]`: circuits for stabilizer/destabilizer measurements
- `circuit["qasm"]`: QASM string to run on a backend
- `circuit["observable"]`: Pauli string like `"+XZI..."` used for expectation values

---
### Inspect and export circuits

```python
# List all circuit IDs (canonical order)
circuit_ids = bench.get_all_circuit_ids()

# Grab the first circuit payload
sample0 = bench.samples[0]
circ0 = sample0["circuits"][0]

print(circ0["circuit_id"])
print(circ0["observable"])   # e.g. "+XZI..."
print(circ0["qasm"][:200])   # QASM prefix
```

### Circuit Execution (External)

The benchmark framework does not execute circuits itself.

Instead, the user:
- extracts the generated QASM circuits
- runs them on a simulator or quantum device of choice
- collects raw measurement counts for each circuit

Measurement results must be stored as mappings from bitstrings to counts.

Required shape (keyed by `circuit_id`):

```python
counts_by_circuit_id = {
    "0_stab_0": {"00000": 520, "11111": 504},
    "0_destab_0": {"00000": 510, "11111": 514},
    # ...
}
```
or 

```python
counts_list = [
  {"00000": 520, "11111": 504},
  {"00000": 510, "11111": 514},
  # ...  
  ]}
```

> Keep the backend’s bitstring convention (endianness) consistent across all circuits.
---

### Step 4: Attaching Experimental Results

Once execution is complete, experimental results can be attached to the benchmark.

```python
bench.add_experimental_results(
    counts_data=counts_by_circuit_id,
    platform="my-backend",
    experiment_metadata={"notes": "first run"},
)
```
This stores results under `bench.experimental_results["results"][circuit_id]["counts"]`. The results are stored inside the benchmark object and merged into the JSON structure.

At this point, the benchmark has access to:
- circuit definitions
- observables (Pauli strings)
- shot counts for each circuit

This enables automated expectation value estimation and statistical analysis.

---

### Benchmark Evaluation

Evaluating the benchmark applies the Clifford Volume success criteria.

```python
evaluation = bench.evaluate_benchmark()
```

Evaluation writes detailed per-sample diagnostics into `bench.experimental_results["evaluation"]`.


### Inspecting Results

When you run `evaluate_benchmark()`, it prints a detailed evaluation report to the console, for example:

```text
==============================================================
 Clifford Benchmark Evaluation (5 qubits)
==============================================================
Thresholds: τ_S = 1/e = 0.367879   τ_D = 1/(2e) = 0.183940
Shots per circuit: 2048
--------------------------------------------------------------
Stabilizers (all measured):
  • mean ± std: 1.000000 ± 0.000000
  • min EV:     1.000000
Destabilizers (all measured):
  • mean ± std: -0.005371 ± 0.024034
  • max |EV|:   0.049805
--------------------------------------------------------------
Per-sample pass count: 10/10
Worst stabilizer margin (min over samples of <S>-2σ): sample 0 -> 1.000000
Worst destabilizer margin (max over samples of |<D>|+2σ): sample 9 -> 0.093944
Benchmark passed (all samples): True
==============================================================
```
```python
# Nested mapping: values[sample_id]["stabilizer"|"destabilizer"][observable] -> (EV, std_error)
values = bench.get_all_expectation_value()

# Plot one sample (stabilizers + destabilizers with error bars)
bench.plot_expected_values(sample_id=0)

# Plot all measured values across the benchmark
bench.plot_all_expectation_values()
```

---

### Reload an existing benchmark JSON

```python
from benchmarks.clifford_volume import CliffordVolumeBenchmark

bench = CliffordVolumeBenchmark.load_json("path/to/your_benchmark.json")

# If results are present, you can evaluate or plot immediately
bench.evaluate_benchmark()
```

### Benchmark Score and Interpretation

The Clifford Volume score is defined as:

> the largest integer $n_{\text{max}}$ such that the benchmark success conditions are satisfied for **all widths** $1 \le n \le n_{\text{max}}$.
