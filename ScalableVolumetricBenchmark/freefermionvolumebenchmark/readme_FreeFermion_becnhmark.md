# Free-Fermion Volume Benchmark

## Table of Contents

- [Background](#background)
- [Benchmark Task](#benchmark-task)
  - [Free-Fermion Systems and Majorana Modes](#free-fermion-systems-and-majorana-modes)
  - [Random Free-Fermion Instances](#random-free-fermion-instances)
- [Benchmark Protocol](#benchmark-protocol)
  - [Step-by-Step Procedure](#step-by-step-procedure)
- [Performance Criteria](#performance-criteria)
- [Implementation Notes](#implementation-notes)
  - [Circuit Synthesis](#circuit-synthesis)
  - [Measurement of Majorana Operators](#measurement-of-majorana-operators)
- [Benchmark Score and Interpretation](#benchmark-score-and-interpretation)
 

## Background

The Free-Fermion Volume Benchmark (FFV) is a **device-level volumetric benchmark** designed to test how reliably a quantum processor can implement **random free-fermionic (Gaussian) unitaries** on $n$ qubits.

The benchmark focuses on **quadratic fermionic dynamics** rather than fully general many-body evolution, and aims to:

- remain **classically verifiable** at all tested sizes via efficient free-fermion simulation,
- be **platform independent** (no fixed connectivity or native gate assumptions),
- probe a **physically motivated and practically relevant** class of circuits (free fermions / matchgate circuits).

The key idea is that time evolution under a **quadratic fermionic Hamiltonian** can be encoded as an orthogonal transformation on **Majorana modes**, and this structure can be efficiently simulated on a classical computer. The FFV benchmark leverages this to define a scalable volumetric task with a clear, group-theoretic target: accurately representing elements of the group $\mathrm{SO}(2n)$ on a quantum device.

---

## Benchmark Task

### Free-Fermion Systems and Majorana Modes

Consider $n$ fermionic modes with annihilation and creation operators $f_j$, $f_j^\dagger$ obeying the canonical anti-commutation relations

$$
\{ f_j, f_k^\dagger \} = \delta_{jk}, \quad
\{ f_j, f_k \} = \{ f_j^\dagger, f_k^\dagger \} = 0.
$$

It is often convenient to work with **Majorana operators**:

$$
m_{2j-1} = f_j + f_j^\dagger, \qquad
m_{2j}   = -i (f_j - f_j^\dagger),
$$

which satisfy

$$
\{ m_j, m_k \} = 2 \delta_{jk}.
$$

A general **free-fermionic (Gaussian) Hamiltonian** is quadratic in Majorana operators:

$$
H = \frac{i}{4} \sum_{j,k=1}^{2n} A_{jk} m_j m_k,
$$

where $A$ is a real antisymmetric matrix ($A = -A^\top$). Time evolution in the Heisenberg picture preserves the linear span of Majoranas:

$$
m_j(t) = U^\dagger(t)\, m_j(0)\, U(t)
       = \sum_{k=1}^{2n} O_{jk}(t) \, m_k(0),
$$

where

$$
O(t) = e^{tA} \in \mathrm{SO}(2n)
$$

is a real orthogonal matrix. Thus, every free-fermionic unitary can be represented by an orthogonal transformation $O$ on the vector of Majorana operators.

The FFV benchmark exploits this one-to-one correspondence: benchmarking a free-fermion circuit is equivalent to checking whether the implemented transformation on Majorana modes matches a target matrix $O \in \mathrm{SO}(2n)$.

### Random Free-Fermion Instances

For each qubit number $n$, the Free-Fermion Volume Benchmark considers:

- an ensemble of **random orthogonal matrices** $O^m \in \mathrm{SO}(2n)$,
- each defining a target free-fermionic unitary on $n$ modes,
- with the implementation acting on $n$ qubits via a chosen fermion–qubit encoding (e.g. Jordan–Wigner).

Given a fixed orthogonal matrix $O$, one can define a corresponding **free-fermion unitary**

$$
F(O) = \exp\left( \frac{1}{4} \sum_{i,j=1}^{2n} [\log(O)]_{ij} m_i m_j \right)
$$

such that

$$
F^\dagger(O)\, m_j\, F(O) = \sum_{k=1}^{2n} O_{jk} m_k.
$$

To probe the action of this unitary, the benchmark uses specially chosen fermionic states. For each randomly chosen index $i \in \{1,\dots,2n\}$, we prepare a Majorana eigenstate $\rho_i$ for which:

$$
\langle m_j \rangle_{\rho_i} = \delta_{ij}.
$$

After applying the free-fermion unitary, the expectation values transform as

$$
\langle m_j \rangle_{F(O) \rho_i F^\dagger(O)} = O_{ji}.
$$

This relation allows us to **reconstruct selected entries of $O$** directly from measured expectation values of Majorana operators. By combining them into suitable linear combinations, we can test the fundamental orthogonality property

$$
\sum_k O_{ki} O_{kj} = \delta_{ij}
$$

using experimental data only.

---

## Benchmark Protocol

### Step-by-Step Procedure

The benchmark is evaluated for increasing numbers of qubits $n$. For each $n$, the following steps are performed.

1. **Select the width $n$**  
   Begin with $n = 2$ (free-fermionic dynamics are trivial on a single qubit) and increment by 1 until the benchmark fails or the device limit is reached.

2. **Sample random free-fermion transformations**  
   Sample $M = 10$ random orthogonal matrices $O^m \in \mathrm{SO}(2n)$, each defining a free-fermion unitary $F^m$.

3. **Select Majorana operators to be measured**  
   For each matrix $O^m$:

   - In principle, reconstructing all entries of each row/column would require measuring all $2n$ Majorana expectation values.
   - To keep the benchmark practical, only a subset of Majorana operators is measured, of size

     $$
     N(n) =
     \begin{cases}
     2n, & n \le 10, \\\\[2mm]
     20 + \left\lfloor \dfrac{n}{5} \right\rfloor, & n > 10.
     \end{cases}
     $$

   - The selected indices correspond to those entries of $O^m$ with **largest absolute values**, so that their contributions dominate the relevant linear combinations.

4. **Prepare and compile circuits**  
   For each $O^m$:

   - Construct a quantum circuit implementing the corresponding free-fermionic unitary $F^m$ on $n$ qubits.
   - This can be done, for example, by decomposing $O^m$ into Givens rotations and mapping them to single- and two-qubit gates via a fermion–qubit encoding.
   - Compile the circuit using the native gate set and connectivity of the device, applying any allowed optimizations (routing, approximate synthesis, ancillas, etc.).

5. **Run the circuits and measure Majorana operators**  
   For each $m \in \{1,\dots,M\}$:

   - Choose an index $i \in \{1,\dots,2n\}$ and prepare a fermionic state $\rho_i$ for which $\langle m_j \rangle_{\rho_i} = \delta_{ij}$. This can be implemented by preparing a specific product state of qubits under the chosen encoding.
   - Apply the circuit corresponding to $F^m$.
   - For each selected Majorana index $k$ in the measurement set for this $O^m$:
     - Map the Majorana operator $m_k$ to a **Pauli string** on the qubits using the fermion–qubit encoding (e.g., Jordan–Wigner).
     - Apply appropriate single-qubit Clifford gates to rotate this Pauli string into a product of $Z$-operators.
     - Measure all qubits in the computational basis.
     - From the bitstrings, compute the eigenvalue of the Pauli string, and hence estimate $\langle m_k \rangle_{F^m \rho_i F^{m\dagger}}$ by repeating the procedure at least

       $$
       L = 512
       $$

       times (shots).

   This yields a set of estimated expectation values

   $$
   \big\{ \langle m_k \rangle_{F^m \rho_i F^{m\dagger}} \big\}
   $$

   for the selected Majorana operators.

6. **Form parallel and orthogonal linear combinations**  
   For each $O^m$, we distinguish two kinds of linear combinations:

   - A **parallel combination**, where the row of $O^m$ associated with the initial index $i$ is used. In the ideal, noise-free case, this combination should be close to **1**.
   - An **orthogonal combination**, where a different row (corresponding to some $j \ne i$) is used. In the ideal case, this combination should be close to **0**.

   Using only the selected subset of indices $J$ (the measured entries), we define reduced linear combinations

   $$
   O^m_{i} \circ \langle m \rangle \quad \text{and} \quad
   O^m_{j} \circ \langle m \rangle,
   $$

   constructed by summing $O^m_{ki} \langle m_k \rangle$ (or $O^m_{kj} \langle m_k \rangle$) over $k \in J$, and normalizing by the corresponding sum of squared coefficients. This preserves the ideal values 1 (parallel) and 0 (orthogonal), while requiring only a reduced set of measurements.

7. **Check success conditions**  
   For the given $n$, the benchmark is considered successful if the measured linear combinations satisfy the criteria below (see **Performance Criteria**) for all $M$ random instances.

8. **Increase $n$**  
   If the benchmark passes for width $n$, repeat from step 1 with $n+1$.  
   The final score is the largest $n$ for which all criteria are satisfied for all smaller widths.

---

## Performance Criteria

For each width $n$ and each sampled free-fermion unitary $F^m$ (defined by $O^m$), we obtain:

- a **parallel combination** $O^m_{i} \circ \langle m \rangle$, ideally equal to 1,
- one or more **orthogonal combinations** $O^m_{j} \circ \langle m \rangle$ (with $j \ne i$), ideally equal to 0.

The device passes for width $n$ if:

- all parallel combinations stay above a threshold $\tau_\parallel$,
- all orthogonal combinations remain small in magnitude (below $\tau_\perp$).

Formally:

$$
\begin{cases}
\displaystyle
\min_{m}
\left( O^m_{i} \circ \langle m \rangle \right)
\;\ge\;
\tau_\parallel, \\\\[3mm]
\displaystyle
\max_{m}
\big|
O^m_{j} \circ \langle m \rangle
\big|
\;\le\;
\tau_\perp.
\end{cases}
$$

Recommended default thresholds:

- **Parallel combinations**:

  $$
  \tau_\parallel = \frac{1}{e},
  $$

  analogous to the stabilizer threshold in the Clifford Volume Benchmark, reflecting exponential decay under typical noise models.

- **Orthogonal combinations**:

  $$
  \tau_\perp = \frac{1}{2e},
  $$

  small but safely above typical statistical fluctuations; any larger systematic value would indicate a significant violation of orthogonality due to noise or coherent errors.

These thresholds ensure that:

- the device is able to **faithfully represent the structure of $\mathrm{SO}(2n)$**, and
- the benchmark is sensitive to both amplitude damping and coherent distortions of the free-fermion dynamics.

---

## Implementation Notes

### Circuit Synthesis

The FFV benchmark is **implementation agnostic**. Any compilation strategy is allowed as long as the implemented circuit realizes the intended free-fermionic unitary (or a sufficiently accurate approximation).

Possible approaches include:

- decomposing $O^m \in \mathrm{SO}(2n)$ into a sequence of **Givens rotations** and mapping them to single- and two-qubit gates via Jordan–Wigner,
- using matchgate-based constructions,
- incorporating ancilla qubits, SWAP networks, or optimized layout strategies.

The **benchmark score depends only on the logical width $n$** of the free-fermion operation, not on the number of physical qubits or detailed gate counts.

### Measurement of Majorana Operators

Under a fermion–qubit encoding (e.g. Jordan–Wigner), each Majorana operator $m_k$ maps to a **Pauli string** acting on the qubits, typically of the form

$$
m_{2p-1} \mapsto Z_1 Z_2 \cdots Z_{p-1} X_p, \qquad
m_{2p}   \mapsto Z_1 Z_2 \cdots Z_{p-1} Y_p.
$$

To measure $\langle m_k \rangle$ on a quantum device:

- map the corresponding Pauli string to a product of $Z$-operators using single-qubit Clifford gates:
  - measure $Z$ directly,
  - measure $X$ via an $H$ gate followed by $Z$,
  - measure $Y$ via $S^\dagger$ then $H$ then $Z$,
- measure all qubits in the computational basis,
- compute the eigenvalue of the Pauli string from the measurement outcomes (the product of the relevant single-qubit eigenvalues),
- repeat for $L$ shots to estimate the expectation value.

As in the Clifford Volume Benchmark, this measurement strategy is compatible with typical hardware constraints (computational-basis readout only).

---

## Benchmark Score and Interpretation

The **Free-Fermion Volume score** is defined as:

> the largest integer $n_{\max}$ such that the benchmark success conditions are satisfied for **all widths** $2 \le n \le n_{\max}$.

This provides a **single-number summary** of how many qubits the device can use to reliably implement **general random free-fermionic unitaries**.

Because the benchmark is based on **random elements of $\mathrm{SO}(2n)$** and tests orthogonality via Majorana expectation values, it probes the device over a wide variety of Gaussian structures and is sensitive to:

- gate errors in the underlying matchgate / free-fermion circuits,
- readout errors,
- crosstalk and correlated noise,
- compilation and routing overhead that affect the effective free-fermion dynamics.

This makes the Free-Fermion Volume Benchmark a **scalable, physically motivated volumetric benchmark** that complements the Clifford Volume Benchmark by probing a different but still efficiently simulable, subset of the unitary group.


## How to Use the Free-fermion Volume Benchmark

This section explains how the Free-Fermion Volume Benchmark codebase is intended to be used, for benchmark generation and evaluation.

---

#### Overview of the Workflow

Using the benchmark consists of the following conceptual steps:

1. Instantiate a benchmark for a fixed number of qubits  
2. Generate benchmark samples (random free-fermion circuits + measurements)  
3. Export circuits and execute them on a backend of choice  
4. Attach experimental measurement results  
5. Evaluate benchmark conditions  
6. Inspect numerical results and plots  

Each step is handled by the benchmark framework and does not assume a specific quantum SDK or hardware platform.

---

### Benchmark Initialization

The Free-Fermion Volume Benchmark is represented by the `FreeFermionVolumeBenchmark` class.

At initialization the user specifies:
- the number of qubits (`number_of_qubits`)
- the number of random samples (`sample_size`)
- the number of measurement shots per circuit (`shots`)
- the output circuit format and target SDK (for QASM export)

```python
from benchmarks import FreeFermionVolumeBenchmark

bench = FreeFermionVolumeBenchmark(
    number_of_qubits=5,
    sample_size=10,
    shots=2048,
    format="qasm2",
    target_sdk="qiskit",
)
```

---

### Benchmark Generation

Calling the benchmark generation step produces the full benchmark dataset in memory and optionally saves it as a JSON file.

```python
# Generate samples (and, by default, auto-save a JSON file under .benchmarks/)
bench.create_benchmark()
```

After generation, the benchmark object contains:
- `bench.samples`: list of samples (each sample is one random free-fermion instance)
- `sample["circuits"]`: measurement circuits
- `circuit["qasm"]`: QASM string to run on a backend
- `circuit["observable"]`: Pauli string used to compute expectation values

---

### Inspect and export circuits

```python
# List all circuit IDs (canonical order)
circuit_ids = bench.get_all_circuit_ids()

# Grab the first circuit payload
sample0 = bench.samples[0]
circ0 = sample0["circuits"][0]

print(circ0["circuit_id"])
print(circ0["observable"])   # Pauli string for the measured observable
print(circ0["qasm"][:200])   # QASM prefix
```

---

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
    "0_0": {"00000": 520, "11111": 504},
    "0_1": {"00000": 510, "11111": 514},
    # ...
}
```

Alternatively, you may provide a list aligned with the benchmark’s circuit traversal order (same order as `bench.get_all_circuit_ids()`):

```python
counts_list = [
    {"00000": 520, "11111": 504},  # for circuit_ids[0]
    {"00000": 510, "11111": 514},  # for circuit_ids[1]
    # ...
]
```

> Keep the backend’s bitstring convention (endianness) consistent across all circuits.

---

### Attaching Experimental Results

Once execution is complete, experimental results can be attached to the benchmark.

```python
bench.add_experimental_results(
    counts_data=counts_by_circuit_id,
    platform="my-backend",
    experiment_metadata={"notes": "first run"},
)
```

This stores results under `bench.experimental_results["results"][circuit_id]["counts"]` and enables evaluation.

---

### Benchmark Evaluation

Evaluating the benchmark applies the Free-Fermion Volume success criteria and computes two projected metrics per sample:

- **parallel projected values** (should be near **1**)  
- **orthogonal projected values** (should be near **0**)

```python
evaluation = bench.evaluate_benchmark()
```

Evaluation writes per-sample diagnostics into `bench.experimental_results["evaluation"]`, including:
- `parallel_values`
- `orthogonal_values`
- any thresholds / pass flags produced by the evaluator

---

### Inspecting Results

When you run `evaluate_benchmark()`, it prints a summary report to the console, for example:

```text
==============================================================
 Free-Fermion Benchmark Evaluation (5 qubits)
==============================================================

Parallel projected values (should be near 1):
  • average: 0.98xxxx ± 0.0xxxx
  • lowest measured value: 0.9xxxx

Orthogonal projected values (should be near 0):
  • average: 0.0xxxx ± 0.0xxxx
  • highest absolute value: 0.0xxxx

==============================================================

Benchmark passed: True
==============================================================
```

You can also access the stored values directly:

```python
parallel = bench.experimental_results["evaluation"]["parallel_values"]
orthogonal = bench.experimental_results["evaluation"]["orthogonal_values"]

print("min(parallel):", min(parallel))
print("max(|orthogonal|):", max(abs(x) for x in orthogonal))
```

---

### Plotting

```python
# Plot parallel values and |orthogonal| values across all samples (with error bars)
bench.plot_all_expectation_values()

# Plot histograms of the projected-value distributions
bench.plot_expectation_histograms(bins=20)
```

---

### Reload an existing benchmark JSON

```python
from benchmarks.free_fermion_volume import FreeFermionVolumeBenchmark

bench = FreeFermionVolumeBenchmark.load_json("path/to/your_benchmark.json")

# If results are present, you can evaluate or plot immediately
bench.evaluate_benchmark()
bench.plot_all_expectation_values()
```

