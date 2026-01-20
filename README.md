# ScalableVolumetricBenchmark

Project names:
- BenchmarQ
- BenchmarkQit
- SQallable
- VolumetricQ

**ScalableVolumetricBenchmark** ($\leftarrow$ replace later) is an open-source Python package for implementing scalable and hardware-agnostic quantum benchmarking protocols. The package provides implementations of recently proposed volumetric benchmarks and offers tools to generate benchmark instances in a reproducible form.

In contrast to component-level tests, this benchmark suite targets system-level characterization, aiming to capture the computational performance of the full quantum processor. It focuses on **volumetric benchmarks** — protocols designed to map the performance of the entire quantum processor (end-to-end), rather than benchmarking isolated components.

Benchmarking quantum devices at scale is challenging, in particular because many benchmark protocols rely on quantum algorithms that do not scale well with system size.

In addition, the lack of standardization across quantum SDKs and provider workflows creates a significant incompatibility gap: applications and algorithms are often difficult to realize across different platforms and may require multiple independent implementations. This makes cross-platform comparison hard and  inefficient.

This project addresses both issues by:

- providing **scalable, platform-independent** volumetric benchmarks, and
- representing benchmark circuits in a **simple intermediate format** based on **OpenQASM**, so that the same benchmark instance can be exported and executed across multiple platforms.

This package does **not** execute circuits directly on hardware providers. Instead, it generates benchmark circuits in **OpenQASM**, which can be run using the provider’s own recommended workflow.

> *Development status:* This project is under active development.

---

## Key features

### Framework utilities

* Open-source Python package designed to simplify the implementation of **platform-independent quantum benchmark protocols**.
* A **benchmark *base* class** with a well-defined internal structure and workflow, including customizable methods for:
  * benchmark instance creation,
  * circuit generation,
  * serialization / saving,
  * loading and re-evaluation.
* A lightweight **Python-based representation of quantum circuits**, enabling intuitive and flexible implementation of benchmark logic while remaining independent of any specific SDK.
* **Hardware-agnostic circuit export** via **OpenQASM** (QASM 2 / QASM 3), with optional SDK-specific adaptations (e.g., gate aliasing).
* A **JSON schema** to store complete benchmark instances, including:
  * benchmark metadata and generated circuits,
  * experimental results (shot counts),
  * evaluation results (scores, pass/fail conditions, and derived metrics),
  together with utilities for saving and reloading benchmark instances reproducibly.

### Implemented scalable volumetric benchmarks

This package currently includes two implementations of scalable volumetric benchmarks introduced in the accompanying paper: https://arxiv.org/abs/2512.19413 :

* **Clifford Volume Benchmark**

  * efficiently verifiable using stabilizer techniques
  * measures stabilizer and destabilizer observables to quantify device performance

* **Free-Fermion Volume Benchmark**

  * based on Gaussian / free-fermionic circuits (SO(2n) transformations)
  * evaluates device performance through Majorana-mode based observables

### Tutorials and demos

* Notebooks, including tutorials and demos, demonstrating the usage of the benchmarks and provided utilities are available in the `notebooks/` folder.

---


## Requirements

- Python **>= 3.8, < 3.12**
- Required dependencies:
- `numpy >= 1.21`
- `scipy >= 1.8` 
- `matplotlib >= 3.5`
- `stim >= 1.12`

**Optional:**
- `jsonschema` — only needed if you want to validate benchmark JSON files against the schema.

**Development tool dependencies (optional)**
The following tools are not required for end users, but are commonly useful during development:
- `pytest` (testing)
- `ruff` / `flake8` (formatting, linting)
- `mypy` (type checking)
- `jupyter` / `notebook` (running tutorial notebooks)

---

## Installation

Clone the repository and install the package in **editable mode** (`pip install -e .`). Editable mode is recommended during development, since changes in the source code are applied immediately without reinstalling.

### Option 1: Setup with `venv` (standard Python)
```bash
git clone <REPO_URL>
cd ScalableVolumetricBenchmark

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install the package
pip install --upgrade pip
pip install -e .
```

### Option 2: Setup with `conda`

```bash
git clone <REPO_URL>
cd ScalableVolumetricBenchmark

# Create and activate a conda environment
conda create -n scalablevolumetricbenchmark python=3.11
conda activate scalablevolumetricbenchmark

# Install the package
pip install --upgrade pip
pip install -e .
```

---

## Quickstart

### 1) Generate a benchmark instance

Example: Clifford Volume Benchmark.

```python
from ScalableVolumetricBenchmark import CliffordVolumeBenchmark
from ScalableVolumetricBenchmark import QasmEmitterOptions

emitter = QasmEmitterOptions(format="qasm3", target_sdk="qiskit")  # or "braket", "tket", or None
bench = CliffordVolumeBenchmark(
  number_of_qubits=5,
  sample_size=10,
  emitter_options=emitter,
  shots=512,
)

bench.create_benchmark()  # generates samples and (by default) auto-saves JSON under .benchmarks/
```

Access the generated circuits:

```python
# Flat list of OpenQASM programs (one per circuit)
qasm_programs = bench.get_all_circuits()

# Flat list of circuit IDs in the same order
circuit_ids = bench.get_all_circuit_ids()

# Access full structure (including observable strings)
samples = bench.samples
first_circuit = samples[0]["circuits"][0] # <- sample index and circuit index 
print(first_circuit["circuit_id"])
print(first_circuit["observable"])
print(first_circuit["qasm"])
```

### 2) Execute circuits on your provider of choice

- Convert/import each OpenQASM program to your platform.
- Execute and collect **counts**: a mapping from bitstring → integer count.

### 3) Attach results back to the benchmark JSON

Provide counts as a dictionary keyed by `circuit_id`:

```python
counts_by_circuit_id = {
  "0_stab_0": {"00000": 260, "00001": 252},
  "0_destab_0": {"00000": 255, "11111": 257},
  # ...
}

#or as an list of counts ordered as the circtuis
list_of_counts = [
{"00000": 260, "00001": 252},
{"00000": 255, "11111": 257},
  # ...
]
bench.add_experimental_results(
  counts_by_circuit_id,
  #list_of_counts,
  platform="my_provider",
  experiment_id="run_001",
)
```

### 4) Evaluate and obtain the benchmark score

```python
evaluation = bench.evaluate_benchmark()
print(evaluation)
```

Some benchmarks also provide built-in plotting helpers, for details see the documentation of the benchmarks.

---

## Benchmarks

### Clifford Volume Benchmark

The Clifford Volume Benchmark samples random **n-qubit Clifford unitaries**, then probes the output state using a set of measured **stabilizers** (ideal expectation value 1) and **destabilizers** (ideal expectation value 0). The benchmark passes for width *n* when stabilizers stay above a threshold and destabilizers stay below a threshold in magnitude.

See: `readme_Clifford_benchmark.md` for the full protocol and interpretation.

### Free-Fermion Volume Benchmark

The Free-Fermion Volume (FFV) Benchmark samples random **SO(2n)** transformations (Gaussian/free-fermionic unitaries), constructs circuits from a decomposition into elementary rotations, and evaluates the device by measuring Majorana-mode observables (mapped to Pauli strings). It checks “parallel” and “orthogonal” projection values against recommended thresholds.

See: `readme_FreeFermion_becnhmark.md` for the full protocol and interpretation.

---

## Benchmark JSON schema

A benchmark instance is stored as a single JSON document containing:

- benchmark metadata (name, id, number of qubits, sample size, target format/SDK, shots),
- a list of samples, each with its circuit list (`circuit_id`, `qasm`, `observable`, and metadata),
- optional experimental results (counts),
- optional evaluation results.

This enables reproducible generation, execution, and scoring while remaining platform-agnostic.

---

## Suggested workflow

1. Generate a benchmark instance and export circuits (OpenQASM).
2. Execute circuits using the provider’s preferred workflow.
3. Attach counts back to the benchmark instance.
4. Evaluate and store results (score + derived metrics).

---

## Acknowledgements

One of the benchmarks implemented in this package (the **Clifford Volume Benchmark**) is included in the set of **Key Performance Indicators (KPIs)** defined within the **EU Quantum Flagship** initiative for quantum computer benchmarking. The implementation provided here has also been collected as part of this initiative.

For details, see: https://arxiv.org/pdf/2512.19653

---
