Table of Contents
=================

* [Contributing](#contributing)
  * [Issue Reporting](#issue-reporting)
  * [Questions](#questions)
  * [Feature Proposals](#feature-proposals)
  * [Code Contributions](#code-contributions)
    * [Code quality tools](#code-quality-tools)
    * [Implementing a New Benchmark](#implementing-a-new-benchmark)
      * [Required documentation for every benchmark](#required-documentation-for-every-benchmark)
      * [Project layout for a new benchmark](#project-layout-for-a-new-benchmark)
      * [Implement a Benchmark subclass](#implement-a-benchmark-subclass)
        * [1) Define the benchmark class](#1-define-the-benchmark-class)
        * [2) Implement `_create_single_sample` method](#2-implement-_create_single_sample-method)
        * [3) Implement `evaluate_benchmark` method](#3-implement-evaluate_benchmark-method)
        * [4) Optional overrides](#4-optional-overrides)
        * [5) Ensure schema compliance](#5-ensure-schema-compliance)
    * [Tutorials and Demonstrations](#tutorials-and-demonstrations)

---

# Contributing

All kinds of contributions are welcome and appreciated.

Frontier aims to be an SDK- and hardware-independent framework and toolkit for benchmarking quantum devices. Feedback and contributions from the community are highly appreciated and essential for the continued development of this project.

### Issue Reporting

If you discover a bug, unexpected behavior, or any inconsistency, please open an issue.

When reporting a problem, make the description as clear and detailed as possible so others can understand and reproduce it. Include any context you think may be relevant, such as what you were trying to do and what result you expected.

### Questions 

Questions about usage, design decisions, or benchmarks are very welcome. If something is unclear or confusing, please open an issue and ask. Discussions about evaluation logic, schema fields, parameters, or metric meaning are especially helpful and often lead to documentation and usability improvements.

## Feature Proposals

Ideas for new features, benchmark types, or structural improvements are encouraged. If you would like to propose something new, please open an issue labeled `feature`.

## Code Contributions

We use **git** to manage development and history.

To keep changes reviewable and stable:

- Any new benchmark, feature, or refactor should be developed on a **separate branch**
- Once the work is complete, open a **Pull Request** into `main`
- New branches are merged by the **maintainers/owners** after review

Branch names are not strictly enforced, but descriptive names are preferred, for example:

- `clifford-volume-metric`
- `freefermion-eval-fix`
- `my-new-benchmark`

### Code quality tools 

Frontier currently uses:

- **Ruff** for linting and formatting to increase code quality and style consistency
- **Pytest** for unit and integration tests

You can run the same checks locally:

    ruff check .
    ruff format .
    pytest -v .

Frontier includes Continuous Integration (CI) via GitHub Actions (`.github/workflows/ci.yml`). For every pull request and for changes pushed to the main branch, CI runs automatically. The CI pipeline checks code style and runs the tests. All checks must pass before a pull request can be merged.

**Tests are required for benchmark additions and changes.** Every new benchmark, as well as any meaningful modification to an existing one, must include tests. Tests should cover circuit generation, evaluation logic, and any non-trivial helper utilities. Good test coverage makes future updates safer and improves the overall stability of the code base.

### Implementing a New Benchmark

This section is a practical guide for implementing a benchmark that fits the introduced architecture and framework.

Before starting, we strongly advise reading these files:

- `frontier/utils/quantumbenchmark.py`
- `frontier/utils/quantumcircuit.py`
- `frontier/notebooks/QuantumCircuit_tutorial.ipynb`
- `frontier/utils/benchmarkschema.py`

### Required documentation for every benchmark

Each new benchmark must include a **README** inside its package folder that explains:

- what the benchmark measures
- the core idea and theory the benchmark is based on
- the benchmark workflow
- the evaluation method
- the reported metrics and how to interpret them

Recommended location:

    frontier/<benchmark_name>/README.md

### Project layout for a new benchmark

Create a new submodule for your benchmark:

    frontier/<benchmark_name>/
        __init__.py
        benchmark.py
        README.md

### Implement a Benchmark subclass

Frontier benchmarks are implemented by subclassing `Benchmark` from `frontier/utils/quantumbenchmark.py`.

#### 1) Define the benchmark class

Subclass `Benchmark` and set a unique name:

```python
from frontier.utils.quantumbenchmark import Benchmark

class MyBenchmark(Benchmark):
    BENCHMARK_NAME = "MyBenchmark"
```

#### 2) Implement `_create_single_sample` method

In Frontier, `_create_single_sample(sample_id)` must return a **single sample dictionary** that follows the benchmark JSON schema.

At minimum, the returned dictionary should have this structure (see the docstring in `frontier/utils/quantumbenchmark.py` or the corresponding documentation for details):

```python
{
    "sample_id": int,
    "sample_metadata": {...},
    "circuits": [
        {
            "circuit_id": str,
            "observable": str | None,
            "qasm": str,
            "metadata": {...},
        },
        ...
    ],
}
```

#### 3) Implement `evaluate_benchmark` method

This method calculates the benchmark metrics from attached results.

It must:

- use stored results only
- be deterministic
- return a JSON-serializable metrics dictionary

#### 4) Optional overrides

If needed, you may override additional already implemented methods from the base class (such as measurement counting helpers).

#### 5) Ensure schema compliance

The structure of the created benchmark should be in accordance with the defined JSON schema used to store benchmark data. The utility methods that handle serialization (saving and loading), adding experimental results, and returning internally stored data such as benchmark circuits are already implemented. If required, most internal methods can be overridden and specialized for a new benchmark.

### Tutorials and Demonstrations

In addition to core code contributions, we strongly encourage tutorial and demonstration materials that show how Frontier benchmarks are used in practice.

This includes examples and walkthroughs for specific SDKs or hardware providers, demonstrating how to generate benchmarks, export circuits, run them on a backend, and evaluate results.

---
Thank you for contributing to Qutility Frontier.
