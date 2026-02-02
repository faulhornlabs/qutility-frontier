# test_freefermion_benchmark.py
from __future__ import annotations

import numpy as np
import pytest
from frontier import FreeFermionVolumeBenchmark


def test_freefermion_compute_number_of_measurements_rule():
    # For n <= 10: 2 * n
    b_small = FreeFermionVolumeBenchmark(
        number_of_qubits=5, sample_size=1, auto_save=False
    )
    assert b_small.number_of_measurements == 2 * 5

    # For n > 10: 20 + floor(n / 5)
    b_large = FreeFermionVolumeBenchmark(
        number_of_qubits=20, sample_size=1, auto_save=False
    )
    expected = int(20 + np.floor(20 / 5))
    assert b_large.number_of_measurements == expected


def _make_simple_freefermion_with_results(
    n_qubits: int = 1,
    shots: int = 10,
) -> FreeFermionVolumeBenchmark:
    """Helper: construct a FreeFermionVolumeBenchmark with simple samples/results.

    This bypasses any stim dependency by manually setting `samples` and
    `experimental_results`.

    We choose O = I so that:
      * indices = [0, 1] for state_index = 0
      * row = [1, 0]
      * parallel_value = evs[0]
      * orthogonal_value = evs[1]

    Then we pick evs = [1, 0], so we expect:
      parallel_value   = 1
      orthogonal_value = 0
    """
    b = FreeFermionVolumeBenchmark(
        number_of_qubits=n_qubits,
        sample_size=1,
        shots=shots,
        auto_save=False,
    )

    # Simple orthogonal matrix: identity on 2n modes
    orthogonal_matrix = np.eye(2 * n_qubits, dtype=float)
    state_index = 0  # so _compute_measurement_indices returns [0, 1] for n_qubits=1

    # One "majorana" circuit per measured mode (2 for n_qubits=1)
    b.samples = [
        {
            "sample_id": 0,
            "sample_metadata": {
                "orthogonal_matrix": orthogonal_matrix,
                "initial_state_index": state_index,
            },
            "circuits": [
                {
                    "circuit_id": "0_maj_0",
                    "observable": "+Z",  # EV = +1 if all counts are '0'
                    "qasm": "",
                    "metadata": {"kind": "majorana"},
                },
                {
                    "circuit_id": "0_maj_1",
                    "observable": "+Z",  # EV = 0 if half '0', half '1'
                    "qasm": "",
                    "metadata": {"kind": "majorana"},
                },
            ],
        }
    ]

    # Experimental results:
    #  - First circuit: always '0' → EV = +1
    #  - Second circuit: half/half → EV = 0
    b.experimental_results = {
        "experiment_id": "exp_ff_0",
        "platform": "sim",
        "experiment_metadata": {},
        "results": {
            "0_maj_0": {"counts": {"0": shots}},
            "0_maj_1": {"counts": {"0": shots // 2, "1": shots - shots // 2}},
        },
    }

    return b


def test_freefermion_evaluate_benchmark_parallel_orthogonal_values():
    shots = 100
    b = _make_simple_freefermion_with_results(n_qubits=1, shots=shots)

    out = b.evaluate_benchmark(auto_save=False)
    assert "parallel_values" in out
    assert "orthogonal_values" in out

    parallel_vals = out["parallel_values"]
    orthogonal_vals = out["orthogonal_values"]

    # One sample only
    assert len(parallel_vals) == 1
    assert len(orthogonal_vals) == 1

    # With the simple setup (O = I, evs = [1, 0]):
    #   parallel_value   = 1
    #   orthogonal_value = 0
    assert parallel_vals[0] == pytest.approx(1.0)
    assert orthogonal_vals[0] == pytest.approx(0.0)

    # Check that per-circuit EVs and std_errors are stored
    results = b.experimental_results["results"]
    assert "expectation_value" in results["0_maj_0"]
    assert "std_error" in results["0_maj_0"]

    # For EV = 1, std_error should be 0
    assert results["0_maj_0"]["std_error"] == pytest.approx(0.0)

    # For EV = 0, std_error ≈ 1/sqrt(shots)
    expected_se = np.sqrt(1.0 / shots)
    assert results["0_maj_1"]["std_error"] == pytest.approx(expected_se, rel=1e-3)


def test_freefermion_evaluate_benchmark_raises_on_missing_results():
    b = FreeFermionVolumeBenchmark(
        number_of_qubits=1,
        sample_size=1,
        shots=10,
        auto_save=False,
    )

    # Missing experimental_results
    with pytest.raises(ValueError):
        b.evaluate_benchmark()

    # Attach malformed experimental_results (missing "results" key)
    b.samples = [
        {
            "sample_id": 0,
            "sample_metadata": {},
            "circuits": [],
        }
    ]
    b.experimental_results = {"experiment_id": "x", "platform": "sim"}

    with pytest.raises(ValueError):
        b.evaluate_benchmark()
