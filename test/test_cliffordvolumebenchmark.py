# test_benchmarks.py
from __future__ import annotations

import numpy as np
import pytest

# Adjust these imports to match your package layout
from frontier import CliffordVolumeBenchmark


def test_clifford_compute_number_of_measurements_rule():
    b_small = CliffordVolumeBenchmark(
        number_of_qubits=5, sample_size=1, auto_save=False
    )
    assert b_small.number_of_measurements == 4

    b_large = CliffordVolumeBenchmark(
        number_of_qubits=20, sample_size=1, auto_save=False
    )

    assert b_large.number_of_measurements == 4


def _make_simple_clifford_with_results(
    n_qubits: int = 1,
    shots: int = 10,
) -> CliffordVolumeBenchmark:
    """Helper: construct a CliffordVolumeBenchmark with simple samples/results.

    This bypasses any stim dependency by manually setting `samples` and
    `experimental_results`.
    """
    b = CliffordVolumeBenchmark(
        number_of_qubits=n_qubits,
        sample_size=1,
        shots=shots,
        auto_save=False,
    )

    # One stabilizer and one destabilizer circuit
    b.samples = [
        {
            "sample_id": 0,
            "sample_metadata": {},
            "circuits": [
                {
                    "circuit_id": "0_stab_0",
                    "observable": "+Z",
                    "qasm": "",
                    "metadata": {"kind": "stabilizer"},
                },
                {
                    "circuit_id": "0_destab_0",
                    "observable": "+Z",
                    "qasm": "",
                    "metadata": {"kind": "destabilizer"},
                },
            ],
        }
    ]

    b.experimental_results = {
        "experiment_id": "exp0",
        "platform": "sim",
        "experiment_metadata": {},
        "results": {
            # Stabilizer: always 0 → EV = +1
            "0_stab_0": {"counts": {"0": shots}},
            # Destabilizer: half/half → EV = 0
            "0_destab_0": {"counts": {"0": shots // 2, "1": shots - shots // 2}},
        },
    }

    return b


def test_clifford_compute_expectation_values_uses_results():
    b = _make_simple_clifford_with_results(n_qubits=1, shots=10)
    evs = b.compute_expectation_values()

    assert pytest.approx(evs["0_stab_0"]) == 1.0
    assert pytest.approx(evs["0_destab_0"]) == 0.0


def test_clifford_evaluate_benchmark_populates_evaluation_and_std_error():
    shots = 100
    b = _make_simple_clifford_with_results(n_qubits=1, shots=shots)

    out = b.evaluate_benchmark(auto_save=False)
    assert "stabilizer_expectation_values" in out
    assert "destabilizer_expectation_values" in out

    stab_evs = out["stabilizer_expectation_values"]
    dest_evs = out["destabilizer_expectation_values"]
    assert stab_evs == [pytest.approx(1.0)]
    assert dest_evs == [pytest.approx(0.0)]

    results = b.experimental_results["results"]
    assert "expectation_value" in results["0_stab_0"]
    assert "std_error" in results["0_stab_0"]

    # For EV = 1, std_error should be 0
    assert results["0_stab_0"]["std_error"] == pytest.approx(0.0)

    # For EV = 0, std_error ≈ 1/sqrt(shots)
    expected_se = np.sqrt(1.0 / shots)
    assert results["0_destab_0"]["std_error"] == pytest.approx(expected_se, rel=1e-3)


def test_clifford_get_all_expectation_value_structure():
    b = _make_simple_clifford_with_results(n_qubits=1, shots=50)

    # Deliberately do not call evaluate_benchmark first; method should
    # compute EVs on the fly and cache them.
    values = b.get_all_expectation_value()

    assert 0 in values
    sample_0 = values[0]
    assert "stabilizer" in sample_0
    assert "destabilizer" in sample_0

    stab_map = sample_0["stabilizer"]
    dest_map = sample_0["destabilizer"]

    # There should be one observable each
    assert len(stab_map) == 1
    assert len(dest_map) == 1

    # Expectation values should match what we encoded (1 and 0)
    ((ev_s, se_s),) = stab_map.values()
    ((ev_d, se_d),) = dest_map.values()

    assert ev_s == pytest.approx(1.0)
    assert ev_d == pytest.approx(0.0)

    # Standard errors must be non-negative floats
    assert isinstance(se_s, float) and se_s >= 0.0
    assert isinstance(se_d, float) and se_d >= 0.0


def test_clifford_evaluate_benchmark_raises_on_missing_results():
    b = CliffordVolumeBenchmark(
        number_of_qubits=1, sample_size=1, shots=10, auto_save=False
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
