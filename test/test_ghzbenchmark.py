from __future__ import annotations

import numpy as np
import pytest

from frontier import (
    GHZBenchmark,
    certify_fidelity_gt_half,
    evaluate_fidelity,
    evaluate_shadow_overlap,
)


def _ideal_counts(bitstring: str, shots: int) -> dict[str, int]:
    return {bitstring: shots}


def test_ghz_witness_generate_and_evaluate() -> None:
    shots = 128
    benchmark = GHZBenchmark(
        number_of_qubits=3,
        sample_size=1,
        shots=shots,
        measurement_scheme="witness",
        auto_save=False,
    )
    benchmark.create_benchmark()

    assert benchmark.samples is not None
    sample = benchmark.samples[0]
    assert len(sample["circuits"]) == 2  # X and Z witness circuits

    zero_string = "0" * benchmark.number_of_qubits
    counts_map: dict[str, dict[str, int]] = {}
    for circuit in sample["circuits"]:
        basis = circuit["metadata"]["basis"]
        counts_map[circuit["circuit_id"]] = _ideal_counts(zero_string, shots)
        assert basis in {"x", "z"}

    benchmark.add_experimental_results(
        counts_map,
        experiment_id="unit-test",
        platform="simulator",
    )

    evaluation = benchmark.evaluate_benchmark(auto_save=False)
    per_sample = evaluation["per_sample"][0]
    assert per_sample["fidelity_lower_bound"] == pytest.approx(1.0)
    assert per_sample["std_dev"] == pytest.approx(0.0)
    assert per_sample["certified_gt_half"] is True

    results = benchmark.experimental_results["results"]
    for circuit in sample["circuits"]:
        cid = circuit["circuit_id"]
        assert results[cid]["fidelity_lower_bound"] == pytest.approx(1.0)
        assert results[cid]["std_dev"] == pytest.approx(0.0)


def test_evaluate_fidelity_helpers() -> None:
    z_counts = {"00": 60, "11": 40}
    x_counts = {"00": 100}

    fidelity, std = evaluate_fidelity(z_counts, x_counts)
    assert fidelity == pytest.approx(1.0)
    assert std == pytest.approx(0.0)
    assert certify_fidelity_gt_half(fidelity, std, confidence=0.95)


def test_shadow_evaluation_matches_helper() -> None:
    np.random.seed(0)
    benchmark = GHZBenchmark(
        number_of_qubits=4,
        sample_size=1,
        shots=32,
        measurement_scheme="shadow",
        measurement_rounds=3,
        auto_save=False,
    )
    benchmark.create_benchmark()

    assert benchmark.samples is not None
    sample = benchmark.samples[0]

    measurement_outcomes: list[str] = []
    basis_sequence: list[dict[int, str]] = []
    results_payload: dict[str, dict[str, list[str]]] = {}

    for circuit in sample["circuits"]:
        cid = circuit["circuit_id"]
        basis_map = circuit["metadata"]["basis_map"]
        # Provide a single all-zero outcome per circuit for determinism.
        shot = "0" * benchmark.number_of_qubits
        measurement_outcomes.append(shot)
        basis_sequence.append({int(k): v for k, v in basis_map.items()})
        results_payload[cid] = {"shots": [shot]}

    benchmark.experimental_results = {
        "experiment_id": "shadow-test",
        "platform": "simulator",
        "results": results_payload,
    }

    evaluation = benchmark.evaluate_benchmark(auto_save=False)
    per_sample = evaluation["per_sample"][0]
    mean, sem = evaluate_shadow_overlap(measurement_outcomes, basis_sequence)

    assert per_sample["shadow_overlap_mean"] == pytest.approx(mean)
    assert per_sample["shadow_overlap_sem"] == pytest.approx(sem)
