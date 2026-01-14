# test_benchmarks.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import jsonschema

from ScalableVolumetricBenchmark import Benchmark, BENCHMARK_JSON_SCHEMA, SCHEMA_VERSION


class DummyBenchmark(Benchmark):
    """Minimal concrete implementation for testing the abstract Benchmark API."""

    BENCHMARK_NAME = "dummy"

    def _create_single_sample(self, sample_id: int) -> Dict[str, Any]:
        # Use numpy arrays in metadata to exercise _jsonify_value/_jsonify_meta
        return {
            "sample_id": sample_id,
            "sample_metadata": {
                "sample_index": sample_id,
                "vector": np.array([1, 2, 3]),
            },
            "circuits": [
                {
                    "circuit_id": f"{sample_id}_c0",
                    "observable": "+Z",
                    "qasm": "OPENQASM 2.0; // dummy",
                    "metadata": {
                        "kind": "test",
                        "array": np.array([[1, 2], [3, 4]]),
                    },
                }
            ],
        }

    def evaluate_benchmark(self) -> Any:  # pragma: no cover - trivial for tests
        return {"status": "ok"}


def test_json_schema_accepts_minimal_valid_payload():
    """Check that BENCHMARK_JSON_SCHEMA accepts a simple valid object."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_name": "dummy",
        "benchmark_id": "dummy_nq1_s1_id",
        "number_of_qubits": 1,
        "sample_size": 1,
        "format": "qasm2",
        "target_sdk": "default",
        "shots": 100,
        "global_metadata": {},
        "samples": [
            {
                "sample_id": 0,
                "sample_metadata": {},
                "circuits": [
                    {
                        "circuit_id": "0_c0",
                        "observable": "+Z",
                        "qasm": "OPENQASM 2.0;",
                        "metadata": {},
                    }
                ],
            }
        ],
        "experimental_results": {
            "experiment_id": "exp0",
            "platform": "sim",
            "experiment_metadata": {},
            "results": {
                "0_c0": {
                    "counts": {"0": 10, "1": 0},
                }
            },
        },
    }

    # Should not raise
    jsonschema.validate(instance=payload, schema=BENCHMARK_JSON_SCHEMA)


def test_dummy_benchmark_create_and_to_json_matches_schema():
    """create_benchmark + to_json_dict should produce schema-valid payload."""
    b = DummyBenchmark(
        number_of_qubits=1,
        sample_size=2,
        shots=100,
        workdir=Path(".tmp_should_not_be_used"),
        auto_save=False,
    )

    samples = b.create_benchmark(auto_save=False)
    assert len(samples) == 2
    assert b.samples is not None
    assert b.samples[0]["sample_id"] == 0

    payload = b.to_json_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["benchmark_name"] == "dummy"
    assert payload["number_of_qubits"] == 1
    assert payload["sample_size"] == 2
    assert payload["shots"] == 100

    # Metadata with numpy arrays should be converted to plain lists
    sm_meta = payload["samples"][0]["sample_metadata"]
    assert isinstance(sm_meta["vector"], list)

    circ_meta = payload["samples"][0]["circuits"][0]["metadata"]
    assert isinstance(circ_meta["array"], list)

    # Validate against JSON schema (integration test)
    jsonschema.validate(instance=payload, schema=BENCHMARK_JSON_SCHEMA)


def test_benchmark_repr_contains_key_fields():
    b = DummyBenchmark(number_of_qubits=3, sample_size=5, auto_save=False)
    r = repr(b)
    assert "DummyBenchmark" in r
    assert f"number_of_qubits={b.number_of_qubits}" in r
    assert f"sample_size={b.sample_size}" in r
    assert f"benchmark_id={b.benchmark_id!r}" in r


def test_save_json_default_and_filename_resolution(tmp_path: Path):
    """Check that save_json handles default path and filename variants."""
    b = DummyBenchmark(
        number_of_qubits=1,
        sample_size=1,
        shots=10,
        workdir=tmp_path,
        auto_save=False,
    )
    b.create_benchmark(auto_save=False)

    # Default path: workdir / {benchmark_id}.json
    path_default = b.save_json()
    assert path_default.parent == tmp_path
    assert path_default.name == f"{b.benchmark_id}.json"
    assert path_default.is_file()

    # Existing directory → file inside that directory
    out_dir = tmp_path / "results_dir"
    out_dir.mkdir()
    path_dir = b.save_json(filepath=out_dir)
    assert path_dir.parent == out_dir
    assert path_dir.name == f"{b.benchmark_id}.json"
    assert path_dir.is_file()
    assert b.workdir == out_dir

    # Non-existing path without suffix → treated as directory
    new_dir = tmp_path / "new_results"
    path_new_dir = b.save_json(filepath=new_dir)
    assert path_new_dir.parent == new_dir
    assert path_new_dir.name == f"{b.benchmark_id}.json"
    assert path_new_dir.is_file()
    assert b.workdir == new_dir

    # Bare filename → saved under current workdir
    path_filename = b.save_json(filepath="custom.json")
    assert path_filename.parent == new_dir
    assert path_filename.name == "custom.json"
    assert path_filename.is_file()


def test_from_json_dict_and_load_json_roundtrip(tmp_path: Path):
    b = DummyBenchmark(
        number_of_qubits=2,
        sample_size=1,
        shots=20,
        workdir=tmp_path,
        auto_save=False,
    )
    b.create_benchmark(auto_save=False)

    path = b.save_json()
    assert path.is_file()

    # Load via classmethod
    loaded = DummyBenchmark.load_json(path)
    assert isinstance(loaded, DummyBenchmark)
    assert loaded.number_of_qubits == 2
    assert loaded.sample_size == 1
    assert loaded.benchmark_id == b.benchmark_id
    assert loaded.workdir == path.parent
    assert loaded.samples is not None
    assert len(loaded.samples) == 1


def test_add_experimental_results_from_dict_and_list(tmp_path: Path):
    # Dict keyed by circuit_id
    b = DummyBenchmark(
        number_of_qubits=1,
        sample_size=1,
        shots=10,
        workdir=tmp_path,
        auto_save=False,
    )
    b.create_benchmark(auto_save=False)
    cid = b.samples[0]["circuits"][0]["circuit_id"]  # type: ignore[index]

    b.add_experimental_results(
        {cid: {"0": 7, "1": 3}},
        experiment_id="exp-dict",
        platform="sim",
        auto_save=False,
    )

    er = b.experimental_results
    assert er is not None
    assert er["experiment_id"] == "exp-dict"
    assert er["platform"] == "sim"
    assert er["results"][cid]["counts"] == {"0": 7, "1": 3}

    # List aligned with circuit order
    b2 = DummyBenchmark(
        number_of_qubits=1,
        sample_size=2,
        shots=10,
        workdir=tmp_path,
        auto_save=False,
    )
    b2.create_benchmark(auto_save=False)
    ids = b2.get_all_circuit_ids()

    counts_list = [{"0": 5, "1": 5}, {"0": 10, "1": 0}]
    b2.add_experimental_results(counts_list, experiment_id="exp-list", auto_save=False)
    er2 = b2.experimental_results
    assert er2 is not None
    assert set(er2["results"].keys()) == set(ids)


def test_add_experimental_results_validation_errors(tmp_path: Path):
    b = DummyBenchmark(
        number_of_qubits=1,
        sample_size=1,
        shots=10,
        workdir=tmp_path,
        auto_save=False,
    )
    b.create_benchmark(auto_save=False)
    cid = b.samples[0]["circuits"][0]["circuit_id"]  # type: ignore[index]

    # Non-dict counts should raise TypeError
    with pytest.raises(TypeError):
        b.add_experimental_results({cid: [1, 2, 3]}, auto_save=False)

    # Negative counts should raise ValueError
    with pytest.raises(ValueError):
        b.add_experimental_results({cid: {"0": -1}}, auto_save=False)


def test_get_all_circuit_ids_and_circuits_and_error_if_not_generated():
    b = DummyBenchmark(number_of_qubits=1, sample_size=1, auto_save=False)

    # Without samples: should raise
    with pytest.raises(ValueError):
        _ = b.get_all_circuit_ids()
    with pytest.raises(ValueError):
        _ = b.get_all_circuits()

    # After generation
    b.create_benchmark(auto_save=False)
    ids = b.get_all_circuit_ids()
    qasms = b.get_all_circuits()

    assert len(ids) == 1
    assert len(qasms) == 1
    assert ids[0].endswith("_c0")
    assert "OPENQASM" in qasms[0]


def test_expected_value_basic_and_errors():
    b = DummyBenchmark(number_of_qubits=1, sample_size=1, auto_save=False)

    # Empty counts
    with pytest.raises(ValueError):
        _ = b.expected_value({}, "Z")

    # Single qubit Z
    counts_all_zero = {"0": 10}
    counts_balanced = {"0": 5, "1": 5}
    counts_all_one = {"1": 10}

    assert b.expected_value(counts_all_zero, "Z") == pytest.approx(1.0)
    assert b.expected_value(counts_balanced, "Z") == pytest.approx(0.0)
    assert b.expected_value(counts_all_one, "Z") == pytest.approx(-1.0)

    # Mismatched Pauli length
    with pytest.raises(ValueError):
        _ = b.expected_value({"00": 1}, "Z")

    # Zero total shots
    with pytest.raises(ValueError):
        _ = b.expected_value({"0": 0, "1": 0}, "Z")
