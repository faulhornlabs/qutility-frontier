SCHEMA_VERSION = "0.1.1"

BENCHMARK_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Quantum Benchmark File",
    "type": "object",
    "required": [
        "schema_version",
        "benchmark_name",
        "benchmark_id",
        "number_of_qubits",
        "sample_size",
        "format",
        "target_sdk",
        "shots",
        "samples",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "benchmark_name": {"type": "string"},
        "benchmark_id": {"type": "string"},
        "number_of_qubits": {"type": "integer", "minimum": 1},
        "sample_size": {"type": "integer", "minimum": 1},
        "format": {"type": ["string", "null"]},
        "target_sdk": {"type": ["string", "null"]},
        "shots": {"type": "integer"},
        "global_metadata": {"type": "object"},
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["sample_id", "sample_metadata", "circuits"],
                "properties": {
                    "sample_id": {"type": "integer"},
                    "sample_metadata": {"type": "object"},
                    "circuits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["circuit_id", "observable", "qasm"],
                            "properties": {
                                "circuit_id": {"type": "string"},
                                "observable": {"type": ["string", "null"]},
                                "qasm": {"type": "string"},
                                "metadata": {"type": "object"},
                            },
                        },
                    },
                },
            },
        },
        "experimental_results": {
            "type": ["object", "null"],
            "description": "Experimental results obtained after executing benchmark circuits.",
            "required": ["experiment_id", "platform", "results"],
            "properties": {
                "experiment_id": {"type": "string"},
                "platform": {"type": "string"},
                "experiment_metadata": {"type": "object"},
                "results": {
                    "type": "object",
                    "description": "Dictionary mapping circuit_id → execution results.",
                    "propertyNames": {"type": "string"},
                    "additionalProperties": {
                        "type": "object",
                        "required": ["counts"],
                        "properties": {
                            "counts": {
                                "type": "object",
                                "description": (
                                    "Measurement results as a mapping bitstring -> count. "
                                    "Bitstrings follow Qiskit's little-endian convention: "
                                    "the RIGHTMOST bit corresponds to logical qubit 0, and "
                                    "the LEFTMOST bit corresponds to logical qubit (n-1). "
                                    "Example (3 qubits): bitstring '010' means qubit2=0, qubit1=1, qubit0=0."
                                ),
                                "propertyNames": {
                                    "type": "string",
                                    "pattern": "^[01]+$",
                                },
                                "additionalProperties": {
                                    "type": "integer",
                                    "minimum": 0,
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}
