"""
benchmarks — Top-level package for scalable volumetric benchmark tools.
"""

# Re-export core benchmark infrastructure
from .utils.quantumbenchmark import Benchmark
from .utils.benchmarkschema import SCHEMA_VERSION, BENCHMARK_JSON_SCHEMA

# Re-export circuit utilities
from .utils import (
    QuantumCircuit,
    QuantumGate,
    TwoQubitQuantumGate,
    QasmEmitter,
    QasmEmitterOptions,
    GivensRotation,
    so_decomposition,
    reconstruct,
    check_decomposition,
    plot_decomposition,
)

# Export available benchmark implementations
from .cliffordvolumebenchmark import CliffordVolumeBenchmark
from .freefermionvolumebenchmark import FreeFermionVolumeBenchmark

__all__ = [
    "Benchmark",
    # JSON Schema
    "SCHEMA_VERSION",
    "BENCHMARK_JSON_SCHEMA",
    # Core circuit utilities
    "QuantumCircuit",
    "QuantumGate",
    "TwoQubitQuantumGate",
    "QasmEmitter",
    "QasmEmitterOptions",
    "GivensRotation",
    "so_decomposition",
    "reconstruct",
    "check_decomposition",
    "plot_decomposition",
    # Implementations
    "CliffordVolumeBenchmark",
    "FreeFermionVolumeBenchmark",
]
