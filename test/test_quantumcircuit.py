"""
Unit tests for the local QuantumCircuit class
"""

# tests/test_quantum_circuit.py
import pytest
from frontier import (
    QasmEmitter,
    QasmEmitterOptions,
    QuantumCircuit,
    QuantumGate,
    TwoQubitQuantumGate,
)

# --------------------------
# Basic circuit bookkeeping
# --------------------------


def test_gate_counts_and_repr_basic():
    """Counts single/two-qubit gates and checks the circuit __repr__ summary lines."""
    c = QuantumCircuit(number_of_qubits=4, number_of_classical_bits=2)
    c.add_x_gate(0)
    c.add_h_gate(1)
    c.add_rx_gate(2, theta=0.25)
    c.add_cx_gate(0, 1)  # 1 two-qubit
    c.add_swap_gate(2, 3)  # another two-qubit
    c.add_measurement([0, 1, 2, 3], [0, 1, 1, 1])

    assert c.single_qubit_gate_count() == 3
    assert c.two_qubit_gate_count() == 2
    rep = repr(c)
    assert "Number of qubits: 4" in rep
    assert "Single-qubit gates: 3" in rep
    assert "Two-qubit gates: 2" in rep
    assert "Measurements: 4" in rep


# ---------------------------------------
# Exact QASM emission (qiskit-style alias)
# ---------------------------------------


def _build_reference_circuit_for_qasm():
    """Helper: builds a 10-qubit reference circuit that exercises all supported gates."""
    pi = 3.14  # keep in sync with precision/expected strings
    c = QuantumCircuit(number_of_qubits=10, number_of_classical_bits=10)

    # Single-qubit Clifford + phase family
    c.add_x_gate(0)
    c.add_y_gate(1)
    c.add_z_gate(2)
    c.add_h_gate(3)
    c.add_s_gate(4)
    c.add_sdg_gate(5)
    c.add_t_gate(6)

    # General single-qubit unitary
    c.add_u_gate(9, theta=pi / 2, phi=pi / 3, lambda_parameter=pi / 4)

    # Rotations
    c.add_rx_gate(0, theta=0.5)
    c.add_ry_gate(1, theta=0.6)
    c.add_rz_gate(2, theta=0.7)

    # Two-qubit / controlled
    for tgt in range(1, c.number_of_qubits):
        c.add_cx_gate(0, tgt)
    c.add_cy_gate(1, 2)
    c.add_cz_gate(2, 3)
    c.add_swap_gate(5, 4)

    # Measure all qubits (1:1)
    c.add_measurement(list(range(10)), list(range(10)))
    return c


def test_qasm2_qiskit_generation_exact():
    """Emits exact QASM 2.0 for target_sdk='qiskit' including includes, u→u3, and measure syntax."""
    c = _build_reference_circuit_for_qasm()

    # qasm2 + qiskit maps "u"->"u3", includes "qelib1.inc", and uses "measure q->c"
    qasm2_ideal = (
        "OPENQASM 2.0;\n\n"
        'include "qelib1.inc";\n\n'
        "qreg q[10];\ncreg c[10];\n\n"
        "x q[0];\n"
        "y q[1];\n"
        "z q[2];\n"
        "h q[3];\n"
        "s q[4];\n"
        "sdg q[5];\n"
        "t q[6];\n"
        "u3(1.570000,1.046667,0.785000) q[9];\n"
        "rx(0.500000) q[0];\n"
        "ry(0.600000) q[1];\n"
        "rz(0.700000) q[2];\n"
        "cx q[0], q[1];\n"
        "cx q[0], q[2];\n"
        "cx q[0], q[3];\n"
        "cx q[0], q[4];\n"
        "cx q[0], q[5];\n"
        "cx q[0], q[6];\n"
        "cx q[0], q[7];\n"
        "cx q[0], q[8];\n"
        "cx q[0], q[9];\n"
        "cy q[1], q[2];\n"
        "cz q[2], q[3];\n"
        "swap q[5], q[4];\n\n"
        "measure q[0] -> c[0];\n"
        "measure q[1] -> c[1];\n"
        "measure q[2] -> c[2];\n"
        "measure q[3] -> c[3];\n"
        "measure q[4] -> c[4];\n"
        "measure q[5] -> c[5];\n"
        "measure q[6] -> c[6];\n"
        "measure q[7] -> c[7];\n"
        "measure q[8] -> c[8];\n"
        "measure q[9] -> c[9];"
    )

    got = c.to_qasm(format="qasm2", target_sdk="qiskit")
    assert got == qasm2_ideal


def test_qasm3_qiskit_generation_exact():
    """Emits exact QASM 3.0 for target_sdk='qiskit' including stdgates.inc and c[i]=measure q[i] form."""
    c = _build_reference_circuit_for_qasm()

    # qasm3 + qiskit maps "u"->"u3", includes "stdgates.inc",
    # and uses "c[i] = measure q[i];"
    qasm3_ideal = (
        "OPENQASM 3.0;\n\n"
        'include "stdgates.inc";\n\n'
        "qubit[10] q;\nbit[10] c;\n\n"
        "x q[0];\n"
        "y q[1];\n"
        "z q[2];\n"
        "h q[3];\n"
        "s q[4];\n"
        "sdg q[5];\n"
        "t q[6];\n"
        "u3(1.570000,1.046667,0.785000) q[9];\n"
        "rx(0.500000) q[0];\n"
        "ry(0.600000) q[1];\n"
        "rz(0.700000) q[2];\n"
        "cx q[0], q[1];\n"
        "cx q[0], q[2];\n"
        "cx q[0], q[3];\n"
        "cx q[0], q[4];\n"
        "cx q[0], q[5];\n"
        "cx q[0], q[6];\n"
        "cx q[0], q[7];\n"
        "cx q[0], q[8];\n"
        "cx q[0], q[9];\n"
        "cy q[1], q[2];\n"
        "cz q[2], q[3];\n"
        "swap q[5], q[4];\n\n"
        "c[0] = measure q[0];\n"
        "c[1] = measure q[1];\n"
        "c[2] = measure q[2];\n"
        "c[3] = measure q[3];\n"
        "c[4] = measure q[4];\n"
        "c[5] = measure q[5];\n"
        "c[6] = measure q[6];\n"
        "c[7] = measure q[7];\n"
        "c[8] = measure q[8];\n"
        "c[9] = measure q[9];"
    )

    got = c.to_qasm(format="qasm3", target_sdk="qiskit")
    assert got == qasm3_ideal


def test_qasm3_braket_generation_exact():
    """Emits exact QASM 3.0 for target_sdk='braket' including u→U, cx→cnot, and no includes."""
    c = _build_reference_circuit_for_qasm()

    # qasm3 + braket maps "u"->"U", "cx"->"cnot", and emits NO includes
    qasm3_braket_ideal = (
        "OPENQASM 3.0;\n\n"
        "\n"  # (no includes) the emitter keeps a spacer line
        "qubit[10] q;\nbit[10] c;\n\n"
        "x q[0];\n"
        "y q[1];\n"
        "z q[2];\n"
        "h q[3];\n"
        "s q[4];\n"
        "si q[5];\n"
        "t q[6];\n"
        "U(1.570000,1.046667,0.785000) q[9];\n"
        "rx(0.500000) q[0];\n"
        "ry(0.600000) q[1];\n"
        "rz(0.700000) q[2];\n"
        "cnot q[0], q[1];\n"
        "cnot q[0], q[2];\n"
        "cnot q[0], q[3];\n"
        "cnot q[0], q[4];\n"
        "cnot q[0], q[5];\n"
        "cnot q[0], q[6];\n"
        "cnot q[0], q[7];\n"
        "cnot q[0], q[8];\n"
        "cnot q[0], q[9];\n"
        "cy q[1], q[2];\n"
        "cz q[2], q[3];\n"
        "swap q[5], q[4];\n\n"
        "c[0] = measure q[0];\n"
        "c[1] = measure q[1];\n"
        "c[2] = measure q[2];\n"
        "c[3] = measure q[3];\n"
        "c[4] = measure q[4];\n"
        "c[5] = measure q[5];\n"
        "c[6] = measure q[6];\n"
        "c[7] = measure q[7];\n"
        "c[8] = measure q[8];\n"
        "c[9] = measure q[9];"
    )

    got = c.to_qasm(format="qasm3", target_sdk="braket")
    assert got == qasm3_braket_ideal


def test_qasm2_tket_generation_exact():
    """Emits exact QASM 2.0 for target_sdk='tket' including qelib1.inc and u→u3 mapping."""
    c = _build_reference_circuit_for_qasm()

    # qasm2 + tket uses "u3" and includes "qelib1.inc"
    qasm2_tket_ideal = (
        "OPENQASM 2.0;\n\n"
        'include "qelib1.inc";\n\n'
        "qreg q[10];\ncreg c[10];\n\n"
        "x q[0];\n"
        "y q[1];\n"
        "z q[2];\n"
        "h q[3];\n"
        "s q[4];\n"
        "sdg q[5];\n"
        "t q[6];\n"
        "u3(1.570000,1.046667,0.785000) q[9];\n"
        "rx(0.500000) q[0];\n"
        "ry(0.600000) q[1];\n"
        "rz(0.700000) q[2];\n"
        "cx q[0], q[1];\n"
        "cx q[0], q[2];\n"
        "cx q[0], q[3];\n"
        "cx q[0], q[4];\n"
        "cx q[0], q[5];\n"
        "cx q[0], q[6];\n"
        "cx q[0], q[7];\n"
        "cx q[0], q[8];\n"
        "cx q[0], q[9];\n"
        "cy q[1], q[2];\n"
        "cz q[2], q[3];\n"
        "swap q[5], q[4];\n\n"
        "measure q[0] -> c[0];\n"
        "measure q[1] -> c[1];\n"
        "measure q[2] -> c[2];\n"
        "measure q[3] -> c[3];\n"
        "measure q[4] -> c[4];\n"
        "measure q[5] -> c[5];\n"
        "measure q[6] -> c[6];\n"
        "measure q[7] -> c[7];\n"
        "measure q[8] -> c[8];\n"
        "measure q[9] -> c[9];"
    )

    got = c.to_qasm(format="qasm2", target_sdk="tket")
    assert got == qasm2_tket_ideal


# ----------------------------------------
# Options, validation, and edge conditions
# ----------------------------------------


def test_measurement_length_mismatch_raises():
    """Raises ValueError when qubit and classical index lists differ in length."""
    c = QuantumCircuit(number_of_qubits=3, number_of_classical_bits=2)
    with pytest.raises(ValueError):
        c.add_measurement([0, 1, 2], [0, 1])  # mismatched lengths


def test_qasm_emitter_option_validation_errors():
    """Validates QasmEmitterOptions rejects unknown format/SDK and negative precision."""
    with pytest.raises(ValueError):
        QasmEmitterOptions(format="qasm9")  # unsupported format

    with pytest.raises(ValueError):
        QasmEmitterOptions(format="qasm2", float_precision=-1)  # negative precision

    with pytest.raises(ValueError):
        QasmEmitterOptions(format="qasm2", target_sdk="weirdsdk")  # invalid sdk


def test_custom_precision_and_includes_override():
    """Allows custom includes and float precision when target_sdk='custom' does not auto-fill."""
    # To override includes, use target_sdk="custom" so auto-fill does NOT overwrite.
    opts = QasmEmitterOptions(
        format="qasm2",
        target_sdk="custom",
        includes=["mygates.inc"],
        float_precision=3,
    )
    c = QuantumCircuit(number_of_qubits=1, number_of_classical_bits=1)
    c.add_rx_gate(0, 0.5)
    c.add_measurement(0, 0)

    got = QasmEmitter(opts).emit(c)
    assert 'include "mygates.inc";' in got
    # 3 decimals for parameters:
    assert "rx(0.500) q[0];" in got
    # qasm2 measurement form:
    assert "measure q[0] -> c[0];" in got


def test_quantum_gate_validation_errors():
    """Checks gate constructors enforce integer indices, numeric parameters, and nonempty controls."""
    # Non-integer target index
    with pytest.raises(ValueError):
        QuantumGate(name="x", target_qubits="a")

    # Parameters must be numeric
    with pytest.raises(TypeError):
        QuantumGate(name="rx", target_qubits=0, parameters=["oops"])

    # TwoQubitQuantumGate requires at least one control qubit
    with pytest.raises(ValueError):
        TwoQubitQuantumGate(name="cx", target_qubits=1, control_qubits=[])


def test_swap_emission_order():
    """Ensures swap emission preserves argument order as 'swap q[i], q[j];'."""
    c = QuantumCircuit(number_of_qubits=6, number_of_classical_bits=0)
    c.add_swap_gate(5, 4)
    qasm = c.to_qasm(format="qasm2", target_sdk="qiskit")
    # Must emit "swap q[5], q[4];" in this exact order (control, target → q1, q2)
    assert "swap q[5], q[4];" in qasm


def test_draw_circuit_diagram_does_not_crash(capsys):
    """Smoke-tests ASCII diagram generation to ensure draw_circuit_diagram() runs without error."""
    c = QuantumCircuit(number_of_qubits=3, number_of_classical_bits=0)
    c.add_h_gate(0)
    c.add_cx_gate(0, 2)
    c.add_measurement([0, 1, 2], [0, 0, 0])  # measurement lines also drawn
    c.draw_circuit_diagram(max_length=5)
    out = capsys.readouterr().out
    assert "q0" in out and "q1" in out and "q2" in out
