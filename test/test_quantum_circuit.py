from benchmarks import QuantumCircuit


def test_creat_circuit():
    pi = 3.14
    # --- build a 10-qubit circuit using every implemented operation once ---
    c = QuantumCircuit(number_of_qubits=10, number_of_classical_bits=10)

    # Single-qubit Clifford + phase family
    c.add_x_gate(0)
    c.add_y_gate(1)
    c.add_z_gate(2)
    c.add_h_gate(3)
    c.add_s_gate(4)
    c.add_t_gate(6)

    c.add_u_gate(9, theta=pi / 2, phi=pi / 3, lambda_parameter=pi / 4)

    # Rotations
    c.add_rx_gate(0, theta=0.5)
    c.add_ry_gate(1, theta=0.6)
    c.add_rz_gate(2, theta=0.7)

    # Two-qubit / controlled
    c.add_cx_gate(
        [0 for _ in range(c.number_of_qubits - 1)],
        [1 + i for i in range(c.number_of_qubits - 1)],
    )
    c.add_cy_gate(1, 2)
    c.add_cz_gate(2, 3)
    c.add_swap_gate(4, 5)  # your current add_swap_gate signature

    # Measure all qubits into all classical bits (1:1 mapping)
    c.add_measurement(list(range(10)), list(range(10)))

    assert c.single_qubit_gate_count() == 10 and c.two_qubit_gate_count() == 12


def test_qiskit_qasm_generation():
    pi = 3.14
    # --- build a 10-qubit circuit using every implemented operation once ---
    c = QuantumCircuit(number_of_qubits=10, number_of_classical_bits=10)

    # Single-qubit Clifford + phase family
    c.add_x_gate(0)
    c.add_y_gate(1)
    c.add_z_gate(2)
    c.add_h_gate(3)
    c.add_s_gate(4)
    c.add_t_gate(6)

    c.add_u_gate(9, theta=pi / 2, phi=pi / 3, lambda_parameter=pi / 4)

    # Rotations
    c.add_rx_gate(0, theta=0.5)
    c.add_ry_gate(1, theta=0.6)
    c.add_rz_gate(2, theta=0.7)

    # Two-qubit / controlled
    c.add_cx_gate(
        [0 for _ in range(c.number_of_qubits - 1)],
        [1 + i for i in range(c.number_of_qubits - 1)],
    )
    c.add_cy_gate(1, 2)
    c.add_cz_gate(2, 3)
    c.add_swap_gate(4, 5)  # your current add_swap_gate signature

    # Measure all qubits into all classical bits (1:1 mapping)
    c.add_measurement(list(range(10)), list(range(10)))

    qasm2_ideal = 'OPENQASM 2.0;\n\ninclude "qelib1.inc";\n\nqreg q[10];\ncreg c[10];\n\nx q[0];\ny q[1];\nz q[2];\nh q[3];\ns q[4];\nt q[6];\nu3(1.570000,1.046667,0.785000) q[9];\nrx(0.500000) q[0];\nry(0.600000) q[1];\nrz(0.700000) q[2];\ncx q[0], q[1];\ncx q[0], q[2];\ncx q[0], q[3];\ncx q[0], q[4];\ncx q[0], q[5];\ncx q[0], q[6];\ncx q[0], q[7];\ncx q[0], q[8];\ncx q[0], q[9];\ncy q[1], q[2];\ncz q[2], q[3];\nswap q[5], q[4];\n\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\nmeasure q[4] -> c[4];\nmeasure q[5] -> c[5];\nmeasure q[6] -> c[6];\nmeasure q[7] -> c[7];\nmeasure q[8] -> c[8];\nmeasure q[9] -> c[9];'
    qasm3_ideal = 'OPENQASM 3.0;\n\ninclude "stdgates.inc";\n\nqubit[10] q;\nbit[10] c;\n\nx q[0];\ny q[1];\nz q[2];\nh q[3];\ns q[4];\nt q[6];\nu3(1.570000,1.046667,0.785000) q[9];\nrx(0.500000) q[0];\nry(0.600000) q[1];\nrz(0.700000) q[2];\ncx q[0], q[1];\ncx q[0], q[2];\ncx q[0], q[3];\ncx q[0], q[4];\ncx q[0], q[5];\ncx q[0], q[6];\ncx q[0], q[7];\ncx q[0], q[8];\ncx q[0], q[9];\ncy q[1], q[2];\ncz q[2], q[3];\nswap q[5], q[4];\n\nc[0] = measure q[0];\nc[1] = measure q[1];\nc[2] = measure q[2];\nc[3] = measure q[3];\nc[4] = measure q[4];\nc[5] = measure q[5];\nc[6] = measure q[6];\nc[7] = measure q[7];\nc[8] = measure q[8];\nc[9] = measure q[9];'
    qasm2_str = c.to_qasm(format="qasm2", target_sdk="qiskit")
    qasm3_str = c.to_qasm(format="qasm3", target_sdk="qiskit")

    assert (qasm3_str == qasm3_ideal) and (qasm2_str == qasm2_ideal)


def test_braket_qasm_generation():
    pi = 3.14
    # --- build a 10-qubit circuit using every implemented operation once ---
    c = QuantumCircuit(number_of_qubits=10, number_of_classical_bits=10)

    # Single-qubit Clifford + phase family
    c.add_x_gate(0)
    c.add_y_gate(1)
    c.add_z_gate(2)
    c.add_h_gate(3)
    c.add_s_gate(4)
    c.add_t_gate(6)

    c.add_u_gate(9, theta=pi / 2, phi=pi / 3, lambda_parameter=pi / 4)

    # Rotations
    c.add_rx_gate(0, theta=0.5)
    c.add_ry_gate(1, theta=0.6)
    c.add_rz_gate(2, theta=0.7)

    # Two-qubit / controlled
    c.add_cx_gate(
        [0 for _ in range(c.number_of_qubits - 1)],
        [1 + i for i in range(c.number_of_qubits - 1)],
    )
    c.add_cy_gate(1, 2)
    c.add_cz_gate(2, 3)
    c.add_swap_gate(4, 5)  # your current add_swap_gate signature

    # Measure all qubits into all classical bits (1:1 mapping)
    c.add_measurement(list(range(10)), list(range(10)))

    qasm_ideal = "OPENQASM 3.0;\n\n\nqubit[10] q;\nbit[10] c;\n\nx q[0];\ny q[1];\nz q[2];\nh q[3];\ns q[4];\nt q[6];\nU(1.570000,1.046667,0.785000) q[9];\nrx(0.500000) q[0];\nry(0.600000) q[1];\nrz(0.700000) q[2];\ncnot q[0], q[1];\ncnot q[0], q[2];\ncnot q[0], q[3];\ncnot q[0], q[4];\ncnot q[0], q[5];\ncnot q[0], q[6];\ncnot q[0], q[7];\ncnot q[0], q[8];\ncnot q[0], q[9];\ncy q[1], q[2];\ncz q[2], q[3];\nswap q[5], q[4];\n\nc[0] = measure q[0];\nc[1] = measure q[1];\nc[2] = measure q[2];\nc[3] = measure q[3];\nc[4] = measure q[4];\nc[5] = measure q[5];\nc[6] = measure q[6];\nc[7] = measure q[7];\nc[8] = measure q[8];\nc[9] = measure q[9];"
    qasm_str = c.to_qasm(format="qasm3", target_sdk="braket")

    assert qasm_str == qasm_ideal


def test_tket_qasm_generation():
    pi = 3.14
    # --- build a 10-qubit circuit using every implemented operation once ---
    c = QuantumCircuit(number_of_qubits=10, number_of_classical_bits=10)

    # Single-qubit Clifford + phase family
    c.add_x_gate(0)
    c.add_y_gate(1)
    c.add_z_gate(2)
    c.add_h_gate(3)
    c.add_s_gate(4)
    c.add_t_gate(6)

    c.add_u_gate(9, theta=pi / 2, phi=pi / 3, lambda_parameter=pi / 4)

    # Rotations
    c.add_rx_gate(0, theta=0.5)
    c.add_ry_gate(1, theta=0.6)
    c.add_rz_gate(2, theta=0.7)

    # Two-qubit / controlled
    c.add_cx_gate(
        [0 for _ in range(c.number_of_qubits - 1)],
        [1 + i for i in range(c.number_of_qubits - 1)],
    )
    c.add_cy_gate(1, 2)
    c.add_cz_gate(2, 3)
    c.add_swap_gate(4, 5)  # your current add_swap_gate signature

    # Measure all qubits into all classical bits (1:1 mapping)
    c.add_measurement(list(range(10)), list(range(10)))

    qasm_ideal = 'OPENQASM 2.0;\n\ninclude "qelib1.inc";\n\nqreg q[10];\ncreg c[10];\n\nx q[0];\ny q[1];\nz q[2];\nh q[3];\ns q[4];\nt q[6];\nu3(1.570000,1.046667,0.785000) q[9];\nrx(0.500000) q[0];\nry(0.600000) q[1];\nrz(0.700000) q[2];\ncx q[0], q[1];\ncx q[0], q[2];\ncx q[0], q[3];\ncx q[0], q[4];\ncx q[0], q[5];\ncx q[0], q[6];\ncx q[0], q[7];\ncx q[0], q[8];\ncx q[0], q[9];\ncy q[1], q[2];\ncz q[2], q[3];\nswap q[5], q[4];\n\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\nmeasure q[4] -> c[4];\nmeasure q[5] -> c[5];\nmeasure q[6] -> c[6];\nmeasure q[7] -> c[7];\nmeasure q[8] -> c[8];\nmeasure q[9] -> c[9];'
    qasm_str = c.to_qasm(format="qasm2", target_sdk="tket")

    assert qasm_str == qasm_ideal
