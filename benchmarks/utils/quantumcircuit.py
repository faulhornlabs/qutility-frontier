import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QasmEmitterOptions:
    """
    Configuration options for building a QASM circuit emitter.

    Attributes:
        format (str): QASM version to use. Expected values: "qasm2" or "qasm3".
        target_sdk (Optional[str]): Target quantum SDK to match gate names for.
            Valid values include "qiskit", "braket", "tket", or "custom". If None,
            the emitter uses "default" mappings for the selected format.
        includes (Optional[Sequence[str]]): Optional list of external include‐file paths.
        float_precision (int): Number of decimal places used when formatting floats.
        custom_template (Optional[Dict[str, Any]]): Optional user-defined gate alias overrides.
        map (Dict[str, str]): Effective gate-name → alias mapping resolved at init-time.
    """

    format: str = "qasm2"
    target_sdk: Optional[str] = None
    includes: Optional[str | Sequence[str]] = None
    float_precision: int = 6
    custom_template: Optional[Dict[str, Any]] = None

    # Resolved mapping (computed in __post_init__)
    map: Dict[str, str] = field(init=False)

    _TARGET_SDK: ClassVar[List[str]] = ["qiskit", "braket", "tket"]
    _GATE_NAME_MAP_TEMPLATES: ClassVar[Dict[str, Dict[str, Dict[str, str]]]] = {
        "qasm2": {
            "default": {
                "x": "x",
                "y": "y",
                "z": "z",
                "h": "h",
                "s": "s",
                "t": "t",
                "u": "u3",
                "cx": "cx",
                "cy": "cy",
                "cz": "cz",
                "swap": "swap",
                "rx": "rx",
                "ry": "ry",
                "rz": "rz",
                "measure": "measure",
            },
            "qiskit": {
                "x": "x",
                "y": "y",
                "z": "z",
                "h": "h",
                "s": "s",
                "t": "t",
                "u": "u3",
                "cx": "cx",
                "cy": "cy",
                "cz": "cz",
                "swap": "swap",
                "rx": "rx",
                "ry": "ry",
                "rz": "rz",
                "measure": "measure",
            },
            "tket": {
                "x": "x",
                "y": "y",
                "z": "z",
                "h": "h",
                "s": "s",
                "t": "t",
                "p": "p",
                "u": "u3",
                "cx": "cx",
                "cy": "cy",
                "cz": "cz",
                "swap": "swap",
                "rx": "rx",
                "ry": "ry",
                "rz": "rz",
                "measure": "measure",
            },
        },
        "qasm3": {
            "default": {
                "x": "x",
                "y": "y",
                "z": "z",
                "h": "h",
                "s": "s",
                "t": "t",
                "p": "p",
                "u": "u",
                "cx": "cx",
                "cy": "cy",
                "cz": "cz",
                "ch": "ch",
                "swap": "swap",
                "rx": "rx",
                "ry": "ry",
                "rz": "rz",
                "measure": "measure",
            },
            "qiskit": {
                "x": "x",
                "y": "y",
                "z": "z",
                "h": "h",
                "s": "s",
                "t": "t",
                "p": "p",
                "u": "u3",
                "cx": "cx",
                "cy": "cy",
                "cz": "cz",
                "ch": "ch",
                "swap": "swap",
                "rx": "rx",
                "ry": "ry",
                "rz": "rz",
                "measure": "measure",
            },
            "braket": {
                "x": "x",
                "y": "y",
                "z": "z",
                "h": "h",
                "s": "s",
                "t": "t",
                "u": "U",
                "cx": "cnot",
                "cy": "cy",
                "cz": "cz",
                "ch": "ch",
                "swap": "swap",
                "rx": "rx",
                "ry": "ry",
                "rz": "rz",
                "measure": "measure",
            },
        },
    }

    _DEFAULT_INCLUDES: ClassVar[Dict[str, Dict[str, Sequence[str]]]] = {
        "qasm2": {
            "default": ["qelib1.inc"],
            "qiskit": ["qelib1.inc"],
            "tket": ["qelib1.inc"],
        },
        "qasm3": {
            "default": ["stdgates.inc"],
            "qiskit": ["stdgates.inc"],  # if Qiskit uses different include for qasm3
            "braket": [],
        },
    }

    def __post_init__(self):
        """Perform validation and resolve effective mapping and includes."""
        if self.float_precision < 0:
            raise ValueError("float_precision must be non-negative")

        format_lower = (self.format or "").lower()
        if format_lower not in self._GATE_NAME_MAP_TEMPLATES:
            raise ValueError(
                f"Unsupported format: {self.format!r}. Expected 'qasm2' or 'qasm3'."
            )

        sdk_input = (self.target_sdk or "default").lower()
        if sdk_input not in {"default", "custom"} and sdk_input not in self._TARGET_SDK:
            raise ValueError(
                f"Invalid value for target_sdk: {self.target_sdk!r}. "
                f"Use one of {self._TARGET_SDK + ['custom']} or None."
            )

        effective_map = self._resolve_effective_map(
            format_lower, sdk_input if sdk_input != "custom" else "default"
        )

        if sdk_input in self._TARGET_SDK:
            default_includes = self._DEFAULT_INCLUDES[format_lower].get(
                sdk_input, self._DEFAULT_INCLUDES[format_lower]["default"]
            )
            object.__setattr__(self, "includes", tuple(default_includes))

        includes_input = self.includes
        if includes_input is None:
            normalized_includes = None
        else:
            if isinstance(includes_input, str):
                normalized_includes = [includes_input]
            else:
                try:
                    normalized_includes = list(includes_input)
                except TypeError:
                    raise ValueError(
                        f"includes must be a string or sequence of strings, got {type(includes_input).__name__}"
                    )
                for idx, inc in enumerate(normalized_includes):
                    if not isinstance(inc, str):
                        raise ValueError(
                            f"includes[{idx}] must be a string, got {type(inc).__name__}"
                        )
            object.__setattr__(self, "includes", tuple(normalized_includes))

        if sdk_input == "custom" and not self.custom_template:
            logger.warning("No custom gates defined; using default mappings instead.")

        object.__setattr__(self, "target_sdk", sdk_input)
        object.__setattr__(self, "map", effective_map)

    def _resolve_effective_map(
        self, format_lower: str, sdk_choice: str
    ) -> Dict[str, str]:
        """
        Resolve default → sdk → custom_template into a single alias map.

        Args:
            format_lower (str): Lower-case QASM format (e.g., "qasm2" or "qasm3").
            sdk_choice (str): Selected SDK key after normalization ("default", "qiskit", etc).

        Returns:
            Dict[str, str]: Mapping from internal gate names to QASM names.
        """
        all_for_format = self._GATE_NAME_MAP_TEMPLATES.get(format_lower, {})
        default_map = all_for_format.get("default", {})
        sdk_map = all_for_format.get(sdk_choice, {})

        merged_map: Dict[str, str] = dict(default_map)
        merged_map.update(sdk_map)

        if self.custom_template:
            for name in self.custom_template:
                if name not in default_map:
                    logger.warning(
                        f"Custom gate '{name}' is new (not in the default list) for format '{format_lower}'."
                    )
            merged_map.update(self.custom_template)  # type: ignore[arg-type]

        return merged_map

    def get_qasm_name(self, internal_name: str) -> str:
        """
        Retrieve the effective QASM gate name for a given internal gate name.

        Args:
            internal_name (str): The internal gate name.

        Returns:
            str: The corresponding QASM gate name. If the gate is not mapped,
                 the internal name is returned unchanged.
        """
        return self.map.get(internal_name, internal_name)


@dataclass(slots=True)
class QuantumGate:
    """
    Basic quantum gate (data only).

    Attributes:
        name (str): The internal name of the gate.
        target_qubits (Sequence[int]): The qubit(s) on which this gate acts.
        parameters (List[float]): Optional list of parameter values.
    """

    name: str
    target_qubits: int | Sequence[int]
    parameters: float | Sequence[float] | None = None

    def __post_init__(self) -> None:
        """Normalize target_qubits and parameters into canonical forms."""
        if isinstance(self.target_qubits, int):
            self.target_qubits = [self.target_qubits]
        else:
            self.target_qubits = [int(q) for q in self.target_qubits]

        param = self.parameters
        if param is None:
            param_list: List[float] = []
        elif isinstance(param, (int, float)):
            param_list = [param]
        else:
            try:
                param_list = list(param)
            except TypeError as exc:
                raise TypeError(
                    f"Gate parameters must be None, a number, or an iterable of numbers "
                    f"(got {type(param).__name__})."
                ) from exc

        try:
            self.parameters = [float(v) for v in param_list]
        except (TypeError, ValueError) as exc:
            raise TypeError("Gate parameters must be convertible to float.") from exc

    @property
    def is_two_qubit_gate(self) -> bool:
        """
        Whether this is a two-qubit (controlled) gate.

        Returns:
            bool: False for base class (single-qubit) gates.
        """
        return False


@dataclass(slots=True)
class TwoQubitQuantumGate(QuantumGate):
    """
    Controlled quantum gate.

    Attributes:
        control_qubits (Sequence[int]): The control qubit(s) for the gate.
    """

    control_qubits: int | Sequence[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize control_qubits and enforce at least one control."""

        super(TwoQubitQuantumGate, self).__post_init__()  # run parent normalization

        if isinstance(self.control_qubits, int):
            self.control_qubits = [self.control_qubits]
        else:
            self.control_qubits = [int(q) for q in self.control_qubits]

        if len(self.control_qubits) == 0:
            raise ValueError("TwoQubitQuantumGate requires at least one control qubit.")

    @property
    def is_two_qubit_gate(self) -> bool:
        """
        Whether this is a two-qubit (controlled) gate.

        Returns:
            bool: True for this subclass.
        """
        return True


class QuantumCircuit:
    """
    Class representing a quantum circuit.

    Attributes:
        number_of_qubits (int): Total number of qubits in the circuit.
        number_of_classical_bits (int): Total number of classical bits in the circuit.
        gate_list (List[QuantumGate]): List of gate objects in the circuit.
        measurements (List[Tuple[int, int]]): List of measurements as (qubit, classical_bit).
        reset_operations (List[int]): List of qubits that have been reset.
    """

    def __init__(self, number_of_qubits: int = 0, number_of_classical_bits: int = 0):
        """
        Initialize the quantum circuit.

        Args:
            number_of_qubits (int, optional): Number of qubits. Default is 0.
            number_of_classical_bits (int, optional): Number of classical bits. Default is 0.
        """
        self.number_of_qubits = number_of_qubits
        self.number_of_classical_bits = number_of_classical_bits
        self.gate_list: List[QuantumGate] = []
        self.measurements: List[tuple[int, int]] = []
        self.reset_operations: List[int] = []

    def __repr__(self) -> str:
        """
        Return a string representation of the quantum circuit.

        Returns:
            str: A multi-line string summarizing the circuit's characteristics.
        """
        single_qubit_gate_count = sum(
            len(gate.target_qubits)
            for gate in self.gate_list
            if not gate.is_two_qubit_gate
        )
        two_qubit_gate_count = sum(
            len(gate.target_qubits) for gate in self.gate_list if gate.is_two_qubit_gate
        )
        measurement_count = len(self.measurements)

        return (
            f"Number of qubits: {self.number_of_qubits}\n"
            f"Number of classical bits: {self.number_of_classical_bits}\n"
            f"Single-qubit gates: {single_qubit_gate_count}\n"
            f"Two-qubit gates: {two_qubit_gate_count}\n"
            f"Measurements: {measurement_count}"
        )

    def display_gate_descriptions(self) -> None:
        """
        Display descriptions of each gate supported by the circuit.

        Prints each gate's internal name along with its description.
        """
        gate_descriptions = {
            "x": "Pauli X gate",
            "y": "Pauli Y gate",
            "z": "Pauli Z gate",
            "h": "Hadamard gate",
            "s": "Phase S gate",
            "t": "T gate",
            "p": "Phase rotation gate",
            "u": "General single-qubit unitary gate",
            "cx": "Controlled-X gate (CNOT)",
            "cy": "Controlled-Y gate",
            "cz": "Controlled-Z gate",
            "swap": "Swap gate",
            "rx": "Rotation about the X-axis",
            "ry": "Rotation about the Y-axis",
            "rz": "Rotation about the Z-axis",
            "measurement": "Measurement operation",
        }
        for gate_name, description in gate_descriptions.items():
            print(f"{gate_name} – {description}")

    def add_qubits(self, additional_qubits: int) -> None:
        """
        Increase the number of qubits in the circuit.

        Args:
            additional_qubits (int): Number of qubits to add.
        """
        self.number_of_qubits += additional_qubits

    def add_classical_bits(self, additional_classical_bits: int) -> None:
        """
        Increase the number of classical bits in the circuit.

        Args:
            additional_classical_bits (int): Number of classical bits to add.
        """
        self.number_of_classical_bits += additional_classical_bits

    def _add_gate(
        self,
        gate_name: str,
        target_qubits: int | Sequence[int],
        parameters: Optional[Sequence[float]] = None,
        control_qubits: Optional[int | Sequence[int]] = None,
    ) -> None:
        """
        Private method to add a gate to the circuit.

        Args:
            gate_name (str): The internal name of the gate.
            target_qubits (int or list[int]): The target qubit(s) for the gate.
            parameters (Sequence[float], optional): Parameters for the gate.
            control_qubits (int or list[int], optional): Control qubit(s) if the gate is controlled.
        """
        if control_qubits is None:
            # single-qubit gate
            new_quantum_gate = QuantumGate(
                name=gate_name, target_qubits=target_qubits, parameters=parameters
            )
        else:
            # controlled / multi-qubit gate
            new_quantum_gate = TwoQubitQuantumGate(
                name=gate_name,
                target_qubits=target_qubits,
                parameters=parameters,
                control_qubits=control_qubits,
            )

        self.gate_list.append(new_quantum_gate)

    # Single-qubit gate methods
    def add_x_gate(self, qubit: int) -> None:
        """Add an X gate to the circuit."""
        self._add_gate("x", qubit)

    def add_y_gate(self, qubit: int) -> None:
        """Add a Y gate to the circuit."""
        self._add_gate("y", qubit)

    def add_z_gate(self, qubit: int) -> None:
        """Add a Z gate to the circuit."""
        self._add_gate("z", qubit)

    def add_h_gate(self, qubit: int) -> None:
        """Add a Hadamard gate to the circuit."""
        self._add_gate("h", qubit)

    def add_s_gate(self, qubit: int) -> None:
        """Add an S gate to the circuit."""
        self._add_gate("s", qubit)

    def add_t_gate(self, qubit: int) -> None:
        """Add a T gate to the circuit."""
        self._add_gate("t", qubit)

    def add_u_gate(
        self, qubit: int, theta: float, phi: float, lambda_parameter: float
    ) -> None:
        """
        Add a general single-qubit unitary gate to the circuit.

        Args:
            qubit (int): The target qubit.
            theta (float): The theta rotation parameter.
            phi (float): The phi rotation parameter.
            lambda_parameter (float): The lambda rotation parameter.
        """
        self._add_gate("u", qubit, [theta, phi, lambda_parameter])

    # Two-qubit gate methods
    def add_cx_gate(self, control_qubit: int, target_qubit: int) -> None:
        """Add a controlled-X (CNOT) gate to the circuit."""
        self._add_gate("cx", target_qubit, control_qubits=control_qubit)

    def add_cy_gate(self, control_qubit: int, target_qubit: int) -> None:
        """Add a controlled-Y gate to the circuit."""
        self._add_gate("cy", target_qubit, control_qubits=control_qubit)

    def add_cz_gate(self, control_qubit: int, target_qubit: int) -> None:
        """Add a controlled-Z gate to the circuit."""
        self._add_gate("cz", target_qubit, control_qubits=control_qubit)

    def add_swap_gate(self, qubit_one: int, qubit_two: int) -> None:
        """Add a SWAP gate to the circuit."""
        self._add_gate("swap", target_qubits=qubit_one, control_qubits=qubit_two)

    # Rotation gate methods
    def add_rx_gate(self, qubit: int, theta: float) -> None:
        """
        Add a rotation about the X-axis gate to the circuit.

        Args:
            qubit (int): The target qubit.
            theta (float): The rotation angle.
        """
        self._add_gate("rx", qubit, [theta])

    def add_ry_gate(self, qubit: int, theta: float) -> None:
        """
        Add a rotation about the Y-axis gate to the circuit.

        Args:
            qubit (int): The target qubit.
            theta (float): The rotation angle.
        """
        self._add_gate("ry", qubit, [theta])

    def add_rz_gate(self, qubit: int, theta: float) -> None:
        """
        Add a rotation about the Z-axis gate to the circuit.

        Args:
            qubit (int): The target qubit.
            theta (float): The rotation angle.
        """
        self._add_gate("rz", qubit, [theta])

    def add_measurement(
        self, qubit: int | Sequence[int], classical_bit: int | Sequence[int]
    ) -> None:
        """
        Add measurement(s) to the circuit.

        Args:
            qubit (int or list[int]): The qubit(s) to measure.
            classical_bit (int or list[int]): The classical bit(s) where the measurement is stored.

        Raises:
            ValueError: If qubit and classical_bit are lists of different lengths.
        """
        if isinstance(qubit, (list, tuple, np.ndarray)):
            if isinstance(classical_bit, (list, tuple, np.ndarray)):
                if len(qubit) != len(classical_bit):
                    raise ValueError(
                        "Length of qubit list must match length of classical_bit list."
                    )
                for q, c in zip(qubit, classical_bit):
                    self.measurements.append((q, c))
            else:
                for q in qubit:
                    self.measurements.append((q, classical_bit))
        else:
            self.measurements.append((qubit, classical_bit))

    def add_reset_operation(self, qubit: int) -> None:
        """
        Add a reset operation for a specified qubit.

        Args:
            qubit (int): The qubit to reset.
        """
        self.reset_operations.append(qubit)

    def single_qubit_gate_count(self):
        """
        Return the number of gates
        """
        c = sum(
            len(gate.target_qubits)
            for gate in self.gate_list
            if not gate.is_two_qubit_gate
        )

        return c

    def two_qubit_gate_count(self):
        """
        Return the number of two qubit gates
        """
        c = sum(
            len(gate.target_qubits) for gate in self.gate_list if gate.is_two_qubit_gate
        )

        return c

    def to_qasm(
        self,
        emitter_options: Optional[QasmEmitterOptions] = None,
        *,
        format: str = "qasm2",
        target_sdk: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate QASM code for the circuit.

        Args:
            emitter_options (Optional[QasmEmitterOptions]): Pre-configured emitter options. If None,
                a new QasmEmitterOptions object is created based on format and target_sdk.
            format (str, optional): QASM format to use if emitter_options is None. Default is "qasm2".
            target_sdk (Optional[str], optional): Target SDK name if emitter_options is None. Default is None.
            **kwargs: Additional keyword arguments forwarded to QasmEmitterOptions.

        Returns:
            str: The QASM source code as a multi-line string.
        """
        if emitter_options is None:
            emitter_options = QasmEmitterOptions(
                format=format, target_sdk=target_sdk, **kwargs
            )
        emitter = QasmEmitter(emitter_options)
        return emitter.emit(self)

    def draw_circuit_diagram(self, max_length=20):
        """
        Print a textual representation of the quantum circuit.

        If any line exceeds the maximum line length, the diagram is split into vertical blocks.

        Arguments:
            max_length (int): Maximum number of gate-length units per block.

        Returns:
            None
        """
        gate_length = 7
        max_line_length = gate_length * max_length
        diagram_lines = ["" for _ in range(2 * self.number_of_qubits)]

        for i in range(self.number_of_qubits):
            qubit_label = f"q{i}"
            diagram_lines[2 * i] = qubit_label + " --"
            diagram_lines[2 * i + 1] = " " * (len(qubit_label) + 3)

        for gate in self.gate_list:
            if gate.is_two_qubit_gate:
                for index in range(len(gate.target_qubits)):
                    control_qubit = gate.control_qubits[index]
                    target_qubit = gate.target_qubits[index]
                    min_qubit = min(control_qubit, target_qubit)
                    max_qubit = max(control_qubit, target_qubit)

                    current_max_length = max(
                        len(diagram_lines[2 * j])
                        for j in range(min_qubit, max_qubit + 1)
                    )
                    for j in range(min_qubit, max_qubit + 1):
                        if len(diagram_lines[2 * j]) < current_max_length:
                            padding = current_max_length - len(diagram_lines[2 * j])
                            diagram_lines[2 * j] += "-" * padding
                            diagram_lines[2 * j + 1] += " " * padding

                    control_str = "--" + f"[{gate.name[0]}]" + "--"
                    diagram_lines[2 * control_qubit] += control_str
                    for j in range(min_qubit + 1, max_qubit):
                        pad_half = (len(control_str) - 1) // 2
                        diagram_lines[2 * j] += "-" * pad_half + "|" + "-" * pad_half
                    for j in range(min_qubit, max_qubit):
                        pad_half = (len(control_str) - 1) // 2
                        diagram_lines[2 * j + 1] += (
                            " " * pad_half + "|" + " " * pad_half
                        )

                    target_str_padding = (6 - len(gate.name)) // 2
                    target_str = (
                        "-" * target_str_padding
                        + f"[{gate.name[1:]}]"
                        + "-" * target_str_padding
                    )
                    diagram_lines[2 * target_qubit] += target_str
                    diagram_lines[2 * max_qubit + 1] += " " * len(target_str)
            else:
                for target in gate.target_qubits:
                    if len(gate.name) % 2 == 0:
                        gate_str = (
                            "-" * ((5 - len(gate.name)) // 2)
                            + f"[{gate.name}]-"
                            + "-" * ((5 - len(gate.name)) // 2)
                        )
                    else:
                        gate_str = (
                            "-" * ((5 - len(gate.name)) // 2)
                            + f"[{gate.name}]"
                            + "-" * ((5 - len(gate.name)) // 2)
                        )
                    diagram_lines[2 * target] += gate_str
                    diagram_lines[2 * target + 1] += " " * len(gate_str)

        for measurement in self.measurements:
            if isinstance(measurement[0], int):
                qubit_index = measurement[0]
                measurement_str = "--" + "[M]" + "=="
                diagram_lines[2 * qubit_index] += measurement_str
                diagram_lines[2 * qubit_index + 1] += " " * len(measurement_str)
            else:
                for qubit_index in measurement[0]:
                    measurement_str = "--" + "[M]" + "=="
                    diagram_lines[2 * qubit_index] += measurement_str
                    diagram_lines[2 * qubit_index + 1] += " " * len(measurement_str)

        max_width = max(len(line) for line in diagram_lines)
        output_lines = []
        for start in range(0, max_width, max_line_length):
            for line in diagram_lines:
                output_lines.append(line[start : start + max_line_length])
            output_lines.append("")

        print("\n".join(output_lines))


class QasmEmitter:
    """
    Translates a QuantumCircuit into OpenQASM source code.

    This class converts the internal circuit representation into textual
    QASM output according to the configuration in QasmEmitterOptions.

    It handles:
      • Header emission (OPENQASM, includes)
      • Register declarations
      • Gate translation and parameter formatting
      • Measurements and reset operations
    """

    def __init__(self, options: QasmEmitterOptions):
        """
        Initialize the emitter with a given configuration.

        Args:
            options (QasmEmitterOptions): Options object specifying translation format,
                includes, float precision, and SDK gate mappings.
        """
        self.options = options

    def emit(self, circuit: QuantumCircuit) -> str:
        """
        Generate a full QASM program for the given circuit.

        Args:
            circuit (QuantumCircuit): The circuit instance to translate.

        Returns:
            str: The QASM source code as a multi-line string.
        """
        lines: List[str] = []

        format_lower = self.options.format.lower()
        includes_list = self.options.includes

        if format_lower == "qasm2":
            lines.append("OPENQASM 2.0;")
            qubit_declaration = f"qreg q[{circuit.number_of_qubits}];"
            classical_declaration = f"creg c[{circuit.number_of_classical_bits}];"
        elif format_lower == "qasm3":
            lines.append("OPENQASM 3.0;")
            qubit_declaration = f"qubit[{circuit.number_of_qubits}] q;"
            classical_declaration = f"bit[{circuit.number_of_classical_bits}] c;"
        else:
            raise ValueError(f"Unsupported QASM format: {self.options.format!r}")

        lines.append("")  # add empty line to make the QASM code clearer

        if includes_list:
            for inc in includes_list:
                lines.append(f'include "{inc}";')
        lines.append("")  # add empty line to make the QASM code clearer

        lines.extend([qubit_declaration, classical_declaration])
        lines.append("")  # add empty line to make the QASM code clearer

        for gate in circuit.gate_list:
            lines.extend(self._emit_gate(gate))

        for qubit in circuit.reset_operations:
            lines.append(f"reset q[{qubit}];")

        lines.append("")  # add empty line to make the QASM code clearer

        for qubit, classical_bit in circuit.measurements:
            measure_name = self.options.get_qasm_name("measure")
            if format_lower == "qasm2":
                lines.append(f"measure q[{qubit}] -> c[{classical_bit}];")
            elif format_lower == "qasm3":
                lines.append(f"c[{classical_bit}] = {measure_name} q[{qubit}];")

        return "\n".join(lines)

    def _emit_gate(self, gate: QuantumGate) -> List[str]:
        """
        Emit one or more QASM instructions for a single quantum gate.

        Args:
            gate (QuantumGate): A QuantumGate or TwoQubitQuantumGate instance.

        Returns:
            List[str]: Lines of QASM code representing this gate.
        """
        qasm_name = self.options.get_qasm_name(gate.name)
        lines: List[str] = []

        if gate.parameters:
            fmt = f"{{:.{self.options.float_precision}f}}"
            params_str = "(" + ",".join(fmt.format(p) for p in gate.parameters) + ")"
        else:
            params_str = ""

        if not gate.is_two_qubit_gate:
            for tgt in gate.target_qubits:
                lines.append(f"{qasm_name}{params_str} q[{tgt}];")
            return lines

        ctrl_qubits = getattr(gate, "control_qubits", [])
        tgt_qubits = gate.target_qubits
        number_of_target_qubits = len(tgt_qubits)

        for i in range(number_of_target_qubits):
            lines.append(
                f"{qasm_name}{params_str} q[{ctrl_qubits[i]}], q[{tgt_qubits[i]}];"
            )

        return lines
