import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================
# QASM Emitter Options
# =============================================================


@dataclass(frozen=True)
class QasmEmitterOptions:
    """
    Configuration options that control how a :class:`QasmEmitter` translates
    a :class:`QuantumCircuit` into OpenQASM source.

    Attributes:
        format: OpenQASM version to target. Supported values are
            ``"qasm2"`` and ``"qasm3"``.
        target_sdk: Optional SDK dialect for gate aliases. If provided,
            it adjusts emitted gate names to match that SDK. Supported
            values: ``"qiskit"``, ``"braket"``, ``"tket"``, or ``"custom"``.
            ``None`` means use the format's default aliasing.
        includes: Optional include file(s). May be a single string path or a
            sequence of paths. If ``target_sdk`` is one of the known SDKs,
            this is automatically populated with sensible defaults for the
            selected ``format``.
        float_precision: Number of decimal places when formatting floating
            parameters in QASM output.
        custom_template: Optional mapping of *internal gate names* to the
            desired *QASM names* that overrides the selected format/SDK
            template.
        map: (Computed) Effective internal-gate → QASM-gate alias mapping.
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
                "sdg": "sdg",
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
                "sdg": "sdg",
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
                "sdg": "sdg",
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
                "sdg": "sdg",
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
                "sdg": "sdg",
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
                "sdg": "si",
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
            "qiskit": ["stdgates.inc"],
            "braket": [],
        },
    }

    def __post_init__(self) -> None:
        """Validate inputs and resolve effective alias mapping and includes."""
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

        # Auto-fill includes for known SDKs
        if sdk_input in self._TARGET_SDK:
            default_includes = self._DEFAULT_INCLUDES[format_lower].get(
                sdk_input, self._DEFAULT_INCLUDES[format_lower]["default"]
            )
            object.__setattr__(self, "includes", tuple(default_includes))

        # Normalize includes type/values
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
                    raise ValueError("includes must be a string or sequence of strings")
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
        Merge the default alias map for ``format_lower``, the SDK-specific map,
        and any user-provided ``custom_template`` into a single mapping.

        Args:
            format_lower: Lower-case QASM format (``"qasm2"`` or ``"qasm3"``).
            sdk_choice: Selected SDK key after normalization (e.g., ``"default"``,
                ``"qiskit"``).

        Returns:
            A mapping from internal gate names to QASM gate names.
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
                        "Custom gate '%s' is new (not in the default list) for format '%s'.",
                        name,
                        format_lower,
                    )
            merged_map.update(self.custom_template)  # type: ignore[arg-type]

        return merged_map

    def get_qasm_name(self, internal_name: str) -> str:
        """
        Translate an internal gate name to its QASM alias.

        If the gate is not present in the mapping, the original name is
        returned unchanged.
        """
        return self.map.get(internal_name, internal_name)


# =============================================================
# Gate Data Structures
# =============================================================


@dataclass(slots=True)
class QuantumGate:
    """
    Data container for a (possibly parameterized) quantum gate.

    Attributes:
        name: Internal gate name (e.g., ``"x"``, ``"rz"``, ``"cx"``).
        target_qubits: Index/indices of the qubits this gate acts on.
        parameters: Optional numeric parameter or sequence of parameters.
            Values are normalized to a list of ``float``.
    """

    name: str
    target_qubits: int | Sequence[int]
    parameters: float | Sequence[float] | None = None

    def __post_init__(self) -> None:
        """Normalize ``target_qubits`` and ``parameters`` to canonical forms."""

        # Normalize target_qubits to list[int]
        val = self.target_qubits

        # Check if it's a *sequence* (list-like) but **not** a string or bytes
        if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            normalized = []
            for q in val:
                if (
                    isinstance(q, Real)
                    and not isinstance(q, bool)
                    and float(q).is_integer()
                ):
                    normalized.append(int(q))
                else:
                    raise ValueError(
                        f"Each target_qubit must be integer-valued, got {q} in {val}"
                    )
            self.target_qubits = normalized

        # Otherwise: treat as scalar
        else:
            if (
                isinstance(val, Real)
                and not isinstance(val, bool)
                and float(val).is_integer()
            ):
                self.target_qubits = [int(val)]
            else:
                raise ValueError(f"target_qubits must be integer-valued, got {val}")

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
                    "Gate parameters must be None, a number, or an iterable of numbers."
                ) from exc

        try:
            self.parameters = [float(v) for v in param_list]
        except (TypeError, ValueError) as exc:
            raise TypeError("Gate parameters must be convertible to float.") from exc

    @property
    def is_two_qubit_gate(self) -> bool:
        """Return ``True`` if this gate is multi-qubit/controlled. Base: ``False``."""
        return False


@dataclass(slots=True)
class TwoQubitQuantumGate(QuantumGate):
    """A controlled or otherwise two‑qubit gate.

    Attributes:
        control_qubits: Index/indices of control qubits. At least one is required.
    """

    control_qubits: int | Sequence[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize ``control_qubits`` and enforce at least one control qubit."""
        # super().__post_init__()  # run parent normalization
        super(TwoQubitQuantumGate, self).__post_init__()  # run parent normalization

        if isinstance(self.control_qubits, int):
            self.control_qubits = [self.control_qubits]
        else:
            self.control_qubits = [int(q) for q in self.control_qubits]

        if len(self.control_qubits) == 0:
            raise ValueError("TwoQubitQuantumGate requires at least one control qubit.")

    @property
    def is_two_qubit_gate(self) -> bool:
        """Return ``True`` for this subclass."""
        return True


# =============================================================
# Circuit API
# =============================================================


class QuantumCircuit:
    """
    In-memory representation of a quantum circuit.

    The circuit stores qubit/bit registers plus an ordered list of gate
    operations, measurements, and resets. Convenience helpers are provided to
    append common gates and to export the circuit to OpenQASM.

    Attributes:
        number_of_qubits: Total number of qubits in the circuit.
        number_of_classical_bits: Total number of classical bits in the circuit.
        gate_list: Ordered list of gate objects.
        measurements: ``(qubit, cbit)`` pairs describing where measurement
            outcomes are stored.
        reset_operations: Indices of qubits that are reset.
    """

    def __init__(self, number_of_qubits: int = 0, number_of_classical_bits: int = 0):
        """Create an empty circuit with ``number_of_qubits`` and classical bits."""
        self.number_of_qubits = number_of_qubits
        self.number_of_classical_bits = number_of_classical_bits
        self.gate_list: List[QuantumGate] = []
        self.measurements: List[tuple[int, int]] = []
        self.reset_operations: List[int] = []

    def __repr__(self) -> str:
        """Return a human-readable summary of the circuit contents."""
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

    # -----------------------------
    # Discovery & reference helpers
    # -----------------------------
    def display_gate_descriptions(self) -> None:
        """Print a short description for each supported internal gate name."""
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
            "measure": "Measurement operation",
        }
        for gate_name, description in gate_descriptions.items():
            print(f"{gate_name} – {description}")

    # -----------------------------
    # Register sizing
    # -----------------------------
    def add_qubits(self, additional_qubits: int) -> None:
        """Increase the number of qubits by ``additional_qubits``."""
        self.number_of_qubits += additional_qubits

    def add_classical_bits(self, additional_classical_bits: int) -> None:
        """Increase the number of classical bits by ``additional_classical_bits``."""
        self.number_of_classical_bits += additional_classical_bits

    # -----------------------------
    # Internal gate append helper
    # -----------------------------
    def _add_gate(
        self,
        gate_name: str,
        target_qubits: int | Sequence[int],
        parameters: Optional[Sequence[float]] = None,
        control_qubits: Optional[int | Sequence[int]] = None,
    ) -> None:
        """
        Append a gate to the circuit.

        Args:
            gate_name: Internal gate name, as used by :class:`QasmEmitterOptions`.
            target_qubits: Target qubit index (or indices) for the gate.
            parameters: Optional parameter list for parameterized gates.
            control_qubits: Optional control qubit index (or indices). If
                provided, a :class:`TwoQubitQuantumGate` is created.
        """
        if control_qubits is None:
            gate_obj = QuantumGate(
                name=gate_name, target_qubits=target_qubits, parameters=parameters
            )
        else:
            gate_obj = TwoQubitQuantumGate(
                name=gate_name,
                target_qubits=target_qubits,
                parameters=parameters,
                control_qubits=control_qubits,
            )
        self.gate_list.append(gate_obj)

    # -----------------------------
    # Single-qubit gates
    # -----------------------------
    def add_x_gate(self, qubit: int) -> None:
        """Append an X gate on ``qubit``."""
        self._add_gate("x", qubit)

    def add_y_gate(self, qubit: int) -> None:
        """Append a Y gate on ``qubit``."""
        self._add_gate("y", qubit)

    def add_z_gate(self, qubit: int) -> None:
        """Append a Z gate on ``qubit``."""
        self._add_gate("z", qubit)

    def add_h_gate(self, qubit: int) -> None:
        """Append a Hadamard gate on ``qubit``."""
        self._add_gate("h", qubit)

    def add_s_gate(self, qubit: int) -> None:
        """Append an S gate on ``qubit``."""
        self._add_gate("s", qubit)

    def add_sdg_gate(self, qubit: int) -> None:
        """Append an S gate on ``qubit``."""
        self._add_gate("sdg", qubit)

    def add_t_gate(self, qubit: int) -> None:
        """Append a T gate on ``qubit``."""
        self._add_gate("t", qubit)

    def add_u_gate(
        self, qubit: int, theta: float, phi: float, lambda_parameter: float
    ) -> None:
        """
        Append a general single-qubit unitary gate ``u(theta, phi, lambda)``.

        Args:
            qubit: Target qubit.
            theta: First Euler angle.
            phi: Second Euler angle.
            lambda_parameter: Third Euler angle.
        """
        self._add_gate("u", qubit, [theta, phi, lambda_parameter])

    # -----------------------------
    # Two-qubit gates
    # -----------------------------
    def add_cx_gate(self, control_qubit: int, target_qubit: int) -> None:
        """Append a controlled-X (CNOT) with ``control_qubit → target_qubit``."""
        self._add_gate("cx", target_qubit, control_qubits=control_qubit)

    def add_cy_gate(self, control_qubit: int, target_qubit: int) -> None:
        """Append a controlled-Y with ``control_qubit → target_qubit``."""
        self._add_gate("cy", target_qubit, control_qubits=control_qubit)

    def add_cz_gate(self, control_qubit: int, target_qubit: int) -> None:
        """Append a controlled-Z with ``control_qubit → target_qubit``."""
        self._add_gate("cz", target_qubit, control_qubits=control_qubit)

    def add_swap_gate(self, qubit_one: int, qubit_two: int) -> None:
        """
        Append a SWAP gate between ``qubit_one`` and ``qubit_two``.

        Notes:
            Internally this is modeled as a two-qubit gate and emitted as
            ``swap q[qubit_one], q[qubit_two];`` in QASM.
        """
        # For emission order (q[ctrl], q[tgt]) → (q1, q2), place q1 in control, q2 in target.
        self._add_gate("swap", target_qubits=qubit_two, control_qubits=qubit_one)

    # -----------------------------
    # Rotation gates
    # -----------------------------
    def add_rx_gate(self, qubit: int, theta: float) -> None:
        """Append an RX rotation of angle ``theta`` on ``qubit``."""
        self._add_gate("rx", qubit, [theta])

    def add_ry_gate(self, qubit: int, theta: float) -> None:
        """Append an RY rotation of angle ``theta`` on ``qubit``."""
        self._add_gate("ry", qubit, [theta])

    def add_rz_gate(self, qubit: int, theta: float) -> None:
        """Append an RZ rotation of angle ``theta`` on ``qubit``."""
        self._add_gate("rz", qubit, [theta])

    # -----------------------------
    # Measurement & reset
    # -----------------------------
    def add_measurement(
        self, qubit: int | Sequence[int], classical_bit: int | Sequence[int]
    ) -> None:
        """
        Append measurement(s).

        Args:
            qubit: Single qubit index or sequence of indices.
            classical_bit: Single classical bit index or sequence. If both are
                sequences, their lengths must match; otherwise a single
                classical bit is reused for each qubit.
        Raises:
            ValueError: If both arguments are sequences of different lengths.
        """
        if isinstance(qubit, (list, tuple, np.ndarray)):
            if isinstance(classical_bit, (list, tuple, np.ndarray)):
                if len(qubit) != len(classical_bit):
                    raise ValueError(
                        "Length of qubit list must match length of classical_bit list."
                    )
                for q, c in zip(qubit, classical_bit):
                    self.measurements.append((int(q), int(c)))
            else:
                for q in qubit:
                    self.measurements.append((int(q), int(classical_bit)))
        else:
            self.measurements.append((int(qubit), int(classical_bit)))

    def add_reset_operation(self, qubit: int) -> None:
        """Append a reset operation for the specified ``qubit``."""
        self.reset_operations.append(int(qubit))

    # -----------------------------
    # Summary helpers
    # -----------------------------
    def single_qubit_gate_count(self) -> int:
        """Return the number of single-qubit gate applications in the circuit."""
        return sum(
            len(g.target_qubits) for g in self.gate_list if not g.is_two_qubit_gate
        )

    def two_qubit_gate_count(self) -> int:
        """Return the number of two-qubit gate applications in the circuit."""
        return sum(len(g.target_qubits) for g in self.gate_list if g.is_two_qubit_gate)

    # -----------------------------
    # Rendering / export
    # -----------------------------
    def to_qasm(
        self,
        emitter_options: Optional[QasmEmitterOptions] = None,
        *,
        format: str = "qasm2",
        target_sdk: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate OpenQASM code for the circuit.

        Args:
            emitter_options: Pre-configured options. If omitted, a new
                :class:`QasmEmitterOptions` is created from ``format``,
                ``target_sdk`` and any extra ``kwargs``.
            format: QASM format used when building the default options.
            target_sdk: Optional SDK aliasing used when building default options.
            **kwargs: Passed to :class:`QasmEmitterOptions`.
        Returns:
            The QASM program as a multi-line ``str``.
        """
        if emitter_options is None:
            emitter_options = QasmEmitterOptions(
                format=format, target_sdk=target_sdk, **kwargs
            )
        emitter = QasmEmitter(emitter_options)
        return emitter.emit(self)

    def draw_circuit_diagram(self, max_length: int = 20) -> None:
        """
        Print a simple ASCII diagram of the circuit.

        The diagram is split into vertical blocks if any line exceeds
        ``max_length`` gate-units.

        Args:
            max_length: Maximum number of gate-units per block.
        """
        UNIT_WIDTH = 7  # characters per gate "cell"
        max_line_length = UNIT_WIDTH * max_length
        diagram_lines = ["" for _ in range(2 * self.number_of_qubits)]

        # Initialize qubit labels/rails
        for i in range(self.number_of_qubits):
            qubit_label = f"q{i}"
            diagram_lines[2 * i] = qubit_label + " --"
            diagram_lines[2 * i + 1] = " " * (len(qubit_label) + 3)

        # Gates
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
                            pad = current_max_length - len(diagram_lines[2 * j])
                            diagram_lines[2 * j] += "-" * pad
                            diagram_lines[2 * j + 1] += " " * pad

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

        # Measurements
        for measurement in self.measurements:
            if isinstance(measurement[0], int):
                qubit_index = measurement[0]
                m_str = "--" + "[M]" + "=="
                diagram_lines[2 * qubit_index] += m_str
                diagram_lines[2 * qubit_index + 1] += " " * len(m_str)
            else:
                for qubit_index in measurement[0]:
                    m_str = "--" + "[M]" + "=="
                    diagram_lines[2 * qubit_index] += m_str
                    diagram_lines[2 * qubit_index + 1] += " " * len(m_str)

        # Chunked printing
        max_width = max(len(line) for line in diagram_lines) if diagram_lines else 0
        for start in range(0, max_width, max_line_length):
            for line in diagram_lines:
                print(line[start : start + max_line_length])
            print()

    # -----------------------------
    # Unitary matrix generation
    # -----------------------------
    def to_unitary(self) -> np.ndarray:
        """
        Compute the overall unitary matrix for the circuit.

        Raises:
            ValueError:
                - if number_of_qubits > 10
                - if number of gates > 100
        """
        n = self.number_of_qubits
        if n < 0:
            raise ValueError("number_of_qubits must be non-negative")

        # --- hard safety limits ---
        if n > 10:
            raise ValueError(
                f"Cannot compute unitary: circuit has {n} qubits (>10). "
                "Matrix would be too large (2^n × 2^n)."
            )

        if len(self.gate_list) > 100:
            raise ValueError(
                f"Cannot compute unitary: circuit has {len(self.gate_list)} gates (>100). "
                "Computation would be too slow."
            )

        dim = 1 << n  # 2**n

        if n == 0:
            return np.array([[1.0 + 0.0j]], dtype=complex)

        U = np.zeros((dim, dim), dtype=complex)

        for basis_index in range(dim):
            state = np.zeros(dim, dtype=complex)
            state[basis_index] = 1.0 + 0.0j

            for gate in self.gate_list:
                state = self._apply_gate_to_state(state, gate, n)

            U[:, basis_index] = state

        return U

    def _apply_gate_to_state(
        self, state: np.ndarray, gate: "QuantumGate", n_qubits: int
    ) -> np.ndarray:
        """
        Apply a single gate to a statevector in-place-like (returns the same array).

        Args:
            state: 1D numpy array of length 2**n_qubits.
            gate:  QuantumGate or TwoQubitQuantumGate.
            n_qubits: number of qubits.

        Returns:
            The updated state (same ndarray object).
        """
        if not gate.is_two_qubit_gate:
            # single-qubit gate (may act on multiple targets, apply independently)
            U = self._single_qubit_matrix(gate.name, gate.parameters)
            for q in gate.target_qubits:
                state = self._apply_1q_matrix(state, U, q, n_qubits)
        else:
            # currently we only support one control and one target per gate
            if len(gate.control_qubits) != 1 or len(gate.target_qubits) != 1:
                raise NotImplementedError(
                    "Unitary builder currently supports exactly one control and "
                    "one target qubit per two-qubit gate."
                )
            U2 = self._two_qubit_matrix(gate.name, gate.parameters)
            c = gate.control_qubits[0]
            t = gate.target_qubits[0]
            state = self._apply_2q_matrix(state, U2, c, t, n_qubits)

        return state

    @staticmethod
    def _single_qubit_matrix(
        name: str, params: Optional[Sequence[float]]
    ) -> np.ndarray:
        """Return the 2x2 matrix for a single-qubit gate."""
        name = name.lower()
        p = list(params or [])

        # Common fixed gates
        if name == "x":
            return np.array([[0, 1], [1, 0]], dtype=complex)
        if name == "y":
            return np.array([[0, -1j], [1j, 0]], dtype=complex)
        if name == "z":
            return np.array([[1, 0], [0, -1]], dtype=complex)
        if name == "h":
            return (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)
        if name == "s":
            return np.array([[1, 0], [0, 1j]], dtype=complex)
        if name == "sdg":
            return np.array([[1, 0], [0, -1j]], dtype=complex)
        if name == "t":
            return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

        # Phase gate p(λ) ~ Rz(λ) up to global phase
        if name == "p":
            if len(p) != 1:
                raise ValueError("Phase gate 'p' expects one parameter.")
            lam = p[0]
            return np.array(
                [[1, 0], [0, np.exp(1j * lam)]],
                dtype=complex,
            )

        # General U(theta, phi, lambda) (QASM / U3 convention)
        if name == "u":
            if len(p) != 3:
                raise ValueError("Gate 'u' expects three parameters.")
            theta, phi, lam = p
            ct = np.cos(theta / 2.0)
            st = np.sin(theta / 2.0)
            return np.array(
                [
                    [ct, -np.exp(1j * lam) * st],
                    [np.exp(1j * phi) * st, np.exp(1j * (phi + lam)) * ct],
                ],
                dtype=complex,
            )

        # Rotations
        if name == "rx":
            if len(p) != 1:
                raise ValueError("Gate 'rx' expects one parameter.")
            theta = p[0]
            ct = np.cos(theta / 2.0)
            st = np.sin(theta / 2.0)
            return np.array([[ct, -1j * st], [-1j * st, ct]], dtype=complex)

        if name == "ry":
            if len(p) != 1:
                raise ValueError("Gate 'ry' expects one parameter.")
            theta = p[0]
            ct = np.cos(theta / 2.0)
            st = np.sin(theta / 2.0)
            return np.array([[ct, -st], [st, ct]], dtype=complex)

        if name == "rz":
            if len(p) != 1:
                raise ValueError("Gate 'rz' expects one parameter.")
            theta = p[0]
            return np.array(
                [
                    [np.exp(-1j * theta / 2.0), 0],
                    [0, np.exp(1j * theta / 2.0)],
                ],
                dtype=complex,
            )

        raise NotImplementedError(f"Unsupported single-qubit gate: {name!r}")

    @staticmethod
    def _two_qubit_matrix(name: str, params: Optional[Sequence[float]]) -> np.ndarray:
        """Return the 4x4 matrix for a two-qubit gate in basis |c t>."""
        _ = params  # currently unused
        name = name.lower()

        if name == "cx":
            # CNOT with control = |1>, flips target
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=complex,
            )

        if name == "cy":
            # control=1, apply Y to target
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, -1j],
                    [0, 0, 1j, 0],
                ],
                dtype=complex,
            )

        if name == "cz":
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, -1],
                ],
                dtype=complex,
            )

        if name == "swap":
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=complex,
            )

        raise NotImplementedError(f"Unsupported two-qubit gate: {name!r}")

    @staticmethod
    def _apply_1q_matrix(
        state: np.ndarray, U: np.ndarray, qubit: int, n_qubits: int
    ) -> np.ndarray:
        """Apply a 2x2 matrix U to `qubit` on the full statevector."""
        dim = state.shape[0]
        if dim != (1 << n_qubits):
            raise ValueError("State dimension does not match number of qubits.")

        mask = 1 << qubit
        for i in range(dim):
            if (i & mask) == 0:
                j = i | mask
                a0 = state[i]
                a1 = state[j]
                state[i] = U[0, 0] * a0 + U[0, 1] * a1
                state[j] = U[1, 0] * a0 + U[1, 1] * a1
        return state

    @staticmethod
    def _apply_2q_matrix(
        state: np.ndarray,
        U: np.ndarray,
        control: int,
        target: int,
        n_qubits: int,
    ) -> np.ndarray:
        """
        Apply a 4x4 matrix U acting on (control, target) in that order.

        Basis ordering for the 4x4 block is:
            |c t> ∈ {|00>, |01>, |10>, |11>}
        """
        dim = state.shape[0]
        if dim != (1 << n_qubits):
            raise ValueError("State dimension does not match number of qubits.")

        mc = 1 << control
        mt = 1 << target

        for i in range(dim):
            # choose only the base indices where both bits are 0
            if (i & mc) == 0 and (i & mt) == 0:
                i00 = i
                i01 = i | mt
                i10 = i | mc
                i11 = i | mc | mt

                v0 = state[i00]
                v1 = state[i01]
                v2 = state[i10]
                v3 = state[i11]

                # U @ [v0, v1, v2, v3]
                new0 = U[0, 0] * v0 + U[0, 1] * v1 + U[0, 2] * v2 + U[0, 3] * v3
                new1 = U[1, 0] * v0 + U[1, 1] * v1 + U[1, 2] * v2 + U[1, 3] * v3
                new2 = U[2, 0] * v0 + U[2, 1] * v1 + U[2, 2] * v2 + U[2, 3] * v3
                new3 = U[3, 0] * v0 + U[3, 1] * v1 + U[3, 2] * v2 + U[3, 3] * v3

                state[i00] = new0
                state[i01] = new1
                state[i10] = new2
                state[i11] = new3

        return state


# =============================================================
# QASM Emitter
# =============================================================


class QasmEmitter:
    """
    Translate a :class:`QuantumCircuit` into OpenQASM source code.

    Responsibilities:
      • Header emission (version + includes)
      • Register declarations
      • Gate translation & parameter formatting
      • Measurements and resets
    """

    def __init__(self, options: QasmEmitterOptions):
        """Initialise the emitter with the given :class:`QasmEmitterOptions`."""
        self.options = options

    def emit(self, circuit: QuantumCircuit) -> str:
        """
        Generate a full QASM program for ``circuit``.

        Returns:
            The QASM source code as a single string.
        """
        lines: List[str] = []

        format_lower = self.options.format.lower()
        includes_list = self.options.includes

        if format_lower == "qasm2":
            lines.append("OPENQASM 2.0;")
            qubit_decl = f"qreg q[{circuit.number_of_qubits}];"
            classical_decl = f"creg c[{circuit.number_of_classical_bits}];"
        elif format_lower == "qasm3":
            lines.append("OPENQASM 3.0;")
            qubit_decl = f"qubit[{circuit.number_of_qubits}] q;"
            classical_decl = f"bit[{circuit.number_of_classical_bits}] c;"
        else:
            raise ValueError(f"Unsupported QASM format: {self.options.format!r}")

        lines.append("")  # readability

        if includes_list:
            for inc in includes_list:
                lines.append(f'include "{inc}";')
        lines.append("")

        lines.extend([qubit_decl, classical_decl, ""])  # declarations + spacer

        for gate in circuit.gate_list:
            lines.extend(self._emit_gate(gate))

        for qubit in circuit.reset_operations:
            lines.append(f"reset q[{qubit}];")

        lines.append("")

        for qubit, classical_bit in circuit.measurements:
            measure_name = self.options.get_qasm_name("measure")
            if format_lower == "qasm2":
                lines.append(f"measure q[{qubit}] -> c[{classical_bit}];")
            else:  # qasm3
                lines.append(f"c[{classical_bit}] = {measure_name} q[{qubit}];")

        return "\n".join(lines)

    def _emit_gate(self, gate: QuantumGate) -> List[str]:
        """Emit QASM instruction(s) for a single ``gate``."""
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
        n_targets = len(tgt_qubits)

        for i in range(n_targets):
            lines.append(
                f"{qasm_name}{params_str} q[{ctrl_qubits[i]}], q[{tgt_qubits[i]}];"
            )

        return lines
