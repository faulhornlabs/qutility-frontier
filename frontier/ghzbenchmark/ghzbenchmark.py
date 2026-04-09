from __future__ import annotations

import copy
from math import isfinite
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.quantumbenchmark import Benchmark
from ..utils.quantumcircuit import QuantumCircuit


class GHZBenchmark(Benchmark):
    """Multipartite GHZ-state benchmark.

    Two measurement protocols are supported:

    * ``"witness"`` – prepares an n-qubit GHZ state and measures
      computational-basis (Z) and global-parity (X) witness circuits to
      estimate the stabilizer-based fidelity lower bound.
    * ``"shadow"`` – implements the r=2 shadow-overlap estimator using
      random two-qubit Pauli measurements after GHZ preparation.
    """

    BENCHMARK_NAME: str = "ghz"

    def __init__(
        self,
        number_of_qubits: int,
        sample_size: int = 1,
        *,
        preparation_method: str = "star",
        measurement_scheme: str = "witness",
        measurement_rounds: Optional[int] = None,
        certification_confidence: float = 0.95,
        **kwargs: Any,
    ) -> None:
        """Initialize a :class:`GHZBenchmark`.

        Args:
            number_of_qubits: Number of qubits of the GHZ state.
            sample_size: Number of independent samples. Each sample
                recreates the GHZ preparation and requested measurements.
            preparation_method: Entanglement fanout topology. One of
                ``"linear"``, ``"log"``, or ``"star"``.
            measurement_scheme: ``"witness"`` (default) or ``"shadow"``.
            measurement_rounds: Number of shadow-overlap rounds per sample.
                Only used when ``measurement_scheme == "shadow"``. Defaults
                to ``max(10, 2 * number_of_qubits)`` if omitted.
            certification_confidence: Confidence level for the
                ``fidelity > 0.5`` certification check (witness mode).
            **kwargs: Forwarded to :class:`Benchmark`.
        """
        self.preparation_method = preparation_method.lower()
        if self.preparation_method not in {"linear", "log", "star"}:
            raise ValueError(
                "preparation_method must be one of 'linear', 'log', or 'star'."
            )

        self.measurement_scheme = measurement_scheme.lower()
        if self.measurement_scheme not in {"witness", "shadow"}:
            raise ValueError("measurement_scheme must be 'witness' or 'shadow'.")

        if self.measurement_scheme == "shadow":
            if measurement_rounds is None:
                measurement_rounds = max(10, 2 * number_of_qubits)
            if measurement_rounds <= 0:
                raise ValueError("measurement_rounds must be a positive integer.")
            self.measurement_rounds = int(measurement_rounds)
        else:
            self.measurement_rounds = 2  # placeholder, not used outside witness

        if not (0.0 < certification_confidence < 1.0):
            raise ValueError("certification_confidence must be between 0 and 1.")
        self.certification_confidence = certification_confidence

        super().__init__(
            number_of_qubits=number_of_qubits,
            sample_size=sample_size,
            **kwargs,
        )

        self.number_of_measurements: int = self._compute_number_of_measurements()

    # ------------------------------------------------------------------
    # Benchmark hooks
    # ------------------------------------------------------------------
    def _compute_number_of_measurements(self) -> int:
        if self.measurement_scheme == "witness":
            return 2
        return self.measurement_rounds

    def _create_single_sample(self, sample_id: int) -> Dict[str, Any]:
        if self.measurement_scheme == "witness":
            circuits = self._create_witness_circuits(sample_id)
        else:
            circuits = self._create_shadow_circuits(sample_id)

        sample_metadata: Dict[str, Any] = {
            "type": "ghz",
            "preparation_method": self.preparation_method,
            "measurement_scheme": self.measurement_scheme,
            "number_of_measurements": self.number_of_measurements,
        }

        if self.measurement_scheme == "shadow":
            sample_metadata["measurement_rounds"] = self.measurement_rounds

        return {
            "sample_id": sample_id,
            "sample_metadata": sample_metadata,
            "circuits": circuits,
        }

    # ------------------------------------------------------------------
    # Circuit generation helpers
    # ------------------------------------------------------------------
    def _base_ghz_circuit(self) -> QuantumCircuit:
        """Create the GHZ preparation circuit for the configured topology."""
        qc = QuantumCircuit(
            number_of_qubits=self.number_of_qubits,
            number_of_classical_bits=self.number_of_qubits,
        )
        qc.add_h_gate(0)

        if self.preparation_method == "linear":
            for i in range(self.number_of_qubits - 1):
                qc.add_cx_gate(i, i + 1)
        elif self.preparation_method == "log":
            num_levels = int(np.ceil(np.log2(self.number_of_qubits)))
            for level in range(num_levels, 0, -1):
                step = 2**level
                offset = 2 ** (level - 1)
                for i in range(0, self.number_of_qubits, step):
                    if i + offset < self.number_of_qubits:
                        qc.add_cx_gate(i, i + offset)
        else:  # star
            for i in range(1, self.number_of_qubits):
                qc.add_cx_gate(0, i)

        return qc

    def _create_witness_circuits(self, sample_id: int) -> List[Dict[str, Any]]:
        base_circuit = self._base_ghz_circuit()

        circ_z = copy.deepcopy(base_circuit)
        circ_x = copy.deepcopy(base_circuit)

        for q in range(self.number_of_qubits):
            circ_x.add_h_gate(q)

        for q in range(self.number_of_qubits):
            circ_x.add_measurement(q, q)
            circ_z.add_measurement(q, q)

        opts = self.emitter_options
        circuits: List[Dict[str, Any]] = [
            {
                "circuit_id": f"{sample_id}_witness_x",
                "observable": None,
                "qasm": circ_x.to_qasm(opts),
                "metadata": {
                    "measurement_scheme": "witness",
                    "basis": "x",
                },
            },
            {
                "circuit_id": f"{sample_id}_witness_z",
                "observable": None,
                "qasm": circ_z.to_qasm(opts),
                "metadata": {
                    "measurement_scheme": "witness",
                    "basis": "z",
                },
            },
        ]
        return circuits

    def _create_shadow_circuits(self, sample_id: int) -> List[Dict[str, Any]]:
        if self.number_of_qubits < 2:
            raise ValueError("Shadow-overlap scheme requires at least 2 qubits.")

        circuits: List[Dict[str, Any]] = []
        base_circuit = self._base_ghz_circuit()

        for round_idx in range(self.measurement_rounds):
            qc = copy.deepcopy(base_circuit)
            for q in range(self.number_of_qubits):
                qc.add_h_gate(q)

            selected = np.random.choice(
                range(self.number_of_qubits), size=2, replace=False
            )

            basis_map: Dict[int, str] = {}
            for q in selected:
                basis_choice = int(np.random.randint(0, 3))
                if basis_choice == 0:
                    qc.add_h_gate(int(q))
                    basis_map[int(q)] = "X"
                elif basis_choice == 1:
                    qc.add_sdg_gate(int(q))
                    qc.add_h_gate(int(q))
                    basis_map[int(q)] = "Y"
                else:
                    basis_map[int(q)] = "Z"

            for q in range(self.number_of_qubits):
                qc.add_measurement(q, q)

            circuits.append(
                {
                    "circuit_id": f"{sample_id}_shadow_{round_idx}",
                    "observable": None,
                    "qasm": qc.to_qasm(self.emitter_options),
                    "metadata": {
                        "measurement_scheme": "shadow",
                        "round": round_idx,
                        "basis_map": basis_map,
                    },
                }
            )

        return circuits

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_benchmark(
        self,
        *,
        auto_save: Optional[bool] = None,
        save_to: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        if self.experimental_results is None:
            raise ValueError(
                "No experimental_results attached. "
                "Call add_experimental_results(...) first."
            )

        if self.samples is None:
            raise ValueError(
                "Benchmark has no samples. Generate or load the benchmark first."
            )

        results = self.experimental_results.get("results")
        if results is None:
            raise ValueError("experimental_results has no 'results' entry.")

        if auto_save is not None:
            self.auto_save = bool(auto_save)

        if self.measurement_scheme == "witness":
            evaluation = self._evaluate_witness(results)
        else:
            evaluation = self._evaluate_shadow(results)

        self.experimental_results.setdefault("evaluation", {})
        self.experimental_results["evaluation"].update(evaluation)

        if self.auto_save:
            if save_to is not None:
                saved_path = self.save_json(filepath=save_to)
            elif self.path is not None:
                saved_path = self.save_json(filepath=self.path)
            else:
                saved_path = self.save_json()
            print(f"[Benchmark] Saved updated JSON to: {saved_path}")

        return evaluation

    def _evaluate_witness(self, results: Dict[str, Any]) -> Dict[str, Any]:
        per_sample: Dict[int, Dict[str, Any]] = {}
        fidelities: List[float] = []
        stds: List[float] = []

        for sample in self.samples or []:
            sid = sample["sample_id"]
            basis_to_circuit: Dict[str, Dict[str, Any]] = {}
            for circuit in sample.get("circuits", []):
                metadata = circuit.get("metadata", {})
                if metadata.get("basis") in {"x", "z"}:
                    basis_to_circuit[metadata["basis"]] = circuit

            if "x" not in basis_to_circuit or "z" not in basis_to_circuit:
                raise ValueError(
                    f"Sample {sid} is missing witness circuits (found bases: "
                    f"{list(basis_to_circuit.keys())})."
                )

            cid_x = basis_to_circuit["x"]["circuit_id"]
            cid_z = basis_to_circuit["z"]["circuit_id"]

            if cid_x not in results or cid_z not in results:
                raise ValueError(
                    f"Missing experimental results for witness circuits of sample {sid}."
                )

            counts_x = results[cid_x].get("counts")
            counts_z = results[cid_z].get("counts")
            if counts_x is None or counts_z is None:
                raise ValueError(
                    f"Witness results for sample {sid} must provide 'counts'."
                )

            fidelity, std = evaluate_fidelity(counts_z, counts_x)
            certificate = certify_fidelity_gt_half(
                fidelity, std, confidence=self.certification_confidence
            )

            results[cid_x]["fidelity_lower_bound"] = fidelity
            results[cid_x]["std_dev"] = std
            results[cid_z]["fidelity_lower_bound"] = fidelity
            results[cid_z]["std_dev"] = std

            per_sample[sid] = {
                "fidelity_lower_bound": fidelity,
                "std_dev": std,
                "certified_gt_half": certificate,
                "confidence": self.certification_confidence,
            }

            fidelities.append(fidelity)
            stds.append(std)

        global_metrics = {
            "mean_fidelity": float(np.mean(fidelities)) if fidelities else None,
            "min_fidelity": float(np.min(fidelities)) if fidelities else None,
            "max_fidelity": float(np.max(fidelities)) if fidelities else None,
            "mean_std_dev": float(np.mean(stds)) if stds else None,
        }

        return {
            "method": "witness",
            "per_sample": per_sample,
            "global_metrics": global_metrics,
        }

    def _evaluate_shadow(self, results: Dict[str, Any]) -> Dict[str, Any]:
        per_sample: Dict[int, Dict[str, Any]] = {}
        means: List[float] = []
        sems: List[float] = []

        for sample in self.samples or []:
            sid = sample["sample_id"]
            measurement_outcomes: List[str] = []
            basis_sequence: List[Dict[int, str]] = []

            for circuit in sample.get("circuits", []):
                cid = circuit["circuit_id"]
                entry = results.get(cid)
                if entry is None:
                    raise ValueError(
                        f"Missing experimental result for circuit_id {cid!r}."
                    )

                basis_map = circuit.get("metadata", {}).get("basis_map")
                if basis_map is None or len(basis_map) != 2:
                    raise ValueError(
                        f"Shadow circuit {cid!r} must include a two-qubit basis_map."
                    )

                shots = entry.get("shots")
                if shots is None:
                    counts = entry.get("counts")
                    if counts is None:
                        raise ValueError(
                            f"Shadow results for circuit {cid!r} require 'shots' "
                            "or 'counts'."
                        )
                    shots = []
                    for bitstring, count in counts.items():
                        shots.extend([bitstring] * int(count))

                if len(shots) == 0:
                    raise ValueError(
                        f"Shadow results for circuit {cid!r} are empty (no shots)."
                    )

                for bitstring in shots:
                    if len(bitstring) != self.number_of_qubits:
                        raise ValueError(
                            f"Shot '{bitstring}' for circuit {cid!r} has length "
                            f"{len(bitstring)} (expected {self.number_of_qubits})."
                        )
                    measurement_outcomes.append(bitstring)
                    basis_sequence.append({int(k): v for k, v in basis_map.items()})

            mean, sem = evaluate_shadow_overlap(measurement_outcomes, basis_sequence)
            per_sample[sid] = {
                "shadow_overlap_mean": mean,
                "shadow_overlap_sem": sem,
            }

            means.append(mean)
            sems.append(sem)

        global_metrics = {
            "mean_shadow_overlap": float(np.mean(means)) if means else None,
            "min_shadow_overlap": float(np.min(means)) if means else None,
            "max_shadow_overlap": float(np.max(means)) if means else None,
            "mean_sem": float(np.mean(sems)) if sems else None,
        }

        return {
            "method": "shadow",
            "per_sample": per_sample,
            "global_metrics": global_metrics,
        }


# ----------------------------------------------------------------------
# Fidelity witness helpers
# ----------------------------------------------------------------------
def evaluate_fidelity(
    z_basis_counts: Dict[str, int], x_basis_counts: Dict[str, int]
) -> Tuple[float, float]:
    if not z_basis_counts or not x_basis_counts:
        raise ValueError("z_basis_counts and x_basis_counts must be non-empty.")

    n = len(next(iter(z_basis_counts)))
    total_z = sum(z_basis_counts.values())
    total_x = sum(x_basis_counts.values())

    mu_x_num = 0.0
    for bits, freq in x_basis_counts.items():
        sign = 1.0 if (bits.count("1") % 2 == 0) else -1.0
        mu_x_num += sign * freq
    mu_x = mu_x_num / total_x

    mu_z = []
    for k in range(n - 1):
        num = 0.0
        for bits, freq in z_basis_counts.items():
            sign = 1.0 if bits[k] == bits[k + 1] else -1.0
            num += sign * freq
        mu_z.append(num / total_z)

    mus = [mu_x] + mu_z
    fidelity = 1.0 - 0.5 * sum(1.0 - mu for mu in mus)
    fidelity = max(0.0, float(fidelity))

    var_mu_x = (1.0 - mu_x**2) / total_x
    var_mu_z = sum((1.0 - mu**2) / total_z for mu in mu_z)
    var_f = 0.25 * (var_mu_x + var_mu_z)
    std = float(np.sqrt(max(0.0, var_f)))

    return fidelity, std


def certify_fidelity_gt_half(
    f_min_hat: float,
    std: float,
    confidence: float = 0.95,
) -> bool:
    if not (isfinite(f_min_hat) and isfinite(std)) or std < 0:
        raise ValueError("Invalid fidelity estimate or standard deviation.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1.")

    z = NormalDist().inv_cdf(confidence)
    return (f_min_hat - z * std) > 0.5


# ----------------------------------------------------------------------
# Shadow-overlap helpers
# ----------------------------------------------------------------------
def compute_ghz_hadamard_amplitude(bitstring: str) -> float | complex:
    n = len(bitstring)
    hw = bitstring.count("1")
    return 1 / np.sqrt(2**n) if hw % 2 == 0 else 0.0


def replace_character_at_index(string: str, index: int, new_character: str) -> str:
    if index < 0 or index >= len(string):
        raise IndexError("index out of range")
    s_list = list(string)
    s_list[index] = new_character
    return "".join(s_list)


def bitstring_to_basis_vector(bitstring: str) -> np.ndarray:
    idx = int(bitstring, 2)
    vec = np.zeros((2 ** len(bitstring), 1))
    vec[idx] = 1
    return vec


def compute_outer_product(bitstring_1: str, bitstring_2: str) -> np.ndarray:
    v1 = bitstring_to_basis_vector(bitstring_1)
    v2 = bitstring_to_basis_vector(bitstring_2)
    return np.dot(v1, v2.T)


def compute_l_matrix(measured_bitstring: str, qubit_1: int, qubit_2: int) -> np.ndarray:
    pairs = [("00", "11"), ("01", "10")]
    l_mat = np.zeros((4, 4), dtype=np.double)

    for a, b in pairs:
        bs1 = replace_character_at_index(measured_bitstring, qubit_1, a[0])
        bs1 = replace_character_at_index(bs1, qubit_2, a[1])

        bs2 = replace_character_at_index(measured_bitstring, qubit_1, b[0])
        bs2 = replace_character_at_index(bs2, qubit_2, b[1])

        amp1 = compute_ghz_hadamard_amplitude(bs1)
        amp2 = compute_ghz_hadamard_amplitude(bs2)
        norm = np.abs(amp1) ** 2 + np.abs(amp2) ** 2

        if norm > 0:
            rho = (
                (amp1**2) * compute_outer_product(a, a)
                + (amp1 * amp2) * compute_outer_product(a, b)
                + (amp2 * amp1) * compute_outer_product(b, a)
                + (amp2**2) * compute_outer_product(b, b)
            ) / norm
            l_mat += rho

    return l_mat


def construct_density_matrix(
    measurement_outcome: str, measurement_basis: str
) -> np.ndarray:
    if measurement_basis == "Z":
        state = (
            np.array([[1], [0]]) if measurement_outcome == "0" else np.array([[0], [1]])
        )
    elif measurement_basis == "X":
        state = (1 / np.sqrt(2)) * (
            np.array([[1], [1]])
            if measurement_outcome == "0"
            else np.array([[1], [-1]])
        )
    elif measurement_basis == "Y":
        state = (1 / np.sqrt(2)) * (
            np.array([[1], [1j]])
            if measurement_outcome == "0"
            else np.array([[1], [-1j]])
        )
    else:
        raise ValueError("measurement_basis must be one of 'Z', 'X', 'Y'.")
    return state @ state.conj().T


def compute_sigma_matrix(
    outcome_1: str, outcome_2: str, basis_1: str, basis_2: str
) -> np.ndarray:
    return np.kron(
        3 * construct_density_matrix(outcome_1, basis_1) - np.eye(2),
        3 * construct_density_matrix(outcome_2, basis_2) - np.eye(2),
    )


def compute_shadow_overlap_fidelity(
    measured_bitstring: str, qubit_1: int, qubit_2: int, basis_1: str, basis_2: str
) -> float:
    l_mat = compute_l_matrix(measured_bitstring, qubit_1, qubit_2)
    sigma = compute_sigma_matrix(
        measured_bitstring[qubit_1], measured_bitstring[qubit_2], basis_1, basis_2
    )
    fidelity_estimate = np.linalg.trace(l_mat @ sigma)
    return float(np.real(fidelity_estimate))


def evaluate_shadow_overlap(
    measurement_outcomes: List[str], random_qubit_pairs: List[Dict[int, str]]
) -> Tuple[float, float]:
    if len(measurement_outcomes) != len(random_qubit_pairs):
        raise ValueError(
            "measurement_outcomes and random_qubit_pairs must have the same length."
        )

    values: List[float] = []
    for bits, basis_map in zip(measurement_outcomes, random_qubit_pairs):
        qubit_indices = list(basis_map.keys())
        if len(qubit_indices) != 2:
            raise ValueError("Each basis_map must specify exactly two qubits.")
        q1, q2 = [int(q) for q in qubit_indices]
        b1, b2 = basis_map[q1], basis_map[q2]
        values.append(compute_shadow_overlap_fidelity(bits, q1, q2, b1, b2))

    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sem = (
        float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        if len(arr) > 1
        else float(np.std(arr))
    )
    return mean, sem
