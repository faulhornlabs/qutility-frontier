from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import special_ortho_group

from ..utils.quantumbenchmark import Benchmark
from ..utils.quantumcircuit import QuantumCircuit
from ..utils.so_decomposition import so_decomposition


class FreeFermionVolumeBenchmark(Benchmark):
    """Free-fermion volumetric benchmark based on a random SO(2N) Gaussian unitary.

    For each sample:

    * Draw a random O ∈ SO(2N) (N = number_of_qubits).
    * Decompose it into Givens rotations + diagonal ±1 using
    ``so_decomposition``.
    * Build a free-fermion circuit from these rotations.
    * Compute a Pauli correction M from the diagonal ±1 data.
    * Construct measurement circuits for each Majorana operator.
    * Export each measurement circuit as QASM + Pauli observable string.

    The per-sample dictionary matches the generic benchmark schema::

        {
            "sample_id": int,
            "sample_metadata": {...},
            "circuits": [
                {
                    "circuit_id": str,
                    "observable": str,
                    "qasm": str,
                    "metadata": {...}
                },
                ...
            ]
        }
    """

    BENCHMARK_NAME: str = "free_fermion"

    def __init__(
        self,
        number_of_qubits: int,
        sample_size: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            number_of_qubits=number_of_qubits,
            sample_size=sample_size,
            **kwargs,
        )
        self.number_of_measurements: int = self._compute_number_of_measurements()

    # Measurement count rule

    def _compute_number_of_measurements(self) -> int:
        """Compute number of distinct measurement schemes.

        Mirrors your original helper:

          * if n <= 10: return 2n
          * else:      return 20 + floor(n/5)
        """
        n = self.number_of_qubits
        if n <= 10:
            return 2 * n
        return int(20 + np.floor(n / 5))

    # Benchmark hook: one sample
    def _create_single_sample(self, sample_id: int) -> Dict[str, Any]:
        n_qubits = self.number_of_qubits

        # 1) Draw random O ∈ SO(2N) and random state preparation i ∈ {1, ..., 2N}
        orthogonal_matrix = special_ortho_group.rvs(2 * n_qubits)
        state_index = np.random.randint(0, 2 * self.number_of_qubits)

        # 2) SO decomposition → list of rotations + diagonal ±1
        givens_rotations, D = so_decomposition(orthogonal_matrix)
        S = np.diag(np.round(D, 2))  # ±1 entries

        # 3) Build the free-fermion circuit:
        #    state preparation + Gaussian unitary
        base_qc = self._build_free_fermion_circuit(givens_rotations, state_index)

        # 4) Compute Pauli correction M from diagonal entries
        pauli_M = self._compute_pauli_from_diagonal(S)

        # 5) Apply Pauli corrections
        corrected_qc = self._apply_pauli_corrections(base_qc, pauli_M)

        # 6) Measurement circuits for each chosen Majorana operator
        observables, circuits = self._generate_majorana_measurement_circuits(
            corrected_qc, orthogonal_matrix, state_index
        )

        # 7) Serialize circuits to schema-compatible objects
        circuit_payloads: List[Dict[str, Any]] = []
        for idx, (obs, qc) in enumerate(zip(observables, circuits)):
            circuit_payloads.append(
                {
                    "circuit_id": f"{sample_id}_maj_{idx}",
                    "observable": obs,  # e.g. "ZXYII"
                    "qasm": qc.to_qasm(self.emitter_options),
                    "metadata": {
                        "kind": "majorana",
                        "majorana_index": idx,
                    },
                }
            )

        sample_metadata: Dict[str, Any] = {
            "type": "free_fermion",
            "orthogonal_matrix": orthogonal_matrix,  # Benchmark._jsonify_meta handles ndarray
            "initial_state_index": state_index,
            "number_of_measurements": self.number_of_measurements,
        }

        return {
            "sample_id": sample_id,
            "sample_metadata": sample_metadata,
            "circuits": circuit_payloads,
        }

    # Internal helpers
    def _compute_measurement_indices(
        self,
        orthogonal_matrix: np.ndarray,
        state_index: int = 0,
    ) -> List[int]:
        """
        Return the indices of the Majorana modes most strongly contributing
        to column `state_index` of the orthogonal matrix O.

        The returned list contains `number_of_measurements` indices,
        sorted from largest to smallest absolute coefficient.
        """
        column = np.abs(orthogonal_matrix[:, state_index])
        sorted_indices = np.argsort(column)[::-1]  # descending order

        return sorted_indices[: self.number_of_measurements].tolist()

    def _build_free_fermion_circuit(
        self, givens_rotations, state_index
    ) -> QuantumCircuit:
        """
        State preparation + Gaussian unitary from Givens rotations.
        """

        qc = QuantumCircuit(
            number_of_qubits=self.number_of_qubits,
            number_of_classical_bits=self.number_of_qubits,
        )

        if state_index % 2 == 0:
            # corresponds to old *odd* i → X-type preparation
            q = state_index // 2
            qc.add_h_gate(q)
        else:
            # corresponds to old *even* i → Y-type preparation
            q = (state_index - 1) // 2
            qc.add_h_gate(q)
            qc.add_s_gate(q)

        #  apply fermionic Gaussian unitary from Givens rotations
        for gr in givens_rotations:
            # Here the gr.n is the 0-based “mode index” as you said.
            n = gr.n
            theta = gr.theta

            if n % 2 == 0:
                # "odd fermionic mode" in 0-based convention → single-qubit exp(i θ Z)
                mode_l = n // 2
                qc.add_rz_gate(mode_l, theta)
            else:
                # "even fermionic mode" in old 0-based convention → two-qubit exp(i θ X_l X_{l+1})
                mode_l = (n - 1) // 2

                qc.add_h_gate(mode_l)
                qc.add_h_gate(mode_l + 1)

                qc.add_cx_gate(mode_l, mode_l + 1)
                qc.add_rz_gate(mode_l + 1, theta)
                qc.add_cx_gate(mode_l, mode_l + 1)

                qc.add_h_gate(mode_l)
                qc.add_h_gate(mode_l + 1)

        return qc

    def _compute_pauli_from_diagonal(self, S: np.ndarray) -> str:
        """Pauli-string construction from the ±1 diagonal."""
        n_qubits = self.number_of_qubits
        M = "I" * n_qubits

        for i, s in enumerate(S):
            if s != -1:
                continue

            if i % 2 == 0:
                # i even → (old code: even index → X at position k)
                k = i // 2
                m = "Z" * k + "X" + "I" * (n_qubits - k - 1)
            else:
                k = (i + 1) // 2 - 1
                m = "Z" * k + "Y" + "I" * (n_qubits - k - 1)

            M, _ = self._pauli_product(M, m)

        return M

    def _apply_pauli_corrections(
        self,
        qc: QuantumCircuit,
        M: str,
    ) -> QuantumCircuit:
        qc = deepcopy(qc)
        for idx, p in enumerate(M):
            if p == "X":
                qc.add_x_gate(idx)
            elif p == "Y":
                qc.add_y_gate(idx)
            elif p == "Z":
                qc.add_z_gate(idx)
        return qc

    def _generate_majorana_measurement_circuits(
        self,
        base_qc: QuantumCircuit,
        orthogonal_matrix: np.ndarray,
        state_index: int,
    ) -> Tuple[List[str], List[QuantumCircuit]]:
        """
        Generate one measurement circuit per Majorana operator.
        """

        n_qubits = self.number_of_qubits
        circuits = []
        observables = []

        indices = self._compute_measurement_indices(orthogonal_matrix, state_index)

        for j in indices:
            qc_m = deepcopy(base_qc)

            if j % 2 == 0:
                qj = j // 2
                qc_m.add_h_gate(qj)
                m = "Z" * qj + "X" + "I" * (n_qubits - qj - 1)

            else:
                qj = (j - 1) // 2
                qc_m.add_sdg_gate(qj)
                qc_m.add_h_gate(qj)
                m = "Z" * qj + "Y" + "I" * (n_qubits - qj - 1)

            for q in range(n_qubits):
                qc_m.add_measurement(q, q)

            circuits.append(qc_m)
            observables.append(m)

        return observables, circuits

    def _pauli_product(self, pauli1: str, pauli2: str) -> Tuple[str, complex]:
        """
        Computes the product of two Pauli strings.
        Returns the resulting Pauli string and the phase factor.
        """
        assert len(pauli1) == len(pauli2), "Pauli strings must be of the same length"

        # Multiplication rules for Pauli matrices
        pauli_mult = {
            ("I", "I"): ("I", 1),
            ("I", "X"): ("X", 1),
            ("I", "Y"): ("Y", 1),
            ("I", "Z"): ("Z", 1),
            ("X", "I"): ("X", 1),
            ("X", "X"): ("I", 1),
            ("X", "Y"): ("Z", 1j),
            ("X", "Z"): ("Y", -1j),
            ("Y", "I"): ("Y", 1),
            ("Y", "X"): ("Z", -1j),
            ("Y", "Y"): ("I", 1),
            ("Y", "Z"): ("X", 1j),
            ("Z", "I"): ("Z", 1),
            ("Z", "X"): ("Y", 1j),
            ("Z", "Y"): ("X", -1j),
            ("Z", "Z"): ("I", 1),
        }

        result = []
        phase: complex = 1

        for p1, p2 in zip(pauli1, pauli2):
            res, factor = pauli_mult[(p1, p2)]
            result.append(res)
            phase *= factor

        return "".join(result), phase

    def evaluate_benchmark(
        self,
        *,
        auto_save: Optional[bool] = None,
        save_to: Optional[Union[str, Path]] = None,
    ) -> Dict[str, List[float]]:
        """
        Evaluate the Free-Fermion benchmark using experimental results.

        For each sample, the expectation values from its measurement circuits
        are combined into two benchmark metrics:

          * parallel_value   = dot( O[state_index, indices], EVs )
          * orthogonal_value = dot( O[random_row, indices], EVs )

        where:
            - O is the SO(2N) orthogonal matrix for the sample,
            - state_index is the prepared Majorana index,
            - indices = self._compute_measurement_indices(O, state_index),
            - EVs is the vector of expectation values (one per circuit),
            - random_row is any j ≠ state_index.

        The first quantity should be close to 1 for well-performing hardware
        (or an ideal simulator), while the second should be close to 0.

        Each circuit’s expectation value and standard error are stored in-place
        inside `self.experimental_results["results"][circuit_id]`.

        Returns:
            dict[str, list[float]]: Dictionary with keys:

                - ``parallel_values`` — projected signal values
                - ``orthogonal_values`` — projected null values
        """

        # Validation — identical structure to Clifford benchmark

        if self.experimental_results is None:
            raise ValueError(
                "No experimental_results attached. "
                "Call add_experimental_results(...) first."
            )

        results = self.experimental_results.get("results")
        if results is None:
            raise ValueError("experimental_results has no 'results' entry.")

        if self.samples is None:
            raise ValueError(
                "Benchmark has no samples. Generate or load the benchmark first."
            )

        if not self.shots:
            raise ValueError("self.shots must be a positive integer.")

        if auto_save is not None:
            self.auto_save = bool(auto_save)

        # Aggregated benchmark output

        parallel_values: List[float] = []
        orthogonal_values: List[float] = []

        # Walk samples in canonical order — consistent with Clifford

        for sample in self.samples:
            meta = sample.get("sample_metadata", {})

            # Recover O and the prepared Majorana state index
            orthogonal_matrix = np.array(meta["orthogonal_matrix"], dtype=float)
            state_index = int(meta["initial_state_index"])

            # Determine which Majorana indices were actually measured
            indices = self._compute_measurement_indices(orthogonal_matrix, state_index)

            # Collect expectation values (one per circuit)
            evs: List[float] = []
            vars_: List[float] = []  # for optional uncertainty propagation later

            # Iterate through measurement circuits

            for circuit in sample.get("circuits", []):
                cid = circuit["circuit_id"]
                pauli = circuit.get("observable")

                if pauli is None:
                    continue  # purely structural circuit — ignore

                if cid not in results:
                    raise ValueError(
                        f"Missing result for circuit_id {cid!r} "
                        "in experimental_results['results']."
                    )

                entry = results[cid]
                counts = entry["counts"]

                # --- Expectation value (compute or reuse cached) ---
                if "expectation_value" in entry:
                    ev = float(entry["expectation_value"])
                else:
                    ev = float(self.expected_value(counts, pauli, little_endian=True))
                    entry["expectation_value"] = ev

                # --- Standard error for ±1-valued estimator ---
                if "std_error" in entry:
                    std = float(entry["std_error"])
                else:
                    var = max(0.0, 1.0 - ev * ev) / self.shots
                    std = float(np.sqrt(var))
                    entry["std_error"] = std

                evs.append(ev)
                vars_.append(std * std)

            evs = np.array(evs, dtype=float)
            vars_ = np.array(vars_, dtype=float)

            # Form reduced rows of O corresponding to measured modes

            signal_row = orthogonal_matrix[indices, state_index]

            # Choose a different orthogonal row for null test
            all_rows = list(range(orthogonal_matrix.shape[0]))
            all_rows.remove(state_index)
            random_row_index = np.random.choice(all_rows)
            null_row = orthogonal_matrix[indices, random_row_index]

            # Compute benchmark metrics

            parallel_value = float(np.dot(signal_row, evs))
            orthogonal_value = float(np.dot(null_row, evs))

            parallel_values.append(parallel_value)
            orthogonal_values.append(orthogonal_value)

        # Store grouped outputs — exactly like Clifford style

        self.experimental_results.setdefault("evaluation", {})
        self.experimental_results["evaluation"]["parallel_values"] = parallel_values
        self.experimental_results["evaluation"]["orthogonal_values"] = orthogonal_values

        # Pretty-print summary — analogous to Clifford report

        print("\n==============================================================")
        print(f" Free-Fermion Benchmark Evaluation ({self.number_of_qubits} qubits)")
        print("==============================================================\n")

        if len(parallel_values) > 0:
            mean_p, std_p = np.mean(parallel_values), np.std(parallel_values)
            print("Parallel projected values (should be near 1):")
            print(f"  • average: {mean_p:.6f} ± {std_p:.6f}")
            print(f"  • lowest measured value: {np.min(parallel_values):.6f}\n")
        else:
            print("No parallel values.\n")

        if len(orthogonal_values) > 0:
            mean_o, std_o = np.mean(orthogonal_values), np.std(orthogonal_values)
            print("Orthogonal projected values (should be near 0):")
            print(f"  • average: {mean_o:.6f} ± {std_o:.6f}")
            print(
                f"  • highest absolute value: {np.max(np.abs(orthogonal_values)):.6f}\n"
            )
        else:
            print("No orthogonal values.\n")

        # No binary pass/fail rule yet — user may define one later

        print("==============================================================\n")

        passed = (
            len(parallel_values) > 0
            and len(orthogonal_values) > 0
            and np.min(parallel_values) > 1 / np.e
            and np.max(np.abs(orthogonal_values)) < 0.25 / np.e
        )

        print(f"Benchmark passed: {passed}")
        print("==============================================================\n")
        # Optional JSON autosave — identical behavior to Clifford

        if self.auto_save:
            if save_to is not None:
                saved_path = self.save_json(filepath=save_to)
            elif self.path is not None:
                saved_path = self.save_json(filepath=self.path)
            else:
                saved_path = self.save_json()
            print(f"[Benchmark] Saved updated JSON to: {saved_path}")

        return {
            "parallel_values": parallel_values,
            "orthogonal_values": orthogonal_values,
        }

    def plot_all_expectation_values(self) -> None:
        """Plot parallel and orthogonal projection values across all samples.

        Plots projection values (with approximate standard error bars) across
        the entire benchmark, with separate markers for:

          * Parallel values      (stabilizer-like).
          * Orthogonal values    (destabilizer-like).

        Requires :meth:`evaluate_benchmark` to have been run so that
        ``self.experimental_results["evaluation"]`` is populated.

        Raises:
          ValueError: If experimental results or evaluation entries are
            missing, or if :attr:`shots` is not a positive integer.
        """
        if self.experimental_results is None:
            raise ValueError(
                "No experimental_results found. Run evaluate_benchmark() first."
            )

        if not self.shots:
            raise ValueError("self.shots must be a positive integer.")

        evaluation = self.experimental_results.get("evaluation", {})
        if "parallel_values" not in evaluation or "orthogonal_values" not in evaluation:
            raise ValueError(
                "Expected values missing — run evaluate_benchmark() first."
            )

        parallel_values = np.asarray(evaluation["parallel_values"], dtype=float)
        orthogonal_values = np.asarray(evaluation["orthogonal_values"], dtype=float)

        # Approximate std errors (using ±1 outcome formula)
        parallel_errors = [
            float(np.sqrt(max(0.0, 1.0 - ev * ev) / self.shots))
            for ev in parallel_values
        ]
        orthogonal_errors = [
            float(np.sqrt(max(0.0, 1.0 - ev * ev) / self.shots))
            for ev in orthogonal_values
        ]

        # x-axis over samples
        x = 1 + np.arange(len(parallel_values))

        plt.figure(figsize=(12, 6))

        # Parallel (blue)
        plt.errorbar(
            x,
            parallel_values,
            yerr=parallel_errors,
            fmt="o",
            markersize=6,
            capsize=4,
            color="tab:blue",
            ecolor="tab:blue",
            label="Parallel values",
        )

        # Orthogonal (red, plot absolute value like destabilizers)
        plt.errorbar(
            x,
            np.abs(orthogonal_values),
            yerr=orthogonal_errors,
            fmt="o",
            markersize=6,
            capsize=4,
            color="tab:red",
            ecolor="tab:red",
            label="Orthogonal values (|·|)",
        )

        # Optional thresholds, analogous to Clifford
        stab_thresh = 1 / np.e
        dest_thresh = 0.25 / np.e

        plt.axhline(
            stab_thresh,
            linestyle="--",
            linewidth=2,
            color="tab:blue",
            label="Parallel threshold = 1/e",
        )
        plt.axhline(
            dest_thresh,
            linestyle="--",
            linewidth=2,
            color="tab:red",
            label="Orthogonal threshold = 1/(4e)",
        )

        plt.xlabel("Sample index", fontsize=14)
        plt.ylabel("Expectation value", fontsize=14)
        plt.title(
            f"Free-Fermion Expectation Values Across Benchmark (N={self.number_of_qubits})",
            fontsize=16,
        )

        plt.grid(alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_expectation_histograms(self, bins: int = 20) -> None:
        """Plot histograms of parallel and orthogonal projection values.

        This is useful for understanding the distribution / quality of the
        projection values across the entire benchmark.

        Requires :meth:`evaluate_benchmark` to have been run so that
        ``self.experimental_results["evaluation"]`` is populated.

        Args:
          bins: Number of histogram bins to use.

        Raises:
          ValueError: If experimental results or evaluation entries are
            missing.
        """
        if self.experimental_results is None:
            raise ValueError(
                "No experimental_results found. Run evaluate_benchmark() first."
            )

        evaluation = self.experimental_results.get("evaluation", {})
        parallel_values = evaluation.get("parallel_values")
        orthogonal_values = evaluation.get("orthogonal_values")

        if parallel_values is None or orthogonal_values is None:
            raise ValueError(
                "Expected values missing — run evaluate_benchmark() first."
            )

        parallel_values = np.asarray(parallel_values, dtype=float)
        orthogonal_values = np.asarray(np.abs(orthogonal_values), dtype=float)

        plt.figure(figsize=(10, 5))
        sc, _, _ = plt.hist(
            parallel_values,
            bins=bins,
            alpha=0.7,
            color="tab:blue",
            label="Parallel values",
        )
        dc, _, _ = plt.hist(
            orthogonal_values,
            bins=bins,
            alpha=0.7,
            color="tab:red",
            label="Expectation values (|·|)",
        )

        plt.xlabel("Projection value", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.title(
            f"Free-Fermion Projection Distributions — N={self.number_of_qubits}",
            fontsize=16,
        )

        # Same visual thresholds as Clifford, just reinterpreted
        plt.vlines(
            1 / np.e,
            0,
            np.max(sc),
            ls="--",
            color="tab:blue",
            label="Parallel threshold = 1/e",
        )
        plt.vlines(
            0.25 / np.e,
            0,
            np.max(dc),
            ls="--",
            color="tab:red",
            label="Orthogonal threshold = 1/(4e)",
        )

        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
