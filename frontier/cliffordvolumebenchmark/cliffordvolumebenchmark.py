from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import stim  # type: ignore
import matplotlib.pyplot as plt


from ..utils.quantumbenchmark import Benchmark
from ..utils.quantumcircuit import QuantumCircuit


class CliffordVolumeBenchmark(Benchmark):
    """Volumetric benchmark based on random Clifford operator.

    For each sample, this benchmark:

    * Draws a random Clifford tableau on ``number_of_qubits`` qubits.
    * Converts the tableau into a base :class:`QuantumCircuit`.
    * Randomly selects stabilizers and destabilizers.
    * Builds measurement circuits for each selected stabilizer/destabilizer.
    * Exports each measurement circuit as QASM plus an observable string.

    The per-sample object returned by :meth:`_create_single_sample` matches
    the JSON schema expected by :class:`Benchmark`, for example::

        {
          "sample_id": int,
          "sample_metadata": {...},
          "circuits": [
            {
              "circuit_id": str,
              "observable": str,    # e.g. "+XZI..."
              "qasm": str,
              "metadata": {...}
            },
            ...
          ]
        }

    Attributes:
      number_of_measurements: Number of stabilizer/destabilizer measurement
        circuits per sample, computed by :meth:`_compute_number_of_measurements`.
    """

    BENCHMARK_NAME: str = "clifford"

    def __init__(
        self,
        number_of_qubits: int,
        sample_size: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize a :class:`CliffordVolumeBenchmark` instance.

        Args:
          number_of_qubits: Number of qubits in the Clifford tableau and
            in the generated circuits.
          sample_size: Number of independent random Clifford samples to
            generate.
          **kwargs: Additional keyword arguments forwarded to
            :class:`Benchmark`, such as ``format``, ``target_sdk``, or
            ``emitter_options``.
        """
        super().__init__(
            number_of_qubits=number_of_qubits,
            sample_size=sample_size,
            **kwargs,
        )

        # Convenience: cache the number of measurement circuits per sample.
        # This is used by `_create_random_clifford_circuit`.
        self.number_of_measurements: int = self._compute_number_of_measurements()

    # Measurement count rule (override base method)
    def _compute_number_of_measurements(self) -> int:
        """Return the fixed number of stabilizer/destabilizer measurements per sample.

        According to the Clifford Volume benchmark protocol, we measure
        exactly 4 stabilizers and 4 destabilizers per Clifford instance.
        """
        return np.min([self.number_of_qubits, 4])

    # Conversion from stim.Circuit → QuantumCircuit

    def _convert_stim_circuit_to_quantum_circuit(
        self,
        stim_circuit: stim.Circuit,
    ) -> QuantumCircuit:
        """Convert a :class:`stim.Circuit` into a :class:`QuantumCircuit`.

        This method interprets the instructions emitted by
        ``Tableau.to_circuit(method="elimination")`` and maps them onto
        the gate API of :class:`QuantumCircuit`.

        Currently supported gates:

        * ``H`` → ``QuantumCircuit.add_h_gate``
        * ``S`` → ``QuantumCircuit.add_s_gate``
        * ``CX`` / ``CNOT`` → ``QuantumCircuit.add_cx_gate``

        Args:
          stim_circuit: Circuit produced from a random tableau, using
            ``tableau.to_circuit(method="elimination")``.

        Returns:
          QuantumCircuit: Circuit instance with equivalent unitary action.

        Raises:
          ValueError: If an unsupported gate is encountered, or if a
            multi-qubit gate has an inconsistent number of operands.
        """
        qc = QuantumCircuit(
            number_of_qubits=self.number_of_qubits,
            number_of_classical_bits=self.number_of_qubits,
        )

        for instr in stim_circuit:
            parts = str(instr).split()
            name = parts[0].upper()  # e.g. "H", "S", "CX", "CNOT"
            qubits = [int(q) for q in parts[1:]]

            if name == "H":
                for q in qubits:
                    qc.add_h_gate(q)

            elif name == "S":
                for q in qubits:
                    qc.add_s_gate(q)

            elif name in ("CX", "CNOT"):
                if len(qubits) % 2 != 0:
                    raise ValueError(
                        f"{name} instruction has odd number of qubits: {qubits!r}"
                    )
                ctrls = qubits[0::2]
                tgts = qubits[1::2]
                for c, t in zip(ctrls, tgts):
                    qc.add_cx_gate(c, t)

            else:
                raise ValueError(f"Unsupported gate from stim: {name}")

        return qc

    # Random Clifford instance generator

    def _create_random_clifford_circuit(
        self,
    ) -> Tuple[
        List[str],  # all_stabilizers
        List[str],  # selected_stabilizers
        List[QuantumCircuit],  # stabilizer_circuits
        List[str],  # all_destabilizers
        List[str],  # selected_destabilizers
        List[QuantumCircuit],  # destabilizer_circuits
    ]:
        """Generate a random Clifford instance and its measurement circuits.

        Returns:
          A 6-tuple:

            * all_stabilizers:     all Z-output stabilizers (one per qubit).
            * stabilizers:         randomly selected subset of stabilizers.
            * stab_circs:          measurement circuits for the selected stabilizers.
            * all_destabilizers:   all X-output destabilizers (one per qubit).
            * destabilizers:       randomly selected subset of destabilizers.
            * destab_circs:        measurement circuits for the selected destabilizers.
        """
        tableau = stim.Tableau.random(self.number_of_qubits)
        n_meas = self.number_of_measurements

        # Base circuit from tableau
        stim_circ = tableau.to_circuit(method="elimination")
        base_qc = self._convert_stim_circuit_to_quantum_circuit(stim_circ)

        # --- FULL sets of stabilizers / destabilizers ---
        all_stabilizers = [
            str(tableau.z_output(i)).replace("_", "I")
            for i in range(self.number_of_qubits)
        ]
        all_destabilizers = [
            str(tableau.x_output(i)).replace("_", "I")
            for i in range(self.number_of_qubits)
        ]

        # --- RANDOMLY SELECTED subsets (for measurement) ---
        indices_z = np.random.choice(
            self.number_of_qubits,
            size=n_meas,
            replace=False,
        )
        stabilizers = [all_stabilizers[i] for i in indices_z]

        indices_x = np.random.choice(
            self.number_of_qubits,
            size=n_meas,
            replace=False,
        )
        destabilizers = [all_destabilizers[i] for i in indices_x]

        stab_circs, destab_circs = self._add_measurements_to_circuits(
            base_qc,
            stabilizers,
            destabilizers,
        )

        return (
            all_stabilizers,
            stabilizers,
            stab_circs,
            all_destabilizers,
            destabilizers,
            destab_circs,
        )

    def _add_measurements_to_circuits(
        self,
        quantum_circuit: QuantumCircuit,
        stabilizers: List[str],
        destabilizers: List[str],
    ) -> Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
        """Attach basis rotations and measurements for each Pauli string.

        For each stabilizer or destabilizer Pauli string, this method:

        * Clones the base Clifford :class:`QuantumCircuit`.
        * Applies single-qubit rotations to map X/Y operators to Z.
        * Measures each qubit into its corresponding classical bit.

        The Pauli strings are expected to have the format ``'+XYZ...'``,
        where the first character is the sign (``'+'`` or ``'-'``) and the
        remaining characters are one-qubit operators on each site.

        Args:
          quantum_circuit: Base Clifford circuit (without measurement).
          stabilizers: List of stabilizer Pauli strings with sign.
          destabilizers: List of destabilizer Pauli strings with sign.

        Returns:
          Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
            A pair ``(stabilizer_circuits, destabilizer_circuits)``, where
            each element is a list of :class:`QuantumCircuit` instances
            ready to be exported (for example, to QASM).
        """
        stabilizer_circuits: List[QuantumCircuit] = []

        # Stabilizers
        for pauli in stabilizers:
            qc_copy = deepcopy(quantum_circuit)
            # pauli[0] is the sign; pauli[1:] contains single-qubit operators
            for j in range(self.number_of_qubits):
                op = pauli[j + 1]
                if op == "X":
                    qc_copy.add_h_gate(j)
                elif op == "Y":
                    qc_copy.add_sdg_gate(j)
                    qc_copy.add_h_gate(j)
                # "Z" and "I" require no basis change

            for qm in range(self.number_of_qubits):
                qc_copy.add_measurement(qm, qm)

            stabilizer_circuits.append(qc_copy)

        destabilizer_circuits: List[QuantumCircuit] = []

        # Destabilizers
        for pauli in destabilizers:
            qc_copy = deepcopy(quantum_circuit)
            for j in range(self.number_of_qubits):
                op = pauli[j + 1]
                if op == "X":
                    qc_copy.add_h_gate(j)
                elif op == "Y":
                    qc_copy.add_sdg_gate(j)
                    qc_copy.add_h_gate(j)
                # "Z" and "I" require no basis change

            for qm in range(self.number_of_qubits):
                qc_copy.add_measurement(qm, qm)

            destabilizer_circuits.append(qc_copy)

        return stabilizer_circuits, destabilizer_circuits

    # Required by Benchmark: create one sample

    def _create_single_sample(self, sample_id: int) -> Dict[str, Any]:
        """Create one benchmark sample.

        This constructs a single sample dictionary matching the schema
        expected by :class:`Benchmark` and your JSON schema, using the
        random Clifford instance and its measurement circuits.

        Args:
          sample_id: Identifier for the sample (0-based index assigned by
            :meth:`Benchmark.create_benchmark`).

        Returns:
          Dict[str, Any]: A dictionary of the form::

              {
                "sample_id": int,
                "sample_metadata": {...},
                "circuits": [
                  {
                    "circuit_id": str,
                    "observable": str | null,
                    "qasm": str,
                    "metadata": {...},
                  },
                  ...
                ],
              }
        """
        (
            all_stabilizers,
            stabilizers,
            stab_circs,
            all_destabilizers,
            destabilizers,
            destab_circs,
        ) = self._create_random_clifford_circuit()

        circuits: List[Dict[str, Any]] = []

        # Stabilizer circuits
        for idx, (pauli, qc) in enumerate(zip(stabilizers, stab_circs)):
            qasm = qc.to_qasm(self.emitter_options)
            circuits.append(
                {
                    "circuit_id": f"{sample_id}_stab_{idx}",
                    "observable": pauli,  # e.g. "+XZI..."
                    "qasm": qasm,
                    "metadata": {
                        "kind": "stabilizer",
                        "index": idx,
                    },
                }
            )

        # Destabilizer circuits
        for idx, (pauli, qc) in enumerate(zip(destabilizers, destab_circs)):
            qasm = qc.to_qasm(self.emitter_options)
            circuits.append(
                {
                    "circuit_id": f"{sample_id}_destab_{idx}",
                    "observable": pauli,
                    "qasm": qasm,
                    "metadata": {
                        "kind": "destabilizer",
                        "index": idx,
                    },
                }
            )

        sample_metadata: Dict[str, Any] = {
            "type": "clifford",
            "number_of_measurements": self.number_of_measurements,
            "clifford_operator": {
                "stabilizers": all_stabilizers,
                "destabilizers": all_destabilizers,
            },
        }

        return {
            "sample_id": sample_id,
            "sample_metadata": sample_metadata,
            "circuits": circuits,
        }

    # Benchmark evaluation helpers

    def compute_expectation_values(self) -> Dict[str, float]:
        """Compute expectation values for all circuits using experimental results.

        This scans all circuits in the benchmark and computes expectation
        values for those with a non-None ``observable`` field, using
        :meth:`Benchmark.expected_value`.

        Returns:
          Dict[str, float]: Mapping from ``circuit_id`` to expectation value.

        Raises:
          ValueError: If experimental results are missing, malformed, or
            inconsistent with the stored samples.
        """
        if self.experimental_results is None:
            raise ValueError(
                "No experimental_results found. "
                "Run benchmark.add_experimental_results() first."
            )

        results = self.experimental_results.get("results")
        if results is None:
            raise ValueError("experimental_results has no 'results' entry.")

        if self.samples is None:
            raise ValueError(
                "Benchmark has no samples. Load or generate the benchmark first."
            )

        expectation_map: Dict[str, float] = {}

        # Iterate through all circuits in canonical order
        for sample in self.samples:
            for circuit in sample["circuits"]:
                cid = circuit["circuit_id"]
                pauli = circuit["observable"]

                # Some circuits may have observable = None
                if pauli is None:
                    continue

                # Extract counts for this circuit
                if cid not in results:
                    raise ValueError(
                        f"Missing result for circuit_id {cid!r} in experimental_results."
                    )

                counts = results[cid]["counts"]

                # Compute expectation value
                ev = self.expected_value(counts, pauli)
                expectation_map[cid] = ev

        return expectation_map

    def evaluate_benchmark(
        self,
        *,
        auto_save: Optional[bool] = None,
        save_to: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Evaluate the Clifford benchmark using experimental results.

        Implements the manuscript criteria:

        (I)  Per-observable (worst-case):
            <S> - 2σ >= τ_S  and  |<D>| + 2σ <= τ_D

        (II) Per-Clifford-instance averages:
            mean(<S>) - 5 σ̄ >= τ_S  and  |mean(<D>)| + 5 σ̄ <= τ_D

        Returns a structured dictionary including pass/fail flags.
        """
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

        # Thresholds from the manuscript
        tau_S = 1 / np.e
        tau_D = 1 / (2 * np.e)

        # Collect all EVs for global summaries/plots
        stabilizer_evs: List[float] = []
        stabilizer_errs: List[float] = []
        destabilizer_evs: List[float] = []
        destabilizer_errs: List[float] = []

        # Per-sample evaluation
        per_sample: Dict[int, Dict[str, Any]] = {}

        # Helper for ±1 outcome EV estimator uncertainty
        def _sigma_from_ev(ev: float) -> float:
            return float(np.sqrt(max(0.0, 1.0 - ev * ev) / self.shots))

        # Walk sample-by-sample (each sample corresponds to one random Clifford)
        for sample in self.samples:
            sid = sample["sample_id"]

            sample_stab: List[Tuple[str, float, float]] = []
            sample_dest: List[Tuple[str, float, float]] = []

            # Evaluate each circuit in this sample
            for circuit in sample.get("circuits", []):
                cid = circuit["circuit_id"]
                pauli = circuit.get("observable")
                if pauli is None:
                    continue

                if cid not in results:
                    raise ValueError(
                        f"Missing result for circuit_id {cid!r} in experimental_results['results']."
                    )

                counts = results[cid]["counts"]
                ev = float(self.expected_value(counts, pauli))
                sigma = _sigma_from_ev(ev)

                # cache per-circuit values
                results[cid]["expectation_value"] = ev
                results[cid]["std_error"] = sigma

                kind = circuit.get("metadata", {}).get("kind")
                if kind == "stabilizer":
                    sample_stab.append((pauli, ev, sigma))
                    stabilizer_evs.append(ev)
                    stabilizer_errs.append(sigma)
                elif kind == "destabilizer":
                    sample_dest.append((pauli, ev, sigma))
                    destabilizer_evs.append(ev)
                    destabilizer_errs.append(sigma)

            if len(sample_stab) == 0 or len(sample_dest) == 0:
                raise ValueError(
                    f"Sample {sid} has missing stabilizer/destabilizer circuits "
                    f"(stab={len(sample_stab)}, dest={len(sample_dest)})."
                )

            # --- Criterion (I): per-observable worst-case ---
            stab_margins = [(ev - 2.0 * sig) for _, ev, sig in sample_stab]
            dest_margins = [(abs(ev) + 2.0 * sig) for _, ev, sig in sample_dest]

            crit_I_stab_pass = all(m >= tau_S for m in stab_margins)
            crit_I_dest_pass = all(m <= tau_D for m in dest_margins)
            crit_I_pass = crit_I_stab_pass and crit_I_dest_pass

            # --- Criterion (II): per-sample averages ---
            stab_vals = np.array([ev for _, ev, _ in sample_stab], dtype=float)
            stab_sigs = np.array([sig for _, _, sig in sample_stab], dtype=float)
            dest_vals = np.array([ev for _, ev, _ in sample_dest], dtype=float)
            dest_sigs = np.array([sig for _, _, sig in sample_dest], dtype=float)

            mean_stab = float(np.mean(stab_vals))
            mean_dest = float(np.mean(dest_vals))

            # Match manuscript definition:
            #   σ̄ = sqrt((1/m) Σ σ_i^2)
            mS = len(stab_sigs)
            mD = len(dest_sigs)
            sigma_bar_stab = float(np.sqrt((1.0 / mS) * np.sum(stab_sigs**2)))
            sigma_bar_dest = float(np.sqrt((1.0 / mD) * np.sum(dest_sigs**2)))

            crit_II_stab_pass = (mean_stab - 5.0 * sigma_bar_stab) >= tau_S
            crit_II_dest_pass = (abs(mean_dest) + 5.0 * sigma_bar_dest) <= tau_D
            crit_II_pass = crit_II_stab_pass and crit_II_dest_pass

            passed = crit_I_pass and crit_II_pass

            per_sample[sid] = {
                "passed": bool(passed),
                "criterion_I": {
                    "passed": bool(crit_I_pass),
                    "stabilizer_passed": bool(crit_I_stab_pass),
                    "destabilizer_passed": bool(crit_I_dest_pass),
                    "min_stab_margin": float(np.min(stab_margins)),
                    "max_dest_margin": float(np.max(dest_margins)),
                },
                "criterion_II": {
                    "passed": bool(crit_II_pass),
                    "stabilizer_passed": bool(crit_II_stab_pass),
                    "destabilizer_passed": bool(crit_II_dest_pass),
                    "mean_stab": mean_stab,
                    "mean_dest": mean_dest,
                    "sigma_bar_stab": sigma_bar_stab,
                    "sigma_bar_dest": sigma_bar_dest,
                },
                "thresholds": {"tau_S": float(tau_S), "tau_D": float(tau_D)},
            }

        # Store grouped lists for your plotting utilities
        self.experimental_results.setdefault("evaluation", {})
        self.experimental_results["evaluation"]["stabilizer_expectation_values"] = (
            stabilizer_evs
        )
        self.experimental_results["evaluation"]["destabilizer_expectation_values"] = (
            destabilizer_evs
        )
        self.experimental_results["evaluation"]["stabilizer_std_errors"] = (
            stabilizer_errs
        )
        self.experimental_results["evaluation"]["destabilizer_std_errors"] = (
            destabilizer_errs
        )
        self.experimental_results["evaluation"]["per_sample"] = per_sample
        self.experimental_results["evaluation"]["thresholds"] = {
            "tau_S": float(tau_S),
            "tau_D": float(tau_D),
        }

        # --- Printing (updated, manuscript-aligned) ---
        n_pass = sum(1 for v in per_sample.values() if v["passed"])
        n_tot = len(per_sample)

        print("\n==============================================================")
        print(f" Clifford Benchmark Evaluation ({self.number_of_qubits} qubits)")
        print("==============================================================")
        print(f"Thresholds: τ_S = 1/e = {tau_S:.6f}   τ_D = 1/(2e) = {tau_D:.6f}")
        print(f"Shots per circuit: {self.shots}")
        print("--------------------------------------------------------------")

        if len(stabilizer_evs) > 0:
            print("Stabilizers (all measured):")
            print(
                f"  • mean ± std: {np.mean(stabilizer_evs):.6f} ± {np.std(stabilizer_evs):.6f}"
            )
            print(f"  • min EV:     {np.min(stabilizer_evs):.6f}")
        if len(destabilizer_evs) > 0:
            print("Destabilizers (all measured):")
            print(
                f"  • mean ± std: {np.mean(destabilizer_evs):.6f} ± {np.std(destabilizer_evs):.6f}"
            )
            print(f"  • max |EV|:   {np.max(np.abs(destabilizer_evs)):.6f}")

        print("--------------------------------------------------------------")
        print(f"Per-sample pass count: {n_pass}/{n_tot}")

        # Show the worst offending margins (useful for debugging)
        worst_stab = min(
            per_sample.items(), key=lambda kv: kv[1]["criterion_I"]["min_stab_margin"]
        )
        worst_dest = max(
            per_sample.items(), key=lambda kv: kv[1]["criterion_I"]["max_dest_margin"]
        )
        print(
            f"Worst stabilizer margin (min over samples of <S>-2σ): "
            f"sample {worst_stab[0]} -> {worst_stab[1]['criterion_I']['min_stab_margin']:.6f}"
        )
        print(
            f"Worst destabilizer margin (max over samples of |<D>|+2σ): "
            f"sample {worst_dest[0]} -> {worst_dest[1]['criterion_I']['max_dest_margin']:.6f}"
        )

        overall_passed = n_pass == n_tot
        print(f"Benchmark passed (all samples): {overall_passed}")
        print("==============================================================\n")

        # Auto-save logic unchanged
        if self.auto_save:
            if save_to is not None:
                saved_path = self.save_json(filepath=save_to)
            elif self.path is not None:
                saved_path = self.save_json(filepath=self.path)
            else:
                saved_path = self.save_json()
            print(f"[Benchmark] Saved updated JSON to: {saved_path}")

        return {
            "stabilizer_expectation_values": stabilizer_evs,
            "destabilizer_expectation_values": destabilizer_evs,
            "per_sample": per_sample,
            "thresholds": {"tau_S": float(tau_S), "tau_D": float(tau_D)},
            "passed": bool(overall_passed),
        }

    def get_all_expectation_value(
        self,
    ) -> Dict[int, Dict[str, Dict[str, Tuple[float, float]]]]:
        """Return expectation values and errors grouped by sample and kind.

        The output is grouped first by ``sample_id``, then by measurement kind
        (``"stabilizer"`` or ``"destabilizer"``), and finally by the Pauli
        observable string.

        Uses stored ``"expectation_value"`` / ``"std_error"`` in
        ``experimental_results["results"]`` if available; otherwise computes
        them on the fly from counts and :attr:`shots`, and caches them.

        Returns:
          Dict[int, Dict[str, Dict[str, Tuple[float, float]]]]: Nested mapping
          of the form::

              {
                sample_id: {
                  "stabilizer": {
                      observable_str: (expectation_value, std_error),
                      ...
                  },
                  "destabilizer": {
                      observable_str: (expectation_value, std_error),
                      ...
                  },
                },
                ...
              }

        Raises:
          ValueError: If experimental results or samples are missing, if
            required count entries are missing, or if :attr:`shots` is not a
            positive integer.
        """
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

        out: Dict[int, Dict[str, Dict[str, Tuple[float, float]]]] = {}

        for sample in self.samples:
            sid = sample["sample_id"]

            out[sid] = {
                "stabilizer": {},
                "destabilizer": {},
            }

            for circuit in sample.get("circuits", []):
                cid = circuit["circuit_id"]
                pauli = circuit.get("observable")
                if pauli is None:
                    continue

                kind = circuit.get("metadata", {}).get("kind")
                if kind not in ("stabilizer", "destabilizer"):
                    continue

                if cid not in results:
                    raise ValueError(
                        f"Missing result for circuit_id {cid!r} "
                        "in experimental_results['results']."
                    )

                entry = results[cid]

                # Expectation value: use stored or compute
                if "expectation_value" in entry:
                    ev = float(entry["expectation_value"])
                else:
                    counts = entry["counts"]
                    ev = float(self.expected_value(counts, pauli))
                    entry["expectation_value"] = ev  # cache

                # Std error: use stored or compute
                if "std_error" in entry:
                    std_err = float(entry["std_error"])
                else:
                    std_err = float(np.sqrt(max(0.0, 1.0 - ev * ev) / self.shots))
                    entry["std_error"] = std_err  # cache

                out[sid][kind][pauli] = (ev, std_err)

        return out

    def plot_expected_values(self, sample_id: int) -> None:
        """Plot stabilizer and destabilizer expectation values for a sample.

        Generates two error-bar plots: one for stabilizer expectation values
        and one for destabilizer expectation values, for a given sample
        index.

        Values are taken from :meth:`get_all_expectation_value`, which
        computes or reuses cached expectation values and standard errors.

        Args:
          sample_id: Index of the benchmark sample to visualize.

        Raises:
          ValueError: If experimental results are missing, the benchmark
            has no samples, or the given sample ID is invalid or has
            missing data.
        """
        # Retrieve per-sample expectation values
        all_values = self.get_all_expectation_value()

        if sample_id not in all_values:
            raise ValueError(f"Sample {sample_id} not found in benchmark.")

        stabilizers = all_values[sample_id]["stabilizer"]
        destabilizers = all_values[sample_id]["destabilizer"]

        stab_ops = list(stabilizers.keys())
        stab_vals = [stabilizers[op][0] for op in stab_ops]
        stab_errs = [stabilizers[op][1] for op in stab_ops]

        x = np.arange(len(stab_ops))

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            x,
            stab_vals,
            yerr=[3 * e for e in stab_errs],
            fmt="s",
            capsize=6,
            markersize=7,
            color="tab:blue",
            ecolor="tab:blue",
            elinewidth=1.2,
        )

        plt.xlabel("Stabilizer Operators", fontsize=14, labelpad=10)
        plt.ylabel("Expectation Value", fontsize=14, labelpad=10)
        plt.title(
            f"Stabilizer Expectation Values — N={self.number_of_qubits} (Sample {sample_id})",
            fontsize=16,
            pad=12,
        )
        plt.xticks(x, stab_ops, rotation=70, ha="right", fontsize=12)

        plt.grid(which="major", linestyle="-", alpha=0.5)
        plt.grid(which="minor", linestyle="--", alpha=0.3)
        plt.minorticks_on()
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.tight_layout()
        plt.show()

        dest_ops = list(destabilizers.keys())
        dest_vals = [destabilizers[op][0] for op in dest_ops]
        dest_errs = [destabilizers[op][1] for op in dest_ops]

        x = np.arange(len(dest_ops))

        plt.figure(figsize=(10, 6))
        plt.errorbar(
            x,
            dest_vals,
            yerr=[3 * e for e in dest_errs],
            fmt="s",
            capsize=6,
            markersize=7,
            color="tab:red",
            ecolor="tab:red",
            elinewidth=1.2,
        )

        plt.xlabel("Destabilizer Operators", fontsize=14, labelpad=10)
        plt.ylabel("Expectation Value", fontsize=14, labelpad=10)
        plt.title(
            f"Destabilizer Expectation Values — N={self.number_of_qubits} (Sample {sample_id})",
            fontsize=16,
            pad=12,
        )
        plt.xticks(x, dest_ops, rotation=70, ha="right", fontsize=12)

        plt.grid(which="major", linestyle="-", alpha=0.5)
        plt.grid(which="minor", linestyle="--", alpha=0.3)
        plt.minorticks_on()
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_all_expectation_values(self) -> None:
        """Plot all stabilizer and destabilizer expectation values.

        Plots expectation values (with standard error bars) across the
        entire benchmark, with separate markers for stabilizers and
        destabilizers and threshold guide lines for pass/fail criteria.

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
        if (
            "stabilizer_expectation_values" not in evaluation
            or "destabilizer_expectation_values" not in evaluation
        ):
            raise ValueError(
                "Expected values missing — run evaluate_benchmark() first."
            )

        stabilizer_expectation_value = evaluation["stabilizer_expectation_values"]
        stabilizer_expectation_error = [
            float(np.sqrt(max(0.0, 1.0 - ev * ev) / self.shots))
            for ev in stabilizer_expectation_value
        ]

        destabilizer_expectation_value = np.abs(
            evaluation["destabilizer_expectation_values"]
        )
        destabilizer_expectation_error = [
            float(np.sqrt(max(0.0, 1.0 - ev * ev) / self.shots))
            for ev in destabilizer_expectation_value
        ]

        # Build x-axes
        x = 1 + np.arange(self.sample_size * self.number_of_measurements)

        plt.figure(figsize=(12, 6))

        # Stabilizers (blue)
        plt.errorbar(
            x,
            stabilizer_expectation_value,
            yerr=[3 * e for e in stabilizer_expectation_error],
            fmt="o",
            markersize=6,
            capsize=4,
            color="tab:blue",
            ecolor="tab:blue",
            label="Stabilizers",
        )

        # Destabilizers (red)
        plt.errorbar(
            x,
            destabilizer_expectation_value,
            yerr=[3 * e for e in destabilizer_expectation_error],
            fmt="o",
            markersize=6,
            capsize=4,
            color="tab:red",
            ecolor="tab:red",
            label="Destabilizers",
        )

        # Thresholds
        stab_thresh = 1 / np.e
        dest_thresh = 1 / (2 * np.e)

        plt.axhline(
            stab_thresh,
            linestyle="--",
            linewidth=2,
            color="tab:blue",
            label="Stabilizer threshold = 1/e",
        )
        plt.axhline(
            dest_thresh,
            linestyle="--",
            linewidth=2,
            color="tab:red",
            label="Destabilizer threshold = 1/(4e)",
        )

        plt.xlabel("Circuit index", fontsize=14)
        plt.ylabel("Expectation value", fontsize=14)
        plt.title(
            f"Expectation Values Across Benchmark (N={self.number_of_qubits})",
            fontsize=16,
        )

        plt.grid(alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_expectation_histograms(self, bins: int = 20) -> None:
        """Plot histograms of stabilizer and destabilizer expectation values.

        This is useful for understanding the distribution / quality of the
        measured expectation values across the entire benchmark.

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
        stab = evaluation.get("stabilizer_expectation_values")
        dest = evaluation.get("destabilizer_expectation_values")

        if stab is None or dest is None:
            raise ValueError(
                "Expected values missing — run evaluate_benchmark() first."
            )

        stab = np.asarray(stab, dtype=float)
        dest = np.asarray(np.abs(dest), dtype=float)

        plt.figure(figsize=(10, 5))
        sc, _, _ = plt.hist(
            stab,
            bins=bins,
            alpha=0.7,
            color="tab:blue",
            label="Stabilizers",
        )
        dc, _, _ = plt.hist(
            dest,
            bins=bins,
            alpha=0.7,
            color="tab:red",
            label="Destabilizers",
        )

        plt.xlabel("Expectation value", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.title(
            f"Expectation Value Distributions — N={self.number_of_qubits}",
            fontsize=16,
        )

        plt.vlines(
            1 / np.e,
            0,
            np.max(sc),
            ls="--",
            color="tab:blue",
            label="Stabilizer threshold = 1/e",
        )
        plt.vlines(
            1 / (2 * np.e),
            0,
            np.max(dc),
            ls="--",
            color="tab:red",
            label="Destabilizer threshold = 1/(4e)",
        )

        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
