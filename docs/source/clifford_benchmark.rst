Clifford Volume Benchmark
=========================

This page summarizes the device-level Clifford benchmark implemented in
``frontier.cliffordvolumebenchmark``. It follows the same structure as the
original Markdown notes shipped with the package and describes the motivation,
protocol, and scoring rules.

.. contents::
   :local:
   :depth: 2

Background
----------

The Clifford Volume Benchmark (CVB) probes how reliably a processor implements
random $n$-qubit Clifford circuits. The benchmark targets *algorithmic
primitives* rather than full applications and was designed to

* remain classically verifiable (via the Gottesman–Knill theorem),
* be hardware agnostic (no assumptions on native gates or connectivity), and
* exercise practically relevant circuit shapes.

Because Clifford circuits map Pauli operators to Pauli operators, any sampled
circuit can be simulated efficiently on a classical reference machine, letting
us check the measured expectation values directly against the ideal ones.

Benchmark Task
--------------

Clifford Group and Stabilizer States
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The $n$-qubit Clifford group $\mathcal{C}(n)$ consists of unitaries that send
the Pauli group $\mathcal{P}_n$ to itself under conjugation. Each Clifford
$C \in \mathcal{C}(n)$ prepares a stabilizer state

.. math::

   \rho_C = C |0\rangle \langle 0|^{\otimes n} C^\dagger,

which can be described by an abelian stabilizer group. A minimal generating
set requires only $n$ Pauli operators.

Random Clifford Instances
^^^^^^^^^^^^^^^^^^^^^^^^^

Each sample of the CVB draws a random Clifford tableau (using ``stim``) on
``number_of_qubits`` qubits, converts it to a ``QuantumCircuit`` and then
derives measurement circuits for randomly selected stabilizers and
destabilizers.

Benchmark Protocol
------------------

1. **Sample generation:** draw a random Clifford, construct the base circuit,
   and enumerate all $Z$-stabilizers and $X$-destabilizers.
2. **Selection:** randomly choose a subset of stabilizers and destabilizers
   (up to four of each) per sample.
3. **Measurement circuits:** convert each selected operator into a circuit that
   measures the corresponding observable and outputs its bitstring.
4. **Execution:** run the circuits on hardware, collect counts, and attach them
   via :meth:`Benchmark.add_experimental_results`.

Performance Criteria
--------------------

Evaluations check two sets of inequalities (mirroring the original manuscript):

* **Per-observable margins:** each stabilizer expectation must exceed the
  threshold $\tau_S = 1/e$ by two standard deviations, and each destabilizer
  must stay within $\tau_D = 1/(2e)$.
* **Per-sample averages:** the mean stabilizer and destabilizer expectations,
  with aggregated error bars, must meet the same thresholds.

Implementation Notes
--------------------

Circuit Synthesis
^^^^^^^^^^^^^^^^^

Stim is used to produce random Clifford tableaux, which are translated to the
``QuantumCircuit`` abstraction (supporting H, S, CX). This translation is
performed once per sample and reused for every measurement circuit derived from
that sample.

Measurement of Pauli Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each measurement circuit appends the inverse Clifford of the selected Pauli
observable so that measuring in the computational basis yields the desired
eigenvalue. The benchmark stores the observable string (e.g. ``"+XZI"``) along
with the serialized QASM.

Benchmark Score and Interpretation
----------------------------------

The top-level :meth:`CliffordVolumeBenchmark.evaluate_benchmark` returns per
sample pass/fail flags, worst offending margins, and aggregate distributions of
stabilizer/destabilizer expectation values. Plotting helpers visualize the
expected values and their standard errors.

How to Use the Clifford Benchmark
---------------------------------

1. Instantiate :class:`frontier.CliffordVolumeBenchmark` with the desired
   number of qubits and samples.
2. Call :meth:`create_benchmark` to generate circuits and serialize the dataset
   (optionally saving the JSON payload).
3. Execute the circuits on the target platform and attach counts with
   :meth:`add_experimental_results`.
4. Run :meth:`evaluate_benchmark` to compute the pass/fail criteria and inspect
   the per-sample diagnostics.
