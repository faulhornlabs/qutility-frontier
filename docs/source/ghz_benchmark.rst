GHZ Benchmark Overview
======================

This page summarizes the multipartite-entanglement benchmark implemented in
``frontier.ghzbenchmark``. It mirrors the structure of the Clifford and
Free-Fermion benchmark notes and highlights how to generate GHZ states, which
measurement schemes are available, and how fidelities are estimated.

.. contents::
   :local:
   :depth: 2

Background
----------

GHZ states

.. math::

   | \mathrm{GHZ}_n \rangle
   = \frac{1}{\sqrt{2}} ( |0\rangle^{\otimes n} + |1\rangle^{\otimes n} )

are genuine multipartite entangled stabilizer states. Because they admit a
compact set of Pauli generators, they can be prepared and characterized
efficiently, making them ideal targets for volumetric benchmarking. The goal of
the GHZ benchmark is to prepare an $n$-qubit GHZ resource circuit and then
evaluate device performance using scalable witness or shadow measurements.

GHZ State Generation
--------------------

Every sample begins with a GHZ preparation circuit. Three fanout patterns are
supported; all apply a Hadamard on qubit 0 before entangling it with the rest.

Star-Like Fanout
^^^^^^^^^^^^^^^^

* Qubit 0 controls a CNOT onto each other qubit.
* Lowest depth when long-range connectivity is available.

Linear / Nearest-Neighbour Fanout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* A ladder of CNOTs where each newly entangled qubit acts as the next control.
* Matches 1-D chains or heavy-hex style topologies with only neighbour couplers.

Logarithmic-Depth Fanout
^^^^^^^^^^^^^^^^^^^^^^^^

* A binary-tree schedule of CNOTs that achieves :math:`O(\log n)` depth if
  disjoint CNOTs can run in parallel.
* Useful on devices with partial parallelism or modular coupling graphs.

Benchmark Protocol
------------------

Each sample contains:

1. The GHZ preparation circuit for the chosen topology.
2. Measurement circuits plus metadata describing the scheme and basis choices.
3. Serialized QASM strings (for export) and a JSON-friendly metadata payload.

Witness (Stabilizer) Measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Witness-X:** Append Hadamards to every qubit before measurement. The
  resulting counts estimate :math:`\langle X^{\otimes n} \rangle`.
* **Witness-Z:** Measure directly in the computational basis to estimate the
  adjacent stabilizers :math:`\langle Z_k Z_{k+1} \rangle`.
* Two circuits per sample, regardless of system size.

Shadow-Overlap Measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Apply Hadamards to all qubits to map GHZ to an even-parity superposition.
* For each measurement round:

  1. Randomly choose two distinct qubits.
  2. Randomly choose Pauli bases (X, Y, or Z) for those qubits.
  3. Apply the corresponding single-qubit rotations and measure all qubits.

* Each circuit stores the random ``basis_map`` so post-processing knows which
  Pauli was measured on each selected qubit.

Fidelity Estimation
-------------------

Stabilizer-Based Fidelity Witness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The stabilizer generator set

.. math::

   \mathcal{G} = \{ X^{\otimes n}, Z_1 Z_2, Z_2 Z_3, \dots, Z_{n-1} Z_n \}

is fully determined by the witness-X and witness-Z measurement settings.
Let :math:`\tilde{\mu}_0 = \langle X^{\otimes n} \rangle` and
:math:`\tilde{\mu}_k = \langle Z_k Z_{k+1} \rangle`. The empirical fidelity
lower bound is

.. math::

   \hat{F}_{\min} = \max
     \left(
       0,\
       1 - \tfrac{1}{2} \sum_{l=0}^{n-1} (1 - \tilde{\mu}_l)
     \right).

Variance estimates follow from the :math:`\pm 1` statistics and allow one-sided
certification of :math:`F > 1/2` with user-selectable confidence.

Shadow-Overlap Fidelity Witness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The shadow-overlap method performs localized two-qubit probes using the random
basis selections described above. For each round one forms the GHZ-dependent
projector :math:`L_{z_k}` for the observed pattern and the local observable
:math:`\sigma = (3|s_1\rangle\langle s_1| - I) \otimes (3|s_2\rangle\langle s_2| - I)`,
then computes :math:`\omega = \mathrm{Tr}(L_{z_k} \sigma)`. Averaging all
:math:`\omega_t` values gives :math:`\bar{\omega}`, a fidelity proxy that
certifies :math:`F > 1/2` once :math:`\bar{\omega} \ge 1 - \varepsilon` with
:math:`T = O(\varepsilon^{-2} \log(1/\delta))` rounds.

Implementation Notes
--------------------

* The GHZ benchmark is implemented in ``frontier.ghzbenchmark`` and exposes
  the same ``Benchmark`` API as the Clifford and Free-Fermion benchmarks.
* Helper functions exported at the package root:

  - :func:`frontier.evaluate_fidelity`
  - :func:`frontier.certify_fidelity_gt_half`
  - :func:`frontier.evaluate_shadow_overlap`

* Witness mode always generates exactly two circuits per sample; shadow mode
  generates ``measurement_rounds`` circuits with explicit basis metadata.

References
----------

1. A. Kalev, A. Kyrillidis, and N. M. Linke, *Validating and Certifying
   Stabilizer States*, Phys. Rev. A **99**, 042337 (2019),
   :arxiv:`1808.10786`.
2. H.-Y. Huang, J. Preskill, and M. Soleimanifar, *Certifying Almost All Quantum
   States with Few Single-Qubit Measurements*, Phys. Rev. A **110**, 052401
   (2024), :arxiv:`2404.07281`.
