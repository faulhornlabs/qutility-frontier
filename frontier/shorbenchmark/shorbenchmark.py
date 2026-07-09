from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from math import log2, pi
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..utils.quantumbenchmark import Benchmark
from ..utils.quantumcircuit import QuantumCircuit
from ._mersenne_factors import mersenne_prime_factors


# -----------------------------------------------------------------------------
# Primitive polynomial helpers
# -----------------------------------------------------------------------------


def _prime_factors(value: int) -> List[int]:
	"""Return the distinct prime factors of ``value`` via trial division.

	Only used for the (small) polynomial degree, never for ``2**degree - 1``
	(see mersenne_prime_factors in ``_mersenne_factors.py``), since trial
	division on the latter is infeasible beyond modest degrees.
	"""
	factors: List[int] = []
	divisor = 2
	remaining = value
	while divisor**2 <= remaining:
		if remaining % divisor == 0:
			factors.append(divisor)
			while remaining % divisor == 0:
				remaining //= divisor
		divisor += 1 if divisor == 2 else 2
	if remaining > 1:
		factors.append(remaining)
	return factors


def _poly_degree(poly: int) -> int:
	"""Return the degree of a polynomial encoded as an integer"""
	return poly.bit_length() - 1


def _poly_mul(a: int, b: int) -> int:
	"""Multiply two GF(2) polynomials represented as integers.

	Schoolbook multiplication: a * b = sum of a * x**i over every term
	x**i present in b. Bit i of an integer holds the coefficient of x**i,
	so a left shift multiplies a polynomial by x, and XOR adds two
	polynomials (coefficient-wise addition mod 2).
	"""
	product = 0
	multiplier = a  # holds a * x**i on iteration i
	multiplicand = b
	while multiplicand != 0:
		if multiplicand & 1 == 1:  # (multiplicand & 1) gives the lowest bit multiplicand; 1, if b contains the term x**i
			product ^= multiplier  # add a * x**i to the running sum
		multiplicand >>= 1  # move on to b's coefficient of x**(i+1)
		multiplier <<= 1  # a * x**i  ->  a * x**(i+1)
	return product


def _poly_mod(dividend: int, divisor: int) -> int:
	"""Compute dividend mod divisor for GF(2) polynomials.

	Polynomial long division, keeping only the remainder. Each step
	cancels the remainder's leading term by subtracting the right
	multiple of the divisor; in GF(2) subtraction is XOR.
	"""
	if divisor == 0:
		raise ValueError("Polynomial divisor must be non-zero.")
	divisor_degree = _poly_degree(divisor)
	remainder = dividend
	while remainder != 0 and _poly_degree(remainder) >= divisor_degree:
		# divisor * x**shift has the same degree as the remainder, so
		# subtracting it (XOR) cancels the remainder's leading term.
		shift = _poly_degree(remainder) - divisor_degree
		remainder ^= divisor << shift # remainder - divisor * x**shift
	return remainder


def _poly_gcd(a: int, b: int) -> int:
	"""Compute the GF(2) greatest common divisor of two polynomials."""
	left = a
	right = b
	while right != 0:
		left, right = right, _poly_mod(left, right)
	return left


def _poly_pow_mod(base: int, exponent: int, modulus: int) -> int:
	"""Raise a GF(2) polynomial to a power modulo another polynomial.

	Square-and-multiply: write the exponent in binary as sum of powers
	2**i, so base**exponent is the product of base**(2**i) over the set
	bits of the exponent. Note ``exponent`` is an ordinary integer, not
	a polynomial; its bits are read as a number's binary digits here.
	"""
	result = 1
	power = base  # holds base**(2**i) on iteration i
	remaining = exponent
	while remaining != 0:
		if remaining & 1 == 1:  # 2**i appears in the exponent's binary expansion
			result = _poly_mod(_poly_mul(result, power), modulus)
		remaining >>= 1
		if remaining != 0:
			power = _poly_mod(_poly_mul(power, power), modulus)  # square: base**(2**i) -> base**(2**(i+1))
	return result


def _is_primitive_polynomial(poly: int) -> bool:
	"""Return True when ``poly`` is a primitive polynomial over GF(2)."""
	degree = _poly_degree(poly)
	if degree <= 0:
		return False
	# poly & 1 is the constant term: if it is 0, x divides poly, so poly is
	# reducible. (poly >> degree) & 1 is the leading coefficient, which must
	# be 1 for a degree-``degree`` polynomial.
	if (poly & 1) == 0 or ((poly >> degree) & 1) == 0:
		return False

	# Look up the factorization of 2**degree - 1 before doing any modular
	# exponentiation below: those checks cost O(degree) big-integer
	# multiplications each on ~degree-bit operands, so a degree outside the
	# table's coverage should fail fast here rather than after that work.
	order_factors = mersenne_prime_factors(degree)

	x_poly = 0b10  # the polynomial x

	# Irreducibility test over GF(2) (Rabin's test). poly of degree n is
	# irreducible iff x**(2**n) == x mod poly and, for every prime q
	# dividing n, gcd(x**(2**(n/q)) - x, poly) == 1. The gcd condition
	# rules out irreducible factors of degree n/q or smaller.
	for prime_factor in _prime_factors(degree):
		exp = 1 << (degree // prime_factor)  # the integer 2**(n/q)
		# XOR is polynomial subtraction: x**(2**(n/q)) - x mod poly
		test_poly = _poly_pow_mod(x_poly, exp, poly) ^ x_poly
		if _poly_gcd(poly, test_poly) != 1:
			return False

	if _poly_pow_mod(x_poly, 1 << degree, poly) != x_poly:
		return False

	# poly is now known irreducible, so GF(2)[x]/(poly) is the field
	# GF(2**n) and the order of x divides 2**n - 1. x generates the
	# multiplicative group (i.e. poly is primitive) iff its order is not a
	# proper divisor of 2**n - 1, i.e. x**((2**n - 1)/q) != 1 for every
	# prime factor q of 2**n - 1.
	order = (1 << degree) - 1 # 2^n - 1
	for prime_factor in order_factors:
		if _poly_pow_mod(x_poly, order // prime_factor, poly) == 1:
			return False

	return True


def _primitive_polynomial_for_degree(degree: int, sample_id: int = 0, max_attempts: int = 1000) -> int:
	"""Select a primitive polynomial of the requested degree via seeded random search.

	Primitive polynomials of degree n are not rare: there are phi(2**n - 1) / n
	of them, and phi(m) / m is bounded below by Omega(1 / log log m) for any m
	(Mertens' third theorem), so with m ~= 2**n the density of primitive
	polynomials among candidates is never smaller than Omega(1 / (n log n)).
	Random sampling therefore finds one in an expected O(n log n) attempts,
	instead of the exhaustive O(2**n) enumeration this replaces.

	The search is deterministic and reproducible for a given (degree, sample_id)
	pair, but does not preserve any particular ordering of primitive polynomials.
	"""
	if degree < 1:
		raise ValueError("degree must be at least 1")

	upper = 1 << (degree - 1)
	rng = random.Random(f"{degree}:{sample_id}")
	for _ in range(max_attempts):
		middle_bits = rng.randrange(upper)
		poly = (1 << degree) | 1 | (middle_bits << 1)
		if _is_primitive_polynomial(poly):
			return poly

	raise ValueError(
		f"No primitive polynomial found for degree {degree} after {max_attempts} attempts."
	)


def _companion_matrix_from_polynomial(poly: int, degree: int) -> np.ndarray:
	"""Build the companion matrix associated with a primitive polynomial."""
	coeffs = [(poly >> idx) & 1 for idx in range(degree)]
	matrix = np.zeros((degree, degree), dtype=np.uint8)
	for idx in range(degree - 1):
		matrix[idx, idx + 1] = 1
	matrix[degree - 1, :] = np.array(coeffs, dtype=np.uint8)
	return matrix


# -----------------------------------------------------------------------------
# Circuit construction helpers
# -----------------------------------------------------------------------------


def _matrix_multiply_mod2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
	"""Multiply two matrices over GF(2)."""
	return (left @ right) % 2


def _matrix_power_mod2(matrix: np.ndarray, exponent: int) -> np.ndarray:
	"""
	Raise a GF(2) matrix to a non-negative integer power.
	Efficient implementation using repeated squaring.
	"""
	if exponent < 0:
		raise ValueError("exponent must be non-negative")
	size = matrix.shape[0]
	result = np.eye(size, dtype=np.uint8)
	power = matrix.copy() % 2
	remaining = exponent
	while remaining != 0:
		if remaining & 1 == 1:
			result = _matrix_multiply_mod2(result, power) # power = matrix**(2**i) on iteration i
		remaining >>= 1
		if remaining != 0:
			power = _matrix_multiply_mod2(power, power)
	return result % 2


def _append_controlled_phase(
	circuit: QuantumCircuit, control: int, target: int, angle: float
) -> None:
	"""Append a controlled phase rotation using the supported gate set."""
	# This is centralized here so the implementation can be swapped to a native
	# controlled-phase gate later without changing the QFT construction code.
	circuit.add_rz_gate(target, angle / 2.0)
	circuit.add_cx_gate(control, target)
	circuit.add_rz_gate(target, -angle / 2.0)
	circuit.add_cx_gate(control, target)
	circuit.add_rz_gate(control, angle / 2.0)


def _append_toffoli(
	circuit: QuantumCircuit, control_one: int, control_two: int, target: int
) -> None:
	"""Append a relative-phase Toffoli decomposition."""
	# Keep the Toffoli logic in one place so the circuit builder can move to a
	# native multi-control gate without touching the higher-level structure.
	# Relative-phase Toffoli decomposition using only supported gates.
	circuit.add_h_gate(target)
	circuit.add_cx_gate(control_two, target)
	circuit.add_rz_gate(target, -pi / 4)
	circuit.add_cx_gate(control_one, target)
	circuit.add_rz_gate(target, pi / 4)
	circuit.add_cx_gate(control_two, target)
	circuit.add_rz_gate(target, -pi / 4)
	circuit.add_cx_gate(control_one, target)
	circuit.add_rz_gate(control_two, pi / 4)
	circuit.add_rz_gate(target, pi / 4)
	circuit.add_h_gate(target)
	circuit.add_cx_gate(control_one, control_two)
	circuit.add_rz_gate(control_one, pi / 4)
	circuit.add_rz_gate(control_two, -pi / 4)
	circuit.add_cx_gate(control_one, control_two)


def _lower_triangularize(matrix: np.ndarray, section_size: int) -> List[Tuple[int, int]]:
	"""Reduce ``matrix`` (in place) to upper-triangular form with row operations.

	This is the core routine of the Patel-Markov-Hayes (PMH) algorithm
	(quant-ph/0302002). It clears everything below the diagonal; each returned
	pair ``(source, target)`` records one row operation
	``row[target] ^= row[source]`` (polynomially: add row ``source`` into row
	``target`` over GF(2)), listed in the order the operations were applied.

	The improvement over plain Gaussian elimination is Step A below: columns
	are processed in sections of width m = ``section_size``. Restricted to an
	m-column section, every row is one of only 2**m possible bit patterns, so
	when many rows share a pattern, all duplicates can be cancelled against
	one representative row -- one row operation each -- *before* the per-column
	elimination runs. With m = Theta(log n) the total operation count drops
	from O(n**2) to O(n**2 / log n), which is asymptotically optimal.
	"""
	size = matrix.shape[0]
	operations: List[Tuple[int, int]] = []

	for section_start in range(0, size, section_size):
		section_stop = min(section_start + section_size, size)

		# --- Step A: eliminate duplicate sub-rows within this section. ---
		# Key the rows (at or below the section's first diagonal entry) by
		# their bit pattern inside the section's columns. The first row seen
		# with a given pattern becomes the representative; XOR-ing it into any
		# later row with the same pattern zeroes that row's entire section in
		# a single row operation, instead of up to m per-column operations.
		representatives: Dict[bytes, int] = {}
		for row in range(section_start, size):
			pattern = matrix[row, section_start:section_stop]
			if not pattern.any():
				continue  # already all-zero in this section, nothing to cancel
			key = pattern.tobytes()
			representative = representatives.get(key)
			if representative is None:
				representatives[key] = row
			else:
				matrix[row, :] ^= matrix[representative, :]
				operations.append((representative, row))

		# --- Step B: ordinary Gaussian elimination inside the section. ---
		# Thanks to Step A, at most 2**m - 1 rows still have a nonzero pattern
		# here, so each column below clears with few operations.
		for col in range(section_start, section_stop):
			if matrix[col, col] == 0:
				# Put a 1 on the diagonal by adding some lower row that has a
				# 1 in this column. (Row swaps are not needed: adding the row
				# is one operation instead of the three a swap would cost.)
				pivot = next(
					(row for row in range(col + 1, size) if matrix[row, col] == 1),
					None,
				)
				if pivot is None:
					# Column col is zero on and below the diagonal, so the
					# leading columns are linearly dependent.
					raise ValueError("Matrix is not invertible over GF(2).")
				matrix[col, :] ^= matrix[pivot, :]
				operations.append((pivot, col))

			for row in range(col + 1, size):
				# Clear the entry below the diagonal. Step A guarantees that
				# few rows still have a 1 here.
				if matrix[row, col] == 1:
					matrix[row, :] ^= matrix[col, :]
					operations.append((col, row))

	return operations


def _pmh_cnot_schedule(
	matrix: np.ndarray, section_size: Optional[int] = None
) -> List[Tuple[int, int]]:
	"""Return CNOT ``(control, target)`` pairs implementing ``matrix``.

	Patel-Markov-Hayes synthesis. A CNOT with control c and target t maps the
	basis-state bit vector v by v[t] ^= v[c], i.e. it is the elementary matrix
	E(c, t) = I + e_t e_c^T over GF(2); any invertible M is a product of such
	matrices, found here by elimination. The returned list is in circuit
	(time) order: applying the CNOTs left to right implements v -> M v.
	"""
	size = matrix.shape[0]
	if section_size is not None:
		candidate_sizes = [min(section_size, size)]
	else:
		# The asymptotically optimal section width is Theta(log n): wider
		# sections have too many (2**m) distinct patterns for Step A to find
		# duplicates, narrower ones degenerate toward plain elimination. At
		# benchmark-relevant sizes the constant matters more than the theory,
		# and synthesis is cheap (O(n**2) bit operations per attempt), so try
		# every width up to ~log2(n) and keep the shortest schedule.
		candidate_sizes = list(range(1, max(1, round(log2(size))) + 1)) if size > 1 else [1]

	best_schedule: Optional[List[Tuple[int, int]]] = None
	for width in candidate_sizes:
		# Pass 1: clear below the diagonal. Records L-ops with
		# E_Lk ... E_L1 . M = U  (U upper triangular, unit diagonal).
		work = (matrix % 2).astype(np.uint8)
		lower_operations = _lower_triangularize(work, width)

		# Pass 2: clear above the diagonal. Instead of a separate
		# upper-triangular routine, triangularize the *transpose*: U^T is
		# lower triangular, so clearing below its diagonal reduces it all the
		# way to the identity. Records U-ops with  E_Uj ... E_U1 . U^T = I.
		work = work.T.copy()
		upper_operations = _lower_triangularize(work, width)

		# Assemble the circuit. From the two passes:
		#   M = E_L1 ... E_Lk . (E_Uj)^T ... (E_U1)^T
		# using that each row operation is its own inverse over GF(2), and
		# that transposing E(c, t) swaps control and target:
		# E(c, t)^T = E(t, c). Gates applied first in a circuit sit rightmost
		# in the matrix product, so emit the transposed U-ops in recorded
		# order, then the L-ops reversed.
		schedule = [(target, source) for source, target in upper_operations]
		schedule.extend(reversed(lower_operations))

		if best_schedule is None or len(schedule) < len(best_schedule):
			best_schedule = schedule

	assert best_schedule is not None  # candidate_sizes is never empty
	return best_schedule


def _synthesize_linear_map(
	circuit: QuantumCircuit,
	matrix: np.ndarray,
	qubits: Sequence[int],
	control_qubit: Optional[int] = None,
) -> None:
	"""Synthesize a linear reversible map over the given qubit register."""
	# The scheduling logic is separated from the actual gate emission so the
	# construction stays readable even if the gate set grows later.
	for source_idx, target_idx in _pmh_cnot_schedule(matrix):
		source_qubit = qubits[source_idx]
		target_qubit = qubits[target_idx]
		if control_qubit is None:
			circuit.add_cx_gate(source_qubit, target_qubit)
		else:
			_append_toffoli(circuit, control_qubit, source_qubit, target_qubit)


def _inverse_qft(circuit: QuantumCircuit, qubits: Sequence[int]) -> None:
	"""Append an inverse QFT to the specified qubit register."""
	# This stays as a helper so the circuit builder remains focused on the
	# benchmark structure, not on QFT mechanics.
	size = len(qubits)
	for outer in range(size // 2):
		circuit.add_swap_gate(qubits[outer], qubits[size - outer - 1])

	for target_index in range(size):
		for control_index in range(target_index):
			angle = -pi / (2 ** (target_index - control_index))
			_append_controlled_phase(
				circuit,
				qubits[target_index],
				qubits[control_index],
				angle,
			)
		circuit.add_h_gate(qubits[target_index])


# -----------------------------------------------------------------------------
# Benchmark class
# -----------------------------------------------------------------------------


class ShorPeriodFindingBenchmark(Benchmark):
	"""Shor period-finding benchmark based on maximum-cycle linear permutations."""

	BENCHMARK_NAME: str = "shor_period_finding"

	def __init__(
		self,
		number_of_qubits: int,
		sample_size: int = 1,
		*,
		control_register_offset: int = 1,
		shots: int = 10_000,
		**kwargs: Any,
	) -> None:
		"""Initialize the Shor period-finding benchmark configuration."""
		super().__init__(
			number_of_qubits=number_of_qubits,
			sample_size=sample_size,
			shots=shots,
			**kwargs,
		)
		if control_register_offset < 1:
			raise ValueError("control_register_offset must be at least 1.")
		self.control_register_offset = int(control_register_offset)
		self.control_register_size = 2 * self.number_of_qubits + self.control_register_offset
		self.number_of_measurements = 1

	def _build_period_finding_circuit(
		self,
		matrix: np.ndarray,
		sample_id: int,
	) -> QuantumCircuit:
		"""Build the full QPE-style circuit for one benchmark sample."""
		n_target = self.number_of_qubits
		n_control = self.control_register_size
		total_qubits = n_control + n_target

		circuit = QuantumCircuit(
			number_of_qubits=total_qubits,
			number_of_classical_bits=total_qubits,
		)

		control_qubits = list(range(n_control))
		target_qubits = list(range(n_control, total_qubits))

		for qubit in control_qubits:
			circuit.add_h_gate(qubit)

		# Start the permutation register in a non-zero computational basis state.
		circuit.add_x_gate(target_qubits[0])

		# Controlled powers of the linear permutation.
		for power_index, control_qubit in enumerate(control_qubits):
			powered_matrix = _matrix_power_mod2(matrix, 1 << power_index) # M**(2**power_index)
			_synthesize_linear_map(
				circuit,
				powered_matrix,
				target_qubits,
				control_qubit=control_qubit,
			)

		_inverse_qft(circuit, control_qubits)

		# Only the control register is measured; the measured phase estimate is
		# what drives the continued-fractions post-processing step.
		for classical_bit, qubit in enumerate(control_qubits):
			circuit.add_measurement(qubit, classical_bit)

		return circuit

	def _create_single_sample(self, sample_id: int) -> Dict[str, Any]:
		"""
		Create one benchmark sample with circuit and metadata payloads.
		For the Shor period finding benchmark, the benchmark sample contains only a single circuit 
		that implements the period finding algorithm for randomly chosen maximum-cycle linear permutation.
		"""
		degree = self.number_of_qubits
		primitive_polynomial = _primitive_polynomial_for_degree(degree, sample_id)
		companion_matrix = _companion_matrix_from_polynomial(
			primitive_polynomial, degree
		)

		circuit = self._build_period_finding_circuit(companion_matrix, sample_id)
		qasm = circuit.to_qasm(self.emitter_options)

		return {
			"sample_id": sample_id,
			"sample_metadata": {
				"type": "shor_period_finding",
				"number_of_target_qubits": degree,
				"control_register_size": self.control_register_size,
				"control_register_offset": self.control_register_offset,
				"primitive_polynomial": primitive_polynomial,
				"companion_matrix": companion_matrix,
			},
			"circuits": [
				{
					"circuit_id": f"{sample_id}_qpe",
					"observable": None,
					"qasm": qasm,
					"metadata": {
						"kind": "shor_period_finding",
						"sample_id": sample_id,
						"target_qubits": degree,
						"control_register_size": self.control_register_size,
					},
				}
			],
		}

	def evaluate_benchmark(
		self,
		*,
		auto_save: Optional[bool] = None,
		save_to: Optional[Union[str, Path]] = None,
	) -> Dict[str, Any]:
		"""Evaluate attached counts against the benchmark success rule."""
		# The score is the empirical fraction of shots that match the expected
		# period-recovery pattern for this benchmark instance.
		if self.experimental_results is None:
			raise ValueError(
				"No experimental_results attached. Call add_experimental_results(...) first."
			)

		if self.samples is None:
			raise ValueError("Benchmark has no samples. Generate or load the benchmark first.")

		results = self.experimental_results.get("results")
		if results is None:
			raise ValueError("experimental_results has no 'results' entry.")

		if auto_save is not None:
			self.auto_save = bool(auto_save)

		evaluation: Dict[str, Any] = {
			"sample_size": self.sample_size,
			"control_register_size": self.control_register_size,
			"results": {},
		}

		for sample in self.samples:
			sample_id = sample["sample_id"]
			sample_key = str(sample_id)
			circuit_entry = sample["circuits"][0]
			circuit_id = circuit_entry["circuit_id"]
			sample_counts = results.get(circuit_id, {}).get("counts", {})

			successful_shots = 0
			total_shots = 0
			for bitstring, count in sample_counts.items():
				total_shots += int(count)
				if self._is_successful_bitstring(str(bitstring)):
					successful_shots += int(count)

			success_probability = (
				successful_shots / total_shots if total_shots else 0.0
			)

			evaluation["results"][sample_key] = {
				"circuit_id": circuit_id,
				"successful_shots": successful_shots,
				"total_shots": total_shots,
				"success_probability": success_probability,
			}

			results[circuit_id]["successful_shots"] = successful_shots
			results[circuit_id]["success_probability"] = success_probability

		self.experimental_results["evaluation"] = evaluation

		if self.auto_save:
			if save_to is not None:
				self.save_json(filepath=save_to)
			else:
				self.save_json()

		return evaluation

	def _is_successful_bitstring(self, bitstring: str) -> bool:
		"""Return True when the phase estimate recovers the expected period."""
		stripped = bitstring.replace(" ", "")
		if not stripped:
			return False

		if len(stripped) != self.control_register_size:
			return False

		measured_value = int(stripped, 2)
		phase_fraction = Fraction(measured_value, 1 << self.control_register_size)
		candidate = phase_fraction.limit_denominator((1 << self.number_of_qubits) - 1)
		return candidate.denominator == (1 << self.number_of_qubits) - 1

