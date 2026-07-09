# Shor Period-Finding Benchmark

## Table of Contents

- [Background](#background)
- [Benchmark Task](#benchmark-task)
  - [Maximum-Cycle Linear Permutations](#maximum-cycle-linear-permutations)
  - [Quantum Phase Estimation Structure](#quantum-phase-estimation-structure)
- [Benchmark Protocol](#benchmark-protocol)
  - [Step-by-Step Procedure](#step-by-step-procedure)
- [Implementation Notes](#implementation-notes)
- [Benchmark Score and Interpretation](#benchmark-score-and-interpretation)
- [How to Use the Shor Period-Finding Benchmark](#how-to-use-the-shor-period-finding-benchmark)

## Background

Shor’s integer-factoring algorithm provides an exponential speedup over known classical approaches, but a full factorization circuit is too deep and resource intensive for near-term devices. This benchmark isolates the period-finding subroutine and replaces modular exponentiation with a family of maximum-cycle linear permutations. The result is a platform-agnostic, application-driven benchmark that tracks progress in the NISQ and early fault-tolerant regimes.

The benchmark is designed to remain:

- classically verifiable,
- hardware agnostic,
- sensitive to two-qubit gate depth and coherence,
- and meaningful at small problem sizes where random guessing is still distinguishable from real quantum performance.

---

## Benchmark Task

### Maximum-Cycle Linear Permutations

Shor's factoring algorithm performs Shor's period finding subroutine on modular multiplication gates.
This benchmark replaces modular multiplications with a linear permutation $P_n$ acting on $n$-bit strings $\{0,1\}^n$ or equivalently on integers $\{0,1,\dots,2^n-1\}$. The permutation is defined by an $n \times n$ binary companion matrix $M$ over $\mathbb{F}_2$:

$$
b'_i = \left( \sum_{j=1}^{n} M_{ij} b_j \right) \bmod 2.
$$


The period of modular multiplication operations appearing in Shor's factoring algorithm scale exponentially with the problem size $n$. To ensure this property for the permutations used by the benchmark, $M$ is generated from a primitive polynomial with binary coefficients:

$$
p(x) = x^n + c_{n-1}x^{n-1} + \cdots + c_1x + c_0.
$$

The corresponding companion matrix is

$$
M =
\begin{pmatrix}
0 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
c_0 & c_1 & c_2 & \cdots & c_{n-1}
\end{pmatrix}.
$$

Because $p(x)$ is primitive, the permutation has exactly two cycles:

1. the zero bitstring, which is a fixed point,
2. the maximum-length cycle containing all remaining $2^n - 1$ bitstrings.

Thus the period of the nontrivial orbit is exactly

$$
r = 2^n - 1.
$$

### Quantum Phase Estimation Structure

The benchmark follows the standard Quantum Phase Estimation (QPE) pattern: a control register applies powers of the controlled permutation operator to a target register, followed by an inverse Quantum Fourier Transform and measurement.

The control register is sized to capture enough phase precision for recovering the candidate period from continued fractions. A convenient choice is

$$
t = 2n + c,
$$

where $c$ is a small constant.

The target register stores the $n$-qubit bitstring being permuted and is initialized to a nonzero state in the large cycle, typically $|1\rangle \otimes |0\rangle^{\otimes n-1}$.

---

## Benchmark Protocol

### Step-by-Step Procedure

For a selected problem size $n$, the benchmark is evaluated as follows.

1. **Initialization**  
   Take a quantum computer with qubit number $N_{\text{tot}}$. Take $n \le N_{\text{tot}}$. Choose a maximum-cycle linear permutation $P_n$ on $n$-long bitstrings, or equivalently on $A_n = \{0,1,\dots,2^n-1\}$, which maps $0$ to $0$ and has a cycle of length $2^n - 1$.

2. **Circuit preparation**  
   Compile Shor's period-finding circuit shown in Fig. 5 for this permutation $P_n$. Optimization relying on the knowledge of the period $r$ is not allowed. An example of how this compilation is possible in a scalable manner is given in Sec. V.E.

3. **Circuit execution**  
   Execute the circuit $10^4$ times. Each run outputs an integer $j$ between $0$ and $2^t - 1$, where $t$ is the size of the control register as shown in Fig. 5.

4. **Performance evaluation**  
   Take the fraction $j/2^t$ and calculate the last convergent $q/\tilde{r}$ of its continued fractions expansion with denominator smaller than $2^n$. The denominator $\tilde{r}$ of this convergent is the output of the post-processing. A shot is successful if $\tilde{r} = r$.

   If this post-processing procedure is performed on the output of a perfect quantum computer with sufficiently many qubits, the success probability of Shor's period-finding circuit is sufficiently close to

   $$
   p_{s,n} = \frac{\phi(r)}{r},
   $$

   where $\phi$ is Euler's totient function. For a permutation with maximum cycle length, the period is $r = 2^n - 1$.

   The benchmark score $n_s$ is the largest integer such that the quantum computer achieves a success ratio

   $$
   \eta \equiv \frac{q_{s,n}}{p_{s,n}} > 0.15,
   $$

   for all $n \le n_s$.

---

## Implementation Notes

- The benchmark leaves freedom in the number of control qubits used. Additionally it is allowed to use feed-forward operations, which allows reducing the number of qubits in the controll register to one.

## Benchmark Score and Interpretation

## How to Use the Shor Period-Finding Benchmark
