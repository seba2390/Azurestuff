from typing import *
from itertools import combinations

import numpy as np
from numba import jit


def min_cost_partition(nr_qubits: int,
                       k: int,
                       mu: np.ndarray,
                       sigma: np.ndarray,
                       alpha: float) -> Tuple[dict, dict, float]:
    def generate_binary_combinations(n: int, k: int) -> np.ndarray:
        """ Generates all the 'n' chose 'k' combinations w. 'k' ones. """
        num_permutations = 2 ** n
        for indices in combinations(range(n), k):
            # Create a numpy array of zeros of size N
            arr = np.zeros(n, dtype=int)
            # Set ones at the specified positions
            arr[list(indices)] = 1
            yield arr

    def generate_binary_permutations(n: int) -> np.ndarray:
        """ Generates all the 2^n permutations of bitstring w. length 'n'. """
        num_permutations = 2 ** n
        for i in range(num_permutations):
            _binary_string_ = bin(i)[2:].zfill(n)
            yield np.array([int(bit) for bit in _binary_string_])

    def cost(state: np.ndarray, mu: np.ndarray, sigma: np.ndarray, alpha: float) -> float:
        return -np.dot(state, mu) + alpha * np.dot(state, np.dot(sigma, state))

    max_cost_1, min_cost_1, min_comb = -np.inf, np.inf, np.empty(shape=(nr_qubits,))
    for comb in generate_binary_combinations(n=nr_qubits, k=k):
        comb_cost = cost(state=comb, mu=mu, sigma=sigma, alpha=alpha)
        if comb_cost < min_cost_1:
            min_cost_1, min_comb = comb_cost, comb
        if comb_cost > max_cost_1:
            max_cost_1 = comb_cost
    binary_comb = min_comb

    max_cost_2, min_cost_2, min_perm = -np.inf, np.inf, np.empty(shape=(nr_qubits,))
    for perm in generate_binary_permutations(n=nr_qubits):
        perm_cost = cost(state=perm, mu=mu, sigma=sigma, alpha=alpha)
        if perm_cost < min_cost_2:
            min_cost_2, min_perm = perm_cost, perm
        if perm_cost > max_cost_2:
            max_cost_2 = perm_cost
    binary_perm = min_perm

    lmbda = 0
    if min_cost_2 < min_cost_1:
        lmbda = abs(min_cost_2) - abs(min_cost_1)

    constrained_result = {'s': binary_comb, 'c_min': min_cost_1, 'c_max': max_cost_1}
    full_result = {'s': binary_perm, 'c_min': min_cost_2, 'c_max': max_cost_2}
    return constrained_result, full_result, lmbda


def get_qubo(mu: np.ndarray, sigma: np.ndarray, alpha: float, lmbda: float, k: float):
    Q = np.zeros_like(sigma)
    N = Q.shape[0]
    for i in range(N):
        for j in range(N):
            if i == j:
                Q[i, j] += -mu[i] - 2 * k * lmbda + lmbda + alpha * sigma[i, j]
            else:
                Q[i, j] += lmbda + alpha * sigma[i, j]
    offset = lmbda * k ** 2
    return Q, offset
