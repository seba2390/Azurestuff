from typing import List, Tuple, Dict
from itertools import combinations

import numpy as np
from numba import jit


##########################################
# ---------- HELPER FUNCTIONS ---------- #
##########################################

def string_to_array(string_rep: str) -> np.ndarray:
    return np.array([int(bit) for bit in string_rep]).astype(np.float64)


@jit(nopython=True, cache=True)
def qubo_cost(state: np.ndarray, QUBO_matrix: np.ndarray) -> float:
    return np.dot(state, np.dot(QUBO_matrix, state))


def portfolio_metrics(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """ Generates random PSD covar matrix and random expected returns vector"""
    np.random.seed(seed)
    _expected_returns_ = np.random.uniform(low=0, high=0.99, size=n)
    _temp_ = np.random.uniform(low=0, high=0.99, size=(n, n))
    _covariances_ = np.dot(_temp_, _temp_.transpose())
    if not np.all(_covariances_ == _covariances_.T) or not np.all(np.linalg.eigvals(_covariances_) >= 0):
        raise ValueError('Covariance matrix is not PSD.')

    return _expected_returns_, _covariances_


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

    _lmbda_ = 0
    if min_cost_2 < min_cost_1:
        _lmbda_ = abs(min_cost_1 - min_cost_2)

    _constrained_result_ = {'s': binary_comb, 'c_min': min_cost_1, 'c_max': max_cost_1}
    _full_result_ = {'s': binary_perm, 'c_min': min_cost_2, 'c_max': max_cost_2}
    return _constrained_result_, _full_result_, _lmbda_


def get_qubo(mu: np.ndarray, sigma: np.ndarray, alpha: float, lmbda: float, k: float):
    """Generates the QUBO for the portfolio objective: s^T*mu + alpha*(s^T*Covar*s) w. a
    lambda penalty for more than 'k' bits set. """
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


def normalized_cost(result: Dict[str, float],
                    QUBO_matrix: np.ndarray,
                    QUBO_offset,
                    max_cost: float,
                    min_cost: float) -> float:
    """ Calculates the QUBO cost of the single most probable state in the
    result state dict, and normalizes it wrt. min and max possible cost."""
    best_state = list(result.keys())[np.argmax(list(result.values()))]
    found_cost = qubo_cost(np.array([float(_) for _ in best_state]).astype(np.float64), QUBO_matrix) + QUBO_offset
    return abs(found_cost - min_cost) / abs(max_cost - min_cost)


def qubo_to_ising(Q, offset=0.0):
    """Convert a QUBO problem to an Ising problem."""
    h = {}
    J = {}
    linear_offset = 0.0
    quadratic_offset = 0.0

    for (u, v), bias in Q.items():
        if u == v:
            if u in h:
                h[u] += .5 * bias
            else:
                h[u] = .5 * bias
            linear_offset += bias

        else:
            if bias != 0.0:
                J[(u, v)] = .25 * bias

            if u in h:
                h[u] += .25 * bias
            else:
                h[u] = .25 * bias

            if v in h:
                h[v] += .25 * bias
            else:
                h[v] = .25 * bias

            quadratic_offset += bias

    offset += .5 * linear_offset + .25 * quadratic_offset

    return h, J, offset


def get_ising(Q: np.ndarray, offset: float):
    _Q_dict = {}
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            _Q_dict[(i, j)] = Q[i, j]

    _h_dict, _J_dict, _offset_ = qubo_to_ising(Q=_Q_dict, offset=offset)
    J_list, h_list = [], []
    for key in _h_dict.keys():
        h_list.append((key, _h_dict[key]))
    for key in _J_dict.keys():
        i, j = key
        J_list.append((i, j, _J_dict[key]))
    return J_list, h_list


def check_qubo(QUBO_matrix: np.ndarray,
               QUBO_offset: float,
               expected_returns: np.ndarray,
               covariances: np.ndarray,
               alpha: float,
               k: int):
    """ Runs through all permutations and checks that QUBO cost is equivalent to
    portfolio cost """

    def generate_binary_permutations(n: int) -> np.ndarray:
        """ Generates all the 2^n permutations of bitstring w. length 'n'. """
        num_permutations = 2 ** n
        for i in range(num_permutations):
            _binary_string_ = bin(i)[2:].zfill(n)
            yield np.array([int(bit) for bit in _binary_string_])

    def qubo_cost(state: np.ndarray, QUBO_matrix: np.ndarray, QUBO_offset: float) -> float:
        return np.dot(state, np.dot(QUBO_matrix, state)) + QUBO_offset

    def portfolio_cost(state: np.ndarray, mu: np.ndarray, sigma: np.ndarray, alpha: float) -> float:
        return -np.dot(state, mu) + alpha * np.dot(state, np.dot(sigma, state))

    N_QUBITS = QUBO_matrix.shape[0]
    for state in generate_binary_permutations(n=N_QUBITS):
        QUBO_cost = qubo_cost(state=state, QUBO_matrix=QUBO_matrix, QUBO_offset=QUBO_offset)
        PORTFOLIO_cost = portfolio_cost(state=state, mu=expected_returns, sigma=covariances, alpha=alpha)
        if not np.isclose(QUBO_cost, PORTFOLIO_cost):
            if np.sum(state) == k:
                raise ValueError(f'state={"|"+"".join([str(_) for _ in state])+">"}, QUBO: {QUBO_cost}, PORTFOLIO: {PORTFOLIO_cost}')


def qubo_limits(Q: np.ndarray, offset: float):
    """Calculates the max and the min cost of the given qubo (and offset),
    together with the corresponding states."""

    def generate_binary_permutations(n: int) -> np.ndarray:
        """ Generates all the 2^n permutations of bitstring w. length 'n'. """
        num_permutations = 2 ** n
        for i in range(num_permutations):
            _binary_string_ = bin(i)[2:].zfill(n)
            yield np.array([int(bit) for bit in _binary_string_])

    def qubo_cost(state: np.ndarray, QUBO_matrix: np.ndarray, QUBO_offset: float) -> float:
        return np.dot(state, np.dot(QUBO_matrix, state)) + QUBO_offset

    N_QUBITS = Q.shape[0]
    min_qubo_cost, max_qubo_cost = np.inf, -np.inf
    min_qubo_state, max_qubo_state = None, None
    for state in generate_binary_permutations(n=N_QUBITS):
        c = qubo_cost(state=state, QUBO_matrix=Q, QUBO_offset=offset)
        if c < min_qubo_cost:
            min_qubo_cost = c
            min_qubo_state = state
        elif c > max_qubo_cost:
            max_qubo_cost = c
            max_qubo_state = state
    return {'c_min': min_qubo_cost, 'c_max': max_qubo_cost,
            'min_state': min_qubo_state, 'max_state': max_qubo_state}
