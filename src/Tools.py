from typing import *
from itertools import combinations
import os

import numpy as np
import matplotlib.pyplot as plt
import h5py

from src.cp_QAOATools import qubo_to_ising


def represent_integer_with_bits(number: int, nr_bits: int) -> str:
    """
    Represent an integer using a specific number of bits.

    Args:
        number (int): The integer to be represented.
        nr_bits (int): The number of bits to use for representation.

    Returns:
        str: A binary string representing the integer with leading zeros if required.
    """
    # Convert the integer to a binary string and remove the '0b' prefix
    binary_string = bin(number)[2:]
    # If the binary string is shorter than n, pad it with leading zeros
    binary_string = binary_string.zfill(nr_bits)
    return binary_string


def generate_bit_string_permutations(n: int) -> str:
    """
    A 'generator' type function that calculates all 2^n-1
    possible bitstring of a 'n-length' bitstring one at a time.
    (All permutations are not stored in memory simultaneously).

    :param n: length of bit-string
    :return: i'th permutation.
    """
    num_permutations = 2 ** n
    for i in range(num_permutations):
        _binary_string_ = bin(i)[2:].zfill(n)
        yield _binary_string_


def _get_state_probabilities_(state_vector_: np.ndarray, reverse_states: bool = False) -> dict:
    """
    Calculate the probabilities of each basis state in a quantum state.

    Returns:
        dict: A dictionary containing the basis state as keys and their respective probabilities as values.
    """
    _state_vector_ = state_vector_
    _probs_ = {}
    for n, c_n in enumerate(_state_vector_):
        _state_string_ = represent_integer_with_bits(number=n, nr_bits=int(np.log2(len(_state_vector_))))
        if reverse_states:
            _state_string_ = _state_string_[::-1]
        _probs_[_state_string_] = np.power(np.linalg.norm(c_n), 2)
    return _probs_


def sparsity(matrix: np.ndarray) -> float:
    return 1.0 - np.sum(matrix != 0.0) / (matrix.shape[0] * matrix.shape[1])


def plot_histogram(result_dict: dict[str, float]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    x_labels = [r'|' + state + r'$\rangle$' for state in list(result_dict.keys())]

    x_positions = [0.3 * i for i in range(len(x_labels))]
    bars = ax.bar(x_positions, list(result_dict.values()), align='center', width=0.1)
    # Place the value of each bar above the respective bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                '{:.3f}'.format(height), ha='center', va='bottom')
    ax.set_ylabel('Probability')
    ax.set_ylim(-0.05, 1.1 * np.max(list(result_dict.values())))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=75)


def portfolio_ising(mu: np.ndarray, sigma: np.ndarray, alpha: float, k: int, n: int):
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

        max_cost_1, min_cost_1, min_comb = -np.inf, np.inf, np.empty(shape=(n,))
        for comb in generate_binary_combinations(n=n, k=k):
            comb_cost = cost(state=comb, mu=mu, sigma=sigma, alpha=alpha)
            if comb_cost < min_cost_1:
                min_cost_1, min_comb = comb_cost, comb
            if comb_cost > max_cost_1:
                max_cost_1 = comb_cost
        binary_comb = min_comb

        max_cost_2, min_cost_2, min_perm = -np.inf, np.inf, np.empty(shape=(n,))
        for perm in generate_binary_permutations(n=n):
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

    def get_qubo(mu: np.ndarray, sigma: np.ndarray, alpha: float, lmbda: float, k: int) -> tuple[np.ndarray, float]:
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

    def get_ising(Q: np.ndarray, offset: float):
        _Q_dict = {}
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                _Q_dict[(i, j)] = Q[i, j]

        _h_dict, _J_dict, _offset_ = qubo_to_ising(Q=_Q_dict, offset=offset)
        h, J = np.zeros_like(mu), np.zeros_like(sigma)
        for key in _h_dict.keys():
            h[key] = _h_dict[key]
        for key in _J_dict.keys():
            i, j = key
            J[i, j] = _J_dict[key]
        return J, h, _offset_

    n_chose_k, full, lmbda = min_cost_partition(nr_qubits=n, k=k, mu=mu, sigma=sigma, alpha=alpha)
    Q, Q_offset = get_qubo(mu=mu, sigma=sigma, alpha=alpha, lmbda=lmbda, k=k)
    J, h, J_offset = get_ising(Q=Q, offset=Q_offset)

    return n_chose_k, full, J, h, J_offset


def save_data_to_hdf(input_data: dict) -> None:
    max_number = max([int(f.split('.')[0]) for f in os.listdir(path='datasets_temp') if f.endswith('.hdf5')] + [0])
    with h5py.File(name='datasets_temp/' + f'{max_number + 1}' + '.hdf5', mode='w') as __file__:
        __file__.create_dataset(name='type', data=input_data['type'], dtype=int)
        __file__.create_dataset(name='N', data=input_data['N'], dtype=int)
        __file__.create_dataset(name='k', data=input_data['k'], dtype=int)
        __file__.create_dataset(name='layers', data=input_data['layers'], dtype=int)
        __file__.create_dataset(name='Max_cost', data=input_data['Max_cost'], dtype=np.float64)
        __file__.create_dataset(name='Min_cost', data=input_data['Min_cost'], dtype=np.float64)
        __file__.create_dataset(name='Min_cost_state', data=input_data['Min_cost_state'], dtype=np.float64)
        __file__.create_dataset(name='Normalized_cost', data=input_data['Normalized_cost'], dtype=np.float64)
        __file__.create_dataset(name='Final_circuit_sample_states', data=input_data['Final_circuit_sample_states'],
                                dtype=int)
        __file__.create_dataset(name='Final_circuit_sample_probabilities',
                                data=input_data['Final_circuit_sample_probabilities'], dtype=float)
        __file__.create_dataset(name='Expected_returns', data=input_data['Expected_returns'], dtype=np.float64)
        __file__.create_dataset(name='Covariances', data=input_data['Covariances'], dtype=np.float64)
        __file__.create_dataset(name='Optimizer_nfev', data=input_data['Optimizer_nfev'], dtype=int)
        __file__.create_dataset(name='Optimizer_maxfev', data=input_data['Optimizer_maxfev'], dtype=int)
        __file__.create_dataset(name='Rng_seed', data=input_data['Rng_seed'], dtype=int)
    __file__.close()


def portfolio_metrics(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    expected_returns = np.random.uniform(low=0, high=0.99, size=n)
    _temp_ = np.random.uniform(low=0, high=0.99, size=(n, n))
    covariances = np.dot(_temp_, _temp_.transpose())
    if not np.all(covariances == covariances.T) or not np.all(np.linalg.eigvals(covariances) >= 0):
        raise ValueError('Covariance matrix is not PSD.')

    return expected_returns, covariances
