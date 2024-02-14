from typing import List, Tuple, Dict, Union
from itertools import combinations

from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit.quantum_info.operators import Pauli
# from qiskit.opflow import X, Y, Z, I
from scipy.sparse import csc_matrix, csr_matrix, kron, identity

import numpy as np
from numba import jit

from src.TorchQcircuit import *


##########################################
# ---------- HELPER FUNCTIONS ---------- #
##########################################

def string_to_array(string_rep: str) -> np.ndarray:
    return np.array([int(bit) for bit in string_rep]).astype(np.float32)


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
                raise ValueError(
                    f'state={"|" + "".join([str(_) for _ in state]) + ">"}, QUBO: {QUBO_cost}, PORTFOLIO: {PORTFOLIO_cost}')


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


def plot_energy_spectrum(QUBO_matrix: np.ndarray, QUBO_offset: float, k: int):
    def qubo_cost(state: np.ndarray, QUBO_matrix: np.ndarray, QUBO_offset: float) -> float:
        return np.dot(state, np.dot(QUBO_matrix, state)) + QUBO_offset

    def generate_binary_combinations(n: int, k: int) -> np.ndarray:
        """ Generates all the 'n' chose 'k' combinations w. 'k' ones. """
        num_permutations = 2 ** n
        for indices in combinations(range(n), k):
            # Create a numpy array of zeros of size N
            arr = np.zeros(n, dtype=int)
            # Set ones at the specified positions
            arr[list(indices)] = 1
            yield arr

    __cost_state_dict__ = {}

    for state in generate_binary_combinations(n=QUBO_matrix.shape[0], k=k):
        str_rep = '|' + ''.join(str(_) for _ in state.flatten()) + '>'
        __cost_state_dict__[str_rep] = qubo_cost(state=state, QUBO_matrix=QUBO_matrix, QUBO_offset=QUBO_offset)

    return __cost_state_dict__


def partitioned_averages(unsorted_list: List[List[Union[int, float]]]) -> Tuple[List[float], List[float]]:
    # Sort the list of lists by their length (shortest first)
    sorted_list = sorted(unsorted_list, key=len)

    partitioned_avgs = []  # To store the partitioned averages
    partitioned_std_devs = []  # To store the partitioned std. devs.

    # Iterate over each sublist
    for l1 in range(len(sorted_list)):
        l1_length = len(sorted_list[l1])  # Length of the current sublist

        # Add corresponding elements from longer sublists and calculate partial averages
        _avg_ = [sorted_list[l1]]
        for l2 in range(l1 + 1, len(sorted_list)):
            _avg_.append(sorted_list[l2][:l1_length])
        _std_dev_ = np.std(np.array(_avg_).astype(float), axis=0)
        _avg_ = np.mean(np.array(_avg_).astype(float), axis=0)
        # Append the new elements of the average to the result
        if l1 > 0:
            partitioned_avgs.append(_avg_[len(sorted_list[l1 - 1]):])
            partitioned_std_devs.append(_std_dev_[len(sorted_list[l1 - 1]):])
        else:
            partitioned_avgs.append(_avg_)
            partitioned_std_devs.append(_std_dev_)

    # Combine all partitioned averages into a single list
    combined_avgs = []
    for avg_part in partitioned_avgs:
        combined_avgs += avg_part.tolist()
    combined_std_devs = []
    for std_dev_part in partitioned_std_devs:
        combined_std_devs += std_dev_part.tolist()

    return combined_avgs, combined_std_devs


def create_state_vector(state_str: str, probability: float) -> np.ndarray:
    # Create a zero vector with length 2^N
    state_vector = np.zeros(2 ** len(state_str), dtype=np.float64)
    # Set the amplitude for the specified state
    state_vector[int(state_str, 2)] = np.sqrt(probability)
    return state_vector


def create_operator(Q: np.ndarray):
    N = Q.shape[0]

    def generate_binary_permutations(n: int) -> Tuple[np.ndarray, str]:
        """ Generates all the 2^n permutations of bitstring w. length 'n'. """
        num_permutations = 2 ** n
        for i in range(num_permutations):
            _binary_string_ = bin(i)[2:].zfill(n)
            yield np.array([int(bit) for bit in _binary_string_]), _binary_string_

    def qubo_cost(state: np.ndarray, QUBO_matrix: np.ndarray) -> float:
        return np.dot(state, np.dot(QUBO_matrix, state))

    operator = np.zeros(shape=(2 ** N, 2 ** N), dtype=np.float64)
    for (array_perm, string_perm) in generate_binary_permutations(n=N):
        E_i = qubo_cost(state=array_perm, QUBO_matrix=Q)
        state_vector = create_state_vector(state_str=string_perm, probability=1.0)
        operator += E_i * np.outer(state_vector, state_vector)
    return operator


def get_generator(i: int, j: int, theta: float, N: int, flip: bool = False) -> np.ndarray:
    if i == 0 or j == 0:
        res_x, res_y = 'X', 'Y'
        for qubit_idx in range(1, N):
            if j == qubit_idx or i == qubit_idx:
                res_x += 'X'
                res_y += 'Y'
            else:
                res_x += 'I'
                res_y += 'I'
    else:
        res_x, res_y = 'I', 'I'
        for qubit_idx in range(1, N):
            if j == qubit_idx or i == qubit_idx:
                res_x += 'X'
                res_y += 'Y'
            else:
                res_x += 'I'
                res_y += 'I'
    if flip:
        theta * (np.array(Operator(Pauli(res_x[::-1]))) + np.array(Operator(Pauli(res_y[::-1]))))
    return theta * (np.array(Operator(Pauli(res_x))) + np.array(Operator(Pauli(res_y))))


def operator_expectation(O: np.ndarray, probability_dict: dict):
    vals = []
    for (binary_state_str, probability) in probability_dict.items():
        state_vector = create_state_vector(state_str=binary_state_str, probability=probability)
        vals.append(state_vector.T.conj() @ (O @ state_vector))
    return np.mean(vals)


def get_qiskit_H(Q: np.ndarray):
    """ Generates H = \sum_ij q_ij(I_i-Z_i)/2(I_j-Z_j)/2 """

    def get_ij_term(i: int, j: int, Q: np.ndarray) -> List[Tuple[str, float]]:
        N = Q.shape[0]
        I_term = ''.join('I' for qubit_idx in range(N))
        Z_i_term = ''.join('Z' if qubit_idx == i else 'I' for qubit_idx in range(N))
        Z_j_term = ''.join('Z' if qubit_idx == i else 'I' for qubit_idx in range(N))
        # Pauli matrices are idempotent: x^2=y^2=z^2=I
        if i == j:
            Z_ij_term = I_term
        else:
            Z_ij_term = ''.join('Z' if qubit_idx == i or qubit_idx == j else 'I' for qubit_idx in range(N))
        total_ij_term = [(I_term, Q[i, j] / 4), (Z_i_term, -Q[i, j] / 4), (Z_j_term, -Q[i, j] / 4),
                         (Z_ij_term, Q[i, j] / 4)]
        return total_ij_term

    H = []
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            H += get_ij_term(i, j, Q)
    return SparsePauliOp.from_list(H)


I = identity(2, format='csc', dtype=np.complex64)
X = csc_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex64))
Y = csc_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex64))
Z = csc_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex64))
gate_map = {'X': X, 'Y': Y, 'Z': Z, 'I': I}


def get_full_hamiltonian(indices: List[Tuple[int, int]], angles: List[float], N_qubits: int,
                         with_z_phase: bool = False):
    terms = []
    for (qubit_i, qubit_j), theta_ij in zip(indices, angles[:len(indices)]):
        x_str = generate_string_representation(gate_name='X',
                                               qubit_i=qubit_i,
                                               qubit_j=qubit_j,
                                               N=N_qubits)
        y_str = generate_string_representation(gate_name='Y',
                                               qubit_i=qubit_i,
                                               qubit_j=qubit_j,
                                               N=N_qubits)
        x_gates, y_gates = [gate_map[gate] for gate in x_str[::-1]], [gate_map[gate] for gate in y_str[::-1]]
        H_xx, H_yy = x_gates[0], y_gates[0]
        for x_gate, y_gate in zip(x_gates[1:], y_gates[1:]):
            H_xx = kron(H_xx, x_gate)
            H_yy = kron(H_yy, y_gate)
        H_ij = float(theta_ij) * (H_xx + H_yy)
        terms.append(H_ij)
    if with_z_phase:
        for qubit_i, theta_i in zip(list(range(N_qubits)), angles[len(angles):]):
            z_str = generate_string_representation_single(gate_name='Z',
                                                          qubit_i=qubit_i,
                                                          N=N_qubits)
            z_gates = [gate_map[gate] for gate in z_str[::-1]]
            H_z = z_gates[0]
            for z_gate in z_gates[1:]:
                H_z = kron(H_z, z_gate)

            H_i = float(theta_i) * H_z
            terms.append(H_i)
    H = terms[0]
    for term in terms[1:]:
        H += term
    return Operator(H.todense())


def get_normal_H(Q: np.ndarray, flip: bool = False) -> np.ndarray:
    """ Generates H = \sum_ij q_ij(I_i-Z_i)/2(I_j-Z_j)/2
    If flip is True, the same convention for indexing as in Qiskit is assumed
    where indexing is done right -> left in bitstring.
    """
    format = 'csr'
    single_idx_terms_cache = {}
    I_term = identity(2 ** Q.shape[0], format=format, dtype=np.float32)
    Z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.float32))

    def get_term(i: int):
        N = Q.shape[0]
        if i not in list(single_idx_terms_cache.keys()):
            if i == N - 1:
                _mat_rep_ = Z
                _after_I_ = identity(2 ** (N - 1), format=format)
                _mat_rep_ = kron(_mat_rep_, _after_I_)
            else:
                _before_I_ = identity(2 ** (N - i - 1), format=format)
                _mat_rep_ = kron(_before_I_, Z)
                _after_I_ = identity(2 ** i, format=format)
                _mat_rep_ = kron(_mat_rep_, _after_I_)
            single_idx_terms_cache[i] = csr_matrix(_mat_rep_)
        return single_idx_terms_cache[i]

    def get_ij_term(i: int, j: int, Q: np.ndarray) -> csr_matrix:
        N = Q.shape[0]
        if flip:
            Z_i_term = get_term(i=N - i - 1)
            if i == j:
                Z_ij_term = I_term
                Z_j_term = Z_i_term
            else:
                Z_j_term = get_term(i=N - j - 1)
                Z_ij_term = get_term(i=N - i - 1) @ get_term(i=N - j - 1)
            return Q[i, j] / 4.0 * (I_term - Z_i_term - Z_j_term + Z_ij_term)

        Z_i_term = get_term(i=i)
        if i == j:
            Z_ij_term = I_term
            Z_j_term = Z_i_term
        else:
            Z_j_term = get_term(i=j)
            Z_ij_term = get_term(i=i) @ get_term(i=j)
        return Q[i, j] / 4.0 * (I_term - Z_i_term - Z_j_term + Z_ij_term)

    H = csr_matrix(np.zeros(shape=(2 ** Q.shape[0], 2 ** Q.shape[1]), dtype=np.float32))
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            H += get_ij_term(i, j, Q)
    return np.array(H.todense())
