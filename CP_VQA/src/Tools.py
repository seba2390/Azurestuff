from typing import List, Tuple, Dict

from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.sparse import csc_matrix, kron, identity

import numpy as np


##########################################
# ---------- HELPER FUNCTIONS ---------- #
##########################################

def string_to_array(string_rep: str) -> np.ndarray:
    return np.array([int(bit) for bit in string_rep]).astype(np.float32)


def array_to_string(array: np.ndarray) -> str:
    return ''.join(str(int(bit)) for bit in array)


def qubo_cost(state: np.ndarray, QUBO_matrix: np.ndarray) -> float:
    return np.dot(state, np.dot(QUBO_matrix, state))


def generate_binary_permutations(n: int) -> np.ndarray:
    """ Generates all the 2^n permutations of bitstring w. length 'n'. """
    num_permutations = 2 ** n
    for i in range(num_permutations):
        _binary_string_ = bin(i)[2:].zfill(n)
        yield np.array([int(bit) for bit in _binary_string_])


def portfolio_cost(state: np.ndarray, mu: np.ndarray, sigma: np.ndarray, alpha: float) -> float:
    return -np.dot(state, mu) + alpha * np.dot(state, np.dot(sigma, state))


def full_qubo_cost(state: np.ndarray, QUBO_matrix: np.ndarray, QUBO_offset: float) -> float:
    return np.dot(state, np.dot(QUBO_matrix, state)) + QUBO_offset


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
    max_cost_1, min_cost_1, min_comb, max_comb = -np.inf, np.inf, np.empty(shape=(nr_qubits,)), np.empty(
        shape=(nr_qubits,))
    max_cost_2, min_cost_2, min_perm, max_perm = -np.inf, np.inf, np.empty(shape=(nr_qubits,)), np.empty(
        shape=(nr_qubits,))
    for perm in generate_binary_permutations(n=nr_qubits):
        cost = portfolio_cost(state=perm, mu=mu, sigma=sigma, alpha=alpha)
        if cost < min_cost_2:
            min_cost_2, min_perm = cost, perm
        if cost > max_cost_2:
            max_cost_2 = cost
        if np.sum(perm) == k:
            if cost < min_cost_1:
                min_cost_1, min_comb = cost, perm
            if cost > max_cost_1:
                max_cost_1, max_comb = cost, perm

    _lmbda_ = 0
    if min_cost_2 < min_cost_1:
        _lmbda_ = abs(min_cost_1 - min_cost_2)

    _constrained_result_ = {'s_min': min_comb, 's_max': max_comb,
                            'c_min': min_cost_1, 'c_max': max_cost_1}
    _full_result_ = {'s_min': min_perm, 's_max': max_perm,
                     'c_min': min_cost_2, 'c_max': max_cost_2}
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


def normalized_cost(state: np.ndarray,
                    QUBO_matrix: np.ndarray,
                    QUBO_offset: float,
                    max_cost: float,
                    min_cost: float) -> float:
    """ Calculates the QUBO cost of the single most probable state in the
    result state dict, and normalizes it wrt. min and max possible cost."""
    found_cost = qubo_cost(state, QUBO_matrix) + QUBO_offset
    return abs(found_cost - min_cost) / abs(max_cost - min_cost)


def check_qubo(QUBO_matrix: np.ndarray,
               QUBO_offset: float,
               expected_returns: np.ndarray,
               covariances: np.ndarray,
               alpha: float,
               k: int):
    """ Runs through all permutations and checks that QUBO cost is equivalent to
    portfolio cost """

    N_QUBITS = QUBO_matrix.shape[0]
    for state in generate_binary_permutations(n=N_QUBITS):
        QUBO_cost = full_qubo_cost(state=state, QUBO_matrix=QUBO_matrix, QUBO_offset=QUBO_offset)
        PORTFOLIO_cost = portfolio_cost(state=state, mu=expected_returns, sigma=covariances, alpha=alpha)
        if np.sum(state) == k:
            if not np.isclose(QUBO_cost, PORTFOLIO_cost):
                raise ValueError(
                    f'state={"|" + "".join([str(_) for _ in state]) + ">"}, QUBO: {QUBO_cost}, PORTFOLIO: {PORTFOLIO_cost}')


def qubo_limits(Q: np.ndarray, offset: float):
    """Calculates the max and the min cost of the given qubo (and offset),
    together with the corresponding states."""

    N_QUBITS = Q.shape[0]
    min_qubo_cost, max_qubo_cost = np.inf, -np.inf
    min_qubo_state, max_qubo_state = None, None
    for state in generate_binary_permutations(n=N_QUBITS):
        c = full_qubo_cost(state=state, QUBO_matrix=Q, QUBO_offset=offset)
        if c < min_qubo_cost:
            min_qubo_cost = c
            min_qubo_state = state
        elif c > max_qubo_cost:
            max_qubo_cost = c
            max_qubo_state = state
    return {'c_min': min_qubo_cost, 'c_max': max_qubo_cost,
            'min_state': min_qubo_state, 'max_state': max_qubo_state}


def generate_string_representation(gate_names: List[str],
                                   qubit_indices: List[int],
                                   N_qubits: int) -> str:
    if len(gate_names) != len(qubit_indices):
        raise ValueError("Length of gate_names and qubit_indices should be the same...")
    if len(qubit_indices) != len(list(set(qubit_indices))):
        raise ValueError("qubit_indices should only contain unique integers...")
    if N_qubits <= 0 or N_qubits < len(qubit_indices):
        raise ValueError("N_qubits should be greater than 0 and at least the value of the size og qubit_indices...")
    counter = 0
    str_repr = []
    for q_i in range(N_qubits):
        if q_i in qubit_indices:
            str_repr.append(gate_names[counter])
            counter += 1
        else:
            str_repr.append('I')
    return ''.join(gate_name for gate_name in str_repr)


I = identity(2, format='csc', dtype=np.complex64)
X = csc_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex64))
Y = csc_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex64))
Z = csc_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex64))
gate_map = {'X': X, 'Y': Y, 'Z': Z, 'I': I}


def get_full_hamiltonian_matrix(indices: List[Tuple[int, int]],
                                angles: List[float],
                                N_qubits: int,
                                with_z_phase: bool = False) -> np.ndarray:
    terms = []
    for (qubit_i, qubit_j), theta_ij in zip(indices, angles[:len(indices)]):
        x_str = generate_string_representation(gate_names=['X', 'X'],
                                               qubit_indices=[qubit_i, qubit_j],
                                               N_qubits=N_qubits)
        y_str = generate_string_representation(gate_names=['Y', 'Y'],
                                               qubit_indices=[qubit_i, qubit_j],
                                               N_qubits=N_qubits)
        x_gates, y_gates = [gate_map[gate] for gate in x_str[::-1]], [gate_map[gate] for gate in y_str[::-1]]
        H_xx, H_yy = x_gates[0], y_gates[0]
        for x_gate, y_gate in zip(x_gates[1:], y_gates[1:]):
            H_xx = kron(H_xx, x_gate)
            H_yy = kron(H_yy, y_gate)
        H_ij = float(theta_ij) * (H_xx + H_yy)
        terms.append(H_ij)
    if with_z_phase:
        for qubit_i, theta_i in zip(list(range(N_qubits)), angles[len(angles):]):
            z_str = generate_string_representation(gate_names=['Z'],
                                                   qubit_indices=[qubit_i],
                                                   N_qubits=N_qubits)
            z_gates = [gate_map[gate] for gate in z_str[::-1]]
            H_z = z_gates[0]
            for z_gate in z_gates[1:]:
                H_z = kron(H_z, z_gate)

            H_i = float(theta_i) * H_z
            terms.append(H_i)
    H = terms[0]
    for term in terms[1:]:
        H += term
    return H.todense()


def get_qiskit_hamiltonian(indices: List[Tuple[int, int]],
                           angles: List[float],
                           N_qubits: int,
                           with_z_phase: bool = False) -> SparsePauliOp:
    coeffs, terms = [], []
    for (qubit_i, qubit_j), theta_ij in zip(indices, angles[:len(indices)]):
        H_xx_str = generate_string_representation(gate_names=['X', 'X'],
                                                  qubit_indices=[qubit_i, qubit_j],
                                                  N_qubits=N_qubits)[::-1]
        terms.append(H_xx_str)
        coeffs.append(float(theta_ij))
        H_yy_str = generate_string_representation(gate_names=['Y', 'Y'],
                                                  qubit_indices=[qubit_i, qubit_j],
                                                  N_qubits=N_qubits)[::-1]
        terms.append(H_yy_str)
        coeffs.append(float(theta_ij))
    if with_z_phase:
        for qubit_i, theta_i in zip(list(range(N_qubits)), angles[len(angles):]):
            H_z_str = generate_string_representation(gate_names=['Z'],
                                                     qubit_indices=[qubit_i],
                                                     N_qubits=N_qubits)[::-1]
            terms.append(H_z_str)
            coeffs.append(float(theta_i))

    return SparsePauliOp(data=terms, coeffs=coeffs)
