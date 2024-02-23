import pytest
import os

from typing import *
from src.Qubo import *

import numpy as np


######################################################################################################
#                                           HELPER FUNCTIONS                                         #
######################################################################################################

def generate_binary_combinations(n: int, k: int) -> np.ndarray:
    """ Generates all the 'n' chose 'k' combinations w. 'k' ones. """
    num_permutations = 2 ** n
    for indices in combinations(range(n), k):
        # Create a numpy array of zeros of size N
        arr = np.zeros(n, dtype=int)
        # Set ones at the specified positions
        arr[list(indices)] = 1
        yield arr


def generate_bit_string_permutations(n: int) -> np.ndarray:
    """
    A 'generator' type function that calculates all 2^n
    permutations of a 'n-length' bitstring one at a time.
    (All permutations are not stored in memory simultaneously).

    :param n: length of bit-string
    :return: i'th permutation.
    """
    num_permutations = 2 ** n
    for i in range(num_permutations):
        binary_string = bin(i)[2:].zfill(n)
        permutation = np.array([int(x) for x in binary_string])
        yield permutation


def qubo_min_cost_partition(nr_nodes: int,
                            Q_mat: np.ndarray) -> Tuple[List[int], float]:
    """
    Given nr_nodes (length of bitstring), determines minimal cost
    and corresponding partition, for a QUBO cost function of type x^T*Q*x.

    :param nr_nodes: nr_nodes in graph - corresponding to length of bitstring
    :param Q_mat: square matrix used for QUBO cost
    :return: min_cost, min_perm: minimal cost, corresponding partition
    """
    if nr_nodes != Q_mat.shape[0]:
        raise ValueError('Size of binary state:', nr_nodes, " dimensions of Q matrix: ", Q_mat.shape)

    def cost(state: np.ndarray, _Q_mat: np.ndarray) -> float:
        return float(np.matmul(state, np.matmul(_Q_mat, state)))

    min_cost, min_perm = np.inf, np.empty(shape=(nr_nodes,))
    for perm in generate_bit_string_permutations(n=nr_nodes):
        perm_cost = cost(state=perm, _Q_mat=Q_mat)
        if perm_cost < min_cost:
            min_cost, min_perm = perm_cost, perm

    return min_perm.tolist(), min_cost


def constrained_qubo_min_cost_partition(nr_nodes: int,
                                        cardinality: int,
                                        Q_mat: np.ndarray) -> Tuple[List[int], float]:
    """
    Given nr_nodes (length of bitstring), determines minimal cost
    and corresponding partition, for a QUBO cost function of type x^T*Q*x.
    w. cardinality of x = cardinality.

    :param nr_nodes: nr_nodes in graph - corresponding to length of bitstring
    :param Q_mat: square matrix used for QUBO cost
    :return: min_cost, min_perm: minimal cost, corresponding partition
    """
    if nr_nodes != Q_mat.shape[0]:
        raise ValueError('Size of binary state:', nr_nodes, " dimensions of Q matrix: ", Q_mat.shape)

    def cost(state: np.ndarray, _Q_mat: np.ndarray) -> float:
        return float(np.matmul(state, np.matmul(_Q_mat, state)))

    min_cost, min_perm = np.inf, np.empty(shape=(nr_nodes,))
    for perm in generate_binary_combinations(n=nr_nodes, k=cardinality):
        perm_cost = cost(state=perm, _Q_mat=Q_mat)
        if perm_cost < min_cost:
            min_cost, min_perm = perm_cost, perm

    return min_perm.tolist(), min_cost


######################################################################################################
#                                     TEST GENERATOR FUNCTIONS                                       #
######################################################################################################


def generate_qubo_test(rng_trials: int,
                       max_size: int = 10) -> List[Tuple[list[int], list[int], float, float]]:
    tests = []
    for size in range(3, max_size + 1):
        for trial in range(rng_trials):
            # Generating random matrix
            Q = np.triu(np.random.uniform(low=-1.0, high=1.0, size=(size, size)))

            # Generating tests
            res = Qubo(Q=Q.astype(np.float32)-0.000001*np.eye(Q.shape[0]), offset=0.0).solve()
            x1, v1 = res['x_min'], res['v_min']
            x2, v2 = qubo_min_cost_partition(nr_nodes=size, Q_mat=Q.astype(np.float32))
            tests.append((x1.astype(int).tolist(), x2, v1, v2))

    return tests


def generate_constrained_qubo_test(rng_trials: int,
                                   max_size: int = 8) -> List[Tuple[list[int], list[int], float, float, int]]:
    tests = []
    for size in range(3, max_size + 1):
        for k in range(1, size // 2 + 1):
            for trial in range(rng_trials):
                # Generating random matrix
                Q = np.triu(np.random.uniform(low=-1.0, high=1.0, size=(size, size)))

                # Generating tests
                res = Qubo(Q=Q.astype(np.float32), offset=0.0).solve_constrained(cardinality=k)
                x1, v1 = res['x_min'], res['v_min']
                x2, v2 = constrained_qubo_min_cost_partition(nr_nodes=size, cardinality=k, Q_mat=Q.astype(np.float32))
                tests.append((x1.astype(int).tolist(), x2, v1, v2, k))

    return tests


##############################################################################################
#                                            TESTS                                           #
##############################################################################################

tests_1 = generate_qubo_test(rng_trials=10)
tests_2 = generate_constrained_qubo_test(rng_trials=10)


@pytest.mark.parametrize('fast_x, correct_x, fast_v, correct_v', tests_1, )
def test_fast_qubo_bruteforce_1(fast_x: List[int],
                                correct_x: List[int],
                                fast_v: float,
                                correct_v: float):
    assert fast_x == correct_x
    assert np.isclose(fast_v, correct_v, atol=1e-5)


@pytest.mark.parametrize('fast_x, correct_x, fast_v, correct_v, k', tests_2, )
def test_fast_qubo_bruteforce_2(fast_x: List[int],
                                correct_x: List[int],
                                fast_v: float,
                                correct_v: float,
                                k: int):
    assert fast_x == correct_x
    assert np.isclose(fast_v, correct_v, atol=1e-5)
    assert np.sum(fast_x) == k
