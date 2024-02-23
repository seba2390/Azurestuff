from typing import Tuple, Union
from itertools import combinations

import numpy as np
from numba import njit, types
from numba.typed import List


@njit(fastmath=True)
def _fast_brute_force_min_(Q: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
    n = Q.shape[0]
    # initialize bit vector and value
    x = np.zeros(n).astype(np.float32)
    v = 0
    # initialize minimal bit vector and value x_min = np.zeros(n)
    x_min, x_max = np.zeros(n).astype(np.float32), np.zeros(n).astype(np.float32)
    v_min, v_max = 1e6, -1e6
    # Separate Q into quadratic and linear parts
    qua = np.triu(Q, 1)
    qua += qua.T
    lin = np.diag(Q)
    for k in range(1, 2 ** n):
        l = int(np.log2((k ^ (k - 1))))
        x[l] = 1 - x[l]
        delta = (2 * x[l] - 1) * (qua[l] @ x + lin[l])
        v += delta
        if v < v_min:
            x_min[:] = x
            v_min = v
        elif v > v_max:
            x_max[:] = x
            v_max = v

    extremum_state_zeroes, extremum_state_ones = np.zeros(n).astype(np.float32), np.ones(n).astype(np.float32)
    c_zeroes, c_ones = (np.dot(extremum_state_zeroes, np.dot(Q.astype(np.float32), extremum_state_zeroes)),
                        np.dot(extremum_state_ones, np.dot(Q.astype(np.float32), extremum_state_ones)))
    if c_zeroes < v_min:
        v_min = c_zeroes
        x_min = extremum_state_zeroes
    elif c_zeroes > v_max:
        v_max = c_zeroes
        x_max = extremum_state_zeroes
    if c_ones < v_min:
        v_min = c_ones
        x_min = extremum_state_ones
    elif c_ones > v_max:
        v_max = c_ones
        x_max = extremum_state_ones
    return x_min, v_min, x_max, v_max


@njit(fastmath=True)
def next_combination(subset):
    """
    Generate the next combination of n elements as bit positions
    in subset. subset is a bitmask of the current combination.
    """
    tmp = (subset | (subset - 1)) + 1
    return tmp | ((((tmp & -tmp) // (subset & -subset)) >> 1) - 1)


@njit(fastmath=True)
def generate_combinations(n, k):
    """
    Generates all combinations of n elements with k selected
    """
    subset = (1 << k) - 1  # The first combination with k ones
    while subset < 1 << n:
        yield subset
        subset = next_combination(subset)


# Now, let's use this function in the main code
@njit(fastmath=True)
def _fast_brute_force_min_cardinality(Q, k):
    # Initialize minimal and maximal bit vectors and values
    n = Q.shape[0]
    x_min, x_max = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    v_min, v_max = 1e6, -1e6

    # Iterate over each combination
    for subset in generate_combinations(n, k):
        x = np.zeros(n, dtype=np.float32)
        for j in range(n):
            if (subset >> j) & 1:
                x[j] = 1

        # Calculate the value v
        v = np.dot(x, np.dot(x, Q))

        # Update min and max values and vectors
        if v < v_min:
            x_min[:] = x
            v_min = v
        elif v > v_max:
            x_max[:] = x
            v_max = v

    return x_min, v_min, x_max, v_max


class Qubo:
    def __init__(self, Q: np.ndarray, offset: float) -> None:
        if not isinstance(Q, np.ndarray):
            raise ValueError(f'"Q" should be given as a numpy array but is: {type(Q)}.')
        if Q.shape[0] != Q.shape[1]:
            raise ValueError(f'Qubo matrix "Q" should be quadratic.')
        self.Q = Q
        self.offset = offset
        self.n = self.Q.shape[0]
        self.solver = _fast_brute_force_min_
        self.constrained_solver = _fast_brute_force_min_cardinality

        # Transform QUBO matrix to upper-triangular if on symmetric form
        if not np.all(self.Q == np.triu(self.Q)):
            self.Q = np.triu(self.Q)
            for row in range(self.Q.shape[0]):
                for col in range(row + 1, self.Q.shape[1]):
                    self.Q[row][col] += Q[col][row]
        self.Q = self.Q.astype(np.float32)

        self.subspace_c_min, self.subspace_c_max = None, None
        self.subspace_x_min, self.subspace_x_max = None, None

        self.full_space_c_min, self.full_space_c_max = None, None
        self.full_space_x_min, self.full_space_x_max = None, None

    def solve(self) -> dict:
        """ Returns the solution to the Qubo problem as:
        (x_min, v_min, x_max, v_max) """
        x_min, v_min, x_max, v_max = self.solver(self.Q)
        self.full_space_c_min, self.full_space_c_max = v_min, v_max
        self.full_space_x_min, self.full_space_x_max = x_min, x_max
        return {'x_min': x_min, 'v_min': v_min + self.offset,
                'x_max': x_max, 'v_max': v_max + self.offset}

    def solve_constrained(self, cardinality: int) -> dict:
        """ Returns the solution,s.t. ||x||_0 = cardinality, to the Qubo problem as:
        (x_min, v_min, x_max, v_max) """
        x_min, v_min, x_max, v_max = self.constrained_solver(self.Q, cardinality)
        self.subspace_c_min, self.subspace_c_max = v_min, v_max
        self.subspace_x_min, self.subspace_x_max = x_min, x_max
        return {'x_min': x_min, 'v_min': v_min + self.offset,
                'x_max': x_max, 'v_max': v_max + self.offset}
