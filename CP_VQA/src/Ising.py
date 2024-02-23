from typing import Tuple, List
from src.Qubo import Qubo


def qubo_to_ising(Q: dict, offset: float = 0.0) -> Tuple[dict, dict, float]:
    """
    Convert a Quadratic Unconstrained Binary Optimization (QUBO) problem to an Ising problem.

    The QUBO problem is defined on binary variables with values in {0,1}, while the Ising problem
    is defined on spin variables with values in {-1, +1}. This function maps a QUBO problem to its
    equivalent Ising problem.

    Parameters:
    - Q (dict): A dictionary representing the QUBO matrix where keys are tuples of variable indices
      (u, v) and values are the corresponding biases.
    - offset (float, optional): A constant offset value to be added to the Ising problem. Defaults to 0.0.

    Returns:
    - Tuple[dict, dict, float]: A tuple containing the linear biases (h), quadratic biases (J),
      and the total offset for the Ising problem.
    """
    h, J = {}, {}  # Linear and quadratic terms for Ising
    linear_offset, quadratic_offset = 0.0, 0.0

    # Iterate over QUBO matrix entries
    for (u, v), bias in Q.items():
        if u == v:
            # Handling linear terms
            h[u] = h.get(u, 0) + .5 * bias
            linear_offset += bias
        else:
            # Handling quadratic terms
            if bias != 0.0:
                J[(u, v)] = J.get((u, v), 0) + .25 * bias
            h[u] = h.get(u, 0) + .25 * bias
            h[v] = h.get(v, 0) + .25 * bias
            quadratic_offset += bias

    # Calculating total offset for Ising problem
    offset += .5 * linear_offset + .25 * quadratic_offset
    return h, J, offset


def get_ising(qubo: Qubo) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, float]]]:
    """
    Convert a Qubo object to its equivalent Ising problem representation.

    This function takes a Qubo object, extracts its Q matrix and offset, and then converts it
    to an Ising problem. The Ising problem is returned in a list format suitable for further processing.

    Parameters:
    - qubo (Qubo): A Qubo object containing the Q matrix and an offset.

    Returns:
    - Tuple[List[Tuple[int, int, float]], List[Tuple[int, float]]]: A tuple containing two lists,
      one for the quadratic biases (J) and one for the linear biases (h) of the Ising problem.
    """
    # Extract Q matrix and offset from the Qubo object
    Q, offset = qubo.Q, qubo.offset

    # Convert the Q matrix to a dictionary form
    _Q_dict = {}
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            _Q_dict[(i, j)] = Q[i, j]

    # Convert the QUBO problem to an Ising problem
    _h_dict, _J_dict, _offset_ = qubo_to_ising(Q=_Q_dict, offset=offset)

    # Transform Ising problem from dictionary to list format
    J_list = [(i, j, _J_dict[(i, j)]) for (i, j) in _J_dict.keys()]
    h_list = [(key, _h_dict[key]) for key in _h_dict.keys()]

    return J_list, h_list
