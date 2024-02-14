from typing import *

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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
                            Q_mat: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Given nr_nodes (length of bitstring), determines minimal cost
    and corresponding partition, for a QUBO cost function of type x^T*Q*x.

    :param nr_nodes: nr_nodes in graph - corresponding to length of bitstring
    :param Q_mat: square matrix used for QUBO cost
    :return: min_cost, min_perm: minimal cost, corresponding partition
    """

    def cost(state: np.ndarray, _Q_mat: np.ndarray) -> float:
        return state.T @ _Q_mat @ state

    min_cost, min_perm = np.inf, np.empty(shape=(nr_nodes,))
    for perm in generate_bit_string_permutations(n=nr_nodes):
        perm_cost = cost(state=perm, _Q_mat=Q_mat)
        if perm_cost < min_cost:
            min_cost, min_perm = perm_cost, perm

    return min_cost, min_perm


def get_qubo(size: int, edges: Union[List[Tuple[int, int, float]], dict[Tuple[int, int], float]],
             mat_type: str = 'UT') -> np.ndarray:
    """
    :param size: number of nodes in graph.
    :param edges: list of edges w. corresponding weight.
    :return: Upper triangular matrix of QUBO model, i.e. Q in x^TQx.

    """
    if mat_type not in ['UT', 'SYM']:
        raise ValueError("mat_type should be one of: ", ['UT', 'SYM'])

    _Q = np.zeros(shape=(size, size), dtype=float)

    if isinstance(edges, list):
        sorted_edges = []
        for edge in edges:
            if edge[0] > edge[1]:
                sorted_edges.append((edge[1], edge[0], edge[2]))
            else:
                sorted_edges.append((edge[0], edge[1], edge[2]))
        for _i, _j, _w in sorted_edges:
            if mat_type == 'UT':
                _Q[_i, _j] += 2 * _w
            else:
                _Q[_i, _j] += _w
                _Q[_j, _i] += _w

            _Q[_i, _i] -= 1 * _w
            _Q[_j, _j] -= 1 * _w
    elif isinstance(edges, dict):
        sorted_edges = {}
        for key, val in edges.items():
            if key[0] > key[1]:
                sorted_edges[(key[1], key[0])] = val
            else:
                sorted_edges[(key[0], key[1])] = val
        for key, _w in sorted_edges.items():
            _i, _j = key
            if mat_type == 'UT':
                _Q[_i, _j] += 2 * _w
            else:
                _Q[_i, _j] += _w
                _Q[_j, _i] += _w

            _Q[_i, _i] -= 1 * _w
            _Q[_j, _j] -= 1 * _w
    else:
        raise ValueError(f'edges should by of type: Union[List[Tuple[int, int, float]], dict[Tuple[int, int], float]]')
    return _Q


def ising_to_qubo(h, J, offset=0.0):
    """Convert an Ising problem to a QUBO problem.

    Map an Ising model defined on spins (variables with {-1, +1} values) to quadratic
    unconstrained binary optimization (QUBO) formulation :math:`x'  Q  x` defined over
    binary variables (0 or 1 values), where the linear term is contained along the diagonal of Q.
    Return matrix Q that defines the model as well as the offset in energy between the two
    problem formulations:

    .. math::

         s'  J  s + h'  s = offset + x'  Q  x

    See :meth:`~dimod.utilities.qubo_to_ising` for the inverse function.

    Args:
        h (dict[variable, bias]):
            Linear biases as a dict of the form {v: bias, ...}, where keys are variables of
            the model and values are biases.
        J (dict[(variable, variable), bias]):
           Quadratic biases as a dict of the form {(u, v): bias, ...}, where keys
           are 2-tuples of variables of the model and values are quadratic biases
           associated with the pair of variables (the interaction).
        offset (numeric, optional, default=0):
            Constant offset to be applied to the energy. Default 0.

    Returns:
        (dict, float): A 2-tuple containing:

            dict: QUBO coefficients.

            float: New energy offset.

    Examples:
        This example converts an Ising problem of two variables that have positive
        biases of value 1 and are positively coupled with an interaction of value 1
        to a QUBO problem and prints the resulting energy offset.

        >>> h = {1: 1, 2: 1}
        >>> J = {(1, 2): 1}
        >>> dimod.ising_to_qubo(h, J, 0.5)[1]
        -0.5

    """
    # the linear biases are the easiest
    q = {(v, v): 2. * bias for v, bias in h.items()}

    # next the quadratic biases
    for (u, v), bias in J.items():
        if bias == 0.0:
            continue
        q[(u, v)] = 4. * bias
        q[(u, u)] = q.setdefault((u, u), 0) - 2. * bias
        q[(v, v)] = q.setdefault((v, v), 0) - 2. * bias

    # finally calculate the offset
    offset += sum(J.values()) - sum(h.values())

    return q, offset


def qubo_to_ising(Q, offset=0.0):
    """Convert a QUBO problem to an Ising problem.

    Map a quadratic unconstrained binary optimization (QUBO) problem :math:`\vec{x}^T  Q  x`
    defined over binary variables (0 or 1 values), where the linear term is contained along
    the diagonal of Q, to an Ising model defined on spins (variables with {-1, +1} values).
    Return h and J that define the Ising model as well as the offset in energy
    between the two problem formulations:

    .. math::

         x^T  Q  x  = offset + s^T  J  s + h^T  s

    Args:
        Q (dict[(variable, variable), coefficient]):
            QUBO coefficients in a dict of form {(u, v): coefficient, ...}, where keys
            are 2-tuples of variables of the model and values are biases
            associated with the pair of variables. Tuples (u, v) represent interactions
            and (v, v) linear biases.
        offset (numeric, optional, default=0):
            Constant offset to be applied to the energy. Default 0.

    Returns:
        (dict, dict, float): A 3-tuple containing:

            dict: Linear coefficients of the Ising problem.

            dict: Quadratic coefficients of the Ising problem.

            float: New energy offset.

    Examples:
        This example converts a QUBO problem of two variables that have positive
        biases of value 1 and are positively coupled with an interaction of value 1
        to an Ising problem, and shows the new energy offset.

        >>> Q = {(1, 1): 1, (2, 2): 1, (1, 2): 1}
        >>> qubo_to_ising(Q, 0.5)[2]
        1.75

    """
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


def generate_random_graph(nodes: int,
                          weighted: bool = False,
                          w_min: float = 1e-3,
                          seed: int = 0) -> List[Tuple[int, int, float]]:
    """
    Generates randomly connected graph of size 'nodes' possibly with random weights in (w_min;1].
    N.B. defaults to all weights = 1.0.

    :param nodes: nr of nodes in graph.
    :param weighted: boolean for having weighted or unweighted edges in graph.
    :param w_min: tolerance for lowest allowable weight in edge.
    :param seed: rng seed for numpy to give reproducibility.
    :return: list of [(head, tail, weight), ...]
    """
    np.random.seed(seed)
    edge_list = []
    avg_nr_edges = int(np.floor((nodes - 1) / 2 + 1))
    for node in range(nodes):
        N_edges = np.random.poisson(lam=avg_nr_edges, size=1)
        while N_edges == 0:  # To avoid node with no connections
            N_edges = np.random.poisson(lam=avg_nr_edges, size=1)
        head = node

        # Doing this to avoid producing multiple weights for same edge in edge_list, i.e. to avoid
        # instances of e.g. [... , (0,1,2.13) , (1,0,0.6) , ... ]
        if len(edge_list) > 0:
            _EDGES = [a.astype(int).tolist() for a in np.array(edge_list)[:, [0, 1]]]
            available_tails = [i for i in range(nodes) if
                               i != head and not ([head, i] in _EDGES or [i, head] in _EDGES)]
        else:
            available_tails = [i for i in range(nodes) if i != head]

        tails = np.random.choice(a=available_tails, replace=False, size=min(N_edges, len(available_tails)))
        for tail in tails:
            weight = np.random.uniform(low=w_min, high=1.0) if weighted else 1.0
            edge_list.append((head, tail, np.round(weight, 4)))
    return edge_list


def draw_graph(G: nx.Graph, colors: List[str], pos: Dict[int, tuple]) -> None:
    """
    Draw a weighted graph using NetworkX and Matplotlib.

    This function visualizes a weighted graph using NetworkX's drawing capabilities
    and Matplotlib for rendering.

    Parameters:
    -----------
    G : networkx.Graph
        The graph to be drawn.
    colors : list
        List of node colors corresponding to the nodes in the graph `G`.
    pos : dict
        A dictionary with nodes as keys and their positions as values, specifying
        the layout of the graph.

    Returns:
    --------
    None

    Notes:
    ------
    The function uses NetworkX's draw_networkx and draw_networkx_edge_labels
    functions for visualization. The graph is drawn with node colors, and edge
    weights are displayed as labels on the edges.

    Example:
    --------
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.Graph()
    G.add_edge(0, 1, weight=5)
    G.add_edge(1, 2, weight=7)
    G.add_edge(2, 0, weight=3)

    colors = ['r', 'g', 'b']
    pos = nx.spring_layout(G)

    draw_graph(G, colors, pos)
    plt.show()
    """
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)


def encode_Ising(Q: np.ndarray, g: np.ndarray = None, offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    _g = g
    if _g is None:
        _g = np.zeros(shape=(Q.shape[0], 1), dtype=float)

    J = 0.25 * Q

    h = np.zeros_like(_g)
    for i in range(Q.shape[0]):
        h -= (Q[:, i].reshape((Q.shape[0], 1)) + Q.T[:, i].reshape((Q.shape[0], 1))) * 0.25

    h -= 0.5 * _g

    const = np.sum(Q) * 0.25 + np.sum(_g) * 0.5 + offset

    return J, h, const

