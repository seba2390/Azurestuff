from typing import List, Tuple

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import X, Y
from qiskit.quantum_info import state_fidelity

import numpy as np
from numba import jit


class Grid:
    def __init__(self,
                 N_qubits: int = None,
                 Rows: int = None,
                 Cols: int = None):

        self.using_N_qubits = False
        if N_qubits is None:
            if Rows is None or Cols is None:
                raise ValueError(
                    'When the grid is not initialized using "N_qubits", it should be initialized, using "Rows" and '
                    '"Cols".')
        else:
            if int(np.sqrt(N_qubits)) - np.sqrt(N_qubits) != 0:
                raise ValueError(
                    f'When grid is initialized w. "N_qubits" it is assumed to be a square grid, and therefore '
                    f'"N_qubits" must be a perfect square integer. ')
            self.N_qubits = N_qubits
            self.using_N_qubits = True
        if Rows is not None or Cols is not None:
            if N_qubits is not None:
                raise ValueError(f'When specifying the grid using "Rows" & "Cols", "N_qubits" should not be specified.')
            if Rows is None or Cols is None:
                raise ValueError(
                    f'When specifying the grid without "N_qubits", both "Rows" & "Cols" has to be specified.')
        else:
            self.rows, self.cols = Rows, Cols

    def get_grid_indexing(self) -> np.ndarray:
        """ Here we assume a Hamiltonian path from upper right corner to lower left corner
         ala L -> R, U -> D, R -> L ... L -> R. """
        if self.using_N_qubits:
            root = int(np.sqrt(self.N_qubits))
            return np.array([[col + row * root for col in range(root)] for row in range(root)])
        else:
            return np.array([[col + row * self.cols for col in range(self.cols)] for row in range(self.rows)])

    def get_NN_indices(self) -> List[Tuple[int, int]]:
        """ Returns pairs of indices corresponding to
        Nearest Neighbor interactions in the grid structure """
        grid_indices = self.get_grid_indexing()
        rows, cols = grid_indices.shape
        NN_pairs = []
        for row in range(rows):
            for col in range(cols):
                if row == rows - 1:
                    if col < cols - 1:
                        NN_pairs.append((grid_indices[row, col], grid_indices[row, col + 1]))
                elif col == cols - 1:
                    NN_pairs.append((grid_indices[row, col], grid_indices[row + 1, col]))
                else:
                    NN_pairs.append((grid_indices[row, col], grid_indices[row + 1, col]))
                    NN_pairs.append((grid_indices[row, col], grid_indices[row, col + 1]))
        return NN_pairs

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


def string_to_array(string_rep: str) -> np.ndarray:
    return np.array([int(bit) for bit in string_rep]).astype(np.float64)


@jit(nopython=True, cache=True)
def qubo_cost(state: np.ndarray, QUBO_matrix: np.ndarray) -> float:
    return np.dot(state, np.dot(QUBO_matrix, state))


class QAOA:
    def __init__(self, N_qubits, layers, QUBO_matrix, QUBO_offset):
        self.n_qubits = N_qubits
        self.layers = layers
        self.QUBO_matrix = QUBO_matrix
        self.J_list, self.h_list = get_ising(Q=QUBO_matrix, offset=QUBO_offset)
        self.simulator = Aer.get_backend('statevector_simulator')

    def set_circuit(self, angles):

        gamma = angles[self.layers:]
        beta = angles[:self.layers]

        qcircuit = QuantumCircuit(self.n_qubits)

        # Initial state: Hadamard gate on each qubit
        for qubit_index in range(self.n_qubits):
            qcircuit.h(qubit_index)

        # For each Cost, Mixer repetition
        for layer in range(self.layers):

            # ------ Cost unitary: ------ #
            # Weighted RZZ gate for each edge
            for qubit_i, qubit_j, J_ij in self.J_list:
                qcircuit.rzz(2 * gamma[layer] * J_ij, qubit_i, qubit_j)

            # Weighted RZ gate for each qubit
            for qubit_i, h_i in self.h_list:
                qcircuit.rz(2 * gamma[layer] * h_i, qubit_i)

            # ------ Mixer unitary: ------ #
            # Mixer unitary: Weighted X rotation on each qubit
            for qubit_i in range(self.n_qubits):
                qcircuit.rx(2 * beta[layer], qubit_i)

        return qcircuit

    def get_cost(self, angles) -> float:
        circuit = self.set_circuit(angles=angles)
        counts = execute(circuit, self.simulator).result().get_counts()
        return np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO_matrix) for
                        bitstring, probability in counts.items()])

    def get_state_probabilities(self, angles, flip_states: bool = True) -> dict:
        circuit = self.set_circuit(angles=angles)
        counts = execute(circuit, self.simulator).result().get_counts()
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring[::-1]: probability for bitstring, probability in counts.items()}


class CP_QAOA:
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 QUBO_matrix,
                 QUBO_offset,
                 grid = None,
                 with_evenly_distributed_start_x: bool = False,
                 with_next_nearest_neighbors: bool = False,
                 with_z_phase: bool = False):
        self.n_qubits = N_qubits
        self.cardinality = cardinality
        self.layers = layers
        self.QUBO_matrix = QUBO_matrix

        self.with_evenly_distributed_start_x = with_evenly_distributed_start_x
        self.with_next_nearest_neighbors = with_next_nearest_neighbors
        self.with_z_phase = with_z_phase

        if grid is None:
            self.nearest_neighbor_pairs = [(i, i+1) for i in range(self.n_qubits-1)]
        else:
            self.nearest_neighbor_pairs = grid.get_NN_indices()

        if self.with_evenly_distributed_start_x:
            # Calculate the step size for distributing X gates
            self.step_size = self.n_qubits / (self.cardinality + 1)

        self.J_list, self.h_list = get_ising(Q=QUBO_matrix, offset=QUBO_offset)
        self.simulator = Aer.get_backend('statevector_simulator')

    def set_circuit(self, angles):

        qcircuit = QuantumCircuit(self.n_qubits)

        # Initial state
        if self.with_evenly_distributed_start_x:
            # Distributing x-gates across string evenly
            for i in range(1, self.cardinality + 1):
                qcircuit.x(int(self.step_size * i))
        else:
            # Setting 'k' first with x-gates
            for qubit_index in range(self.cardinality):
                qcircuit.x(qubit_index)

        NN_angles_per_layer = self.n_qubits - 1
        NNN_angles_per_layer = self.n_qubits - 2
        total_NN_angles = NN_angles_per_layer * self.layers
        for layer in range(self.layers):
            # Nearest Neighbor
            counter = 0
            for (qubit_i, qubit_j) in self.nearest_neighbor_pairs:
                theta_ij = angles[(layer * NN_angles_per_layer) + counter]

                # Define the Hamiltonian for XX and YY interactions
                xx_term = theta_ij * (X ^ X)
                yy_term = theta_ij * (Y ^ Y)
                hamiltonian = xx_term + yy_term

                # Create the time-evolved operator
                time_evolved_operator = PauliEvolutionGate(hamiltonian, time=1.0)
                qcircuit.append(time_evolved_operator, [qubit_i, qubit_j])

                # Increment counter for angles
                counter += 1

            if self.with_z_phase:
                angle_start_idx = self.layers * NN_angles_per_layer
                for qubit_i in range(self.n_qubits):
                    qcircuit.rz(phi=2*angles[angle_start_idx + (layer * self.n_qubits) + qubit_i], qubit=qubit_i)

            if self.with_next_nearest_neighbors:
                # Next Nearest Neighbor
                for qubit_i in range(self.n_qubits - 2):
                    theta_ij = angles[total_NN_angles + (layer * NNN_angles_per_layer) + qubit_i]
                    qubit_j = qubit_i + 2

                    # Define the Hamiltonian for XX and YY interactions
                    xx_term = theta_ij * (X ^ X)
                    yy_term = theta_ij * (Y ^ Y)
                    hamiltonian = xx_term + yy_term

                    # Create the time-evolved operator
                    time_evolved_operator = PauliEvolutionGate(hamiltonian, time=1.0)
                    qcircuit.append(time_evolved_operator, [qubit_i, qubit_j])

        return qcircuit

    def get_cost(self, angles) -> float:
        circuit = self.set_circuit(angles=angles)
        counts = execute(circuit, self.simulator).result().get_counts()
        return np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO_matrix) for
                        bitstring, probability in counts.items()])

    def get_state_probabilities(self, angles, flip_states: bool = True) -> dict:
        circuit = self.set_circuit(angles=angles)
        counts = execute(circuit, self.simulator).result().get_counts()
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring: probability for bitstring, probability in counts.items()}

    def get_fidelity(self, angles_1, angles_2) -> float:
        circuit_1 = self.set_circuit(angles=angles_1)
        circuit_2 = self.set_circuit(angles=angles_2)
        state_vector_1 = execute(circuit_1, self.simulator).result().get_statevector(circuit_1)
        state_vector_2 = execute(circuit_2, self.simulator).result().get_statevector(circuit_2)
        return state_fidelity(state_vector_1, state_vector_2)