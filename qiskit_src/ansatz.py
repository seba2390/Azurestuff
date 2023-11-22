from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import X, Y, PauliOp

import numpy as np
from numba import jit


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
                index = int(self.step_size * i)
                print(index)
                qcircuit.x(index)
        else:
            # Setting 'k' first with x-gates
            for qubit_index in range(self.cardinality):
                qcircuit.x(qubit_index)

        NN_angles_per_layer = self.n_qubits - 1
        NNN_angles_per_layer = self.n_qubits - 2
        total_NN_angles = NN_angles_per_layer * self.layers
        for layer in range(self.layers):
            # Nearest Neighbor
            for qubit_i in range(self.n_qubits - 1):
                theta_ij = angles[(layer * NN_angles_per_layer) + qubit_i]
                qubit_j = qubit_i + 1

                # Define the Hamiltonian for XX and YY interactions
                xx_term = theta_ij * (X ^ X)
                yy_term = theta_ij * (Y ^ Y)
                hamiltonian = xx_term + yy_term

                # Create the time-evolved operator
                time_evolved_operator = PauliEvolutionGate(hamiltonian, time=1.0)
                qcircuit.append(time_evolved_operator, [qubit_i, qubit_j])

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
