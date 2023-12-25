from typing import List, Dict

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import X, Y
import numpy as np

from src.Tools import qubo_cost, string_to_array


class CP_QAOA:
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 QUBO_matrix,
                 grid=None,
                 with_evenly_distributed_start_x: bool = False,
                 with_next_nearest_neighbors: bool = False):

        self.n_qubits = N_qubits
        self.cardinality = cardinality
        self.layers = layers
        self.QUBO_matrix = QUBO_matrix

        self.with_evenly_distributed_start_x = with_evenly_distributed_start_x
        self.with_next_nearest_neighbors = with_next_nearest_neighbors

        if grid is None:
            # Normal 1D chain Nearest Neighbors
            self.nearest_neighbor_pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
        else:
            # Grid Nearest Neighbors
            self.nearest_neighbor_pairs = grid.get_NN_indices()

        if self.with_evenly_distributed_start_x:
            # Calculate the step size for distributing X gates
            self.step_size = self.n_qubits / (self.cardinality + 1)

        self.simulator = Aer.get_backend('statevector_simulator')

        # print("Using cardinality: ", self.cardinality)
        # print('Initial excitations at: ', [int(self.step_size * i) for i in range(1, self.cardinality+1)])

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

            # Next Nearest Neighbor
            if self.with_next_nearest_neighbors:
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

    def get_state_probabilities(self, angles, flip_states: bool = True) -> Dict:
        circuit = self.set_circuit(angles=angles)
        counts = execute(circuit, self.simulator).result().get_counts()
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring: probability for bitstring, probability in counts.items()}

    def get_layer_prob_dist(self, N_layers: int, angles) -> List[Dict]:
        original_number_of_layers = self.layers
        result = []
        ## STARTING STATE ##
        qcircuit = QuantumCircuit(self.n_qubits)
        if self.with_evenly_distributed_start_x:
            # Distributing x-gates across string evenly
            for i in range(1, self.cardinality + 1):
                qcircuit.x(int(self.step_size * i))
        else:
            # Setting 'k' first with x-gates
            for qubit_index in range(self.cardinality):
                qcircuit.x(qubit_index)
        counts = execute(qcircuit, self.simulator).result().get_counts()
        result.append({bitstring: probability for bitstring, probability in counts.items()})

        ## REMAINING LAYERS ##
        for layers in range(1, N_layers + 1):
            self.layers = layers
            result.append(self.get_state_probabilities(angles=angles, flip_states=False))
        self.layers = original_number_of_layers
        return result
