from typing import List, Dict, Union

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import X, Y
import numpy as np

from src.Tools import qubo_cost, string_to_array
from src.Grid import Grid
from src.Chain import Chain


class CP_QAOA:
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 QUBO_matrix,
                 topology: Union[Grid, Chain],
                 with_z_phase: bool = False,
                 with_next_nearest_neighbors: bool = False,
                 backend: str = 'state_vector'):

        self.n_qubits = N_qubits
        self.cardinality = cardinality
        self.layers = layers
        self.QUBO_matrix = QUBO_matrix
        self.with_next_nearest_neighbors = with_next_nearest_neighbors
        self.with_z_phase = with_z_phase

        if topology.N_qubits != self.n_qubits:
            raise ValueError(f'provided topology consists of different number of qubits that provided for this ansatz.')

        # Nearest Neighbors
        self.nearest_neighbor_pairs = topology.get_NN_indices()
        # Nearest + Next Nearest Neighbors
        self.next_nearest_neighbor_pairs = topology.get_NNN_indices()
        # Strategy for which qubits to set:
        self.initialization_strategy = topology.get_initialization_indices()

        # For storing probability <-> state dict during opt. to avoid extra call for callback function
        self.counts = None

        if backend == 'state_vector':
            # Using state-vector sim. for theoretical accuracy
            self.simulator = Aer.get_backend('statevector_simulator')

    def set_circuit(self, angles):

        qcircuit = QuantumCircuit(self.n_qubits)

        # Setting 'k' qubits to |1>
        for qubit_index in self.initialization_strategy:
            qcircuit.x(qubit_index)

        # Setting aside first (N-1)*L angles for NN-interactions
        NN_angles_per_layer = len(self.nearest_neighbor_pairs)
        NN_angles = angles[:NN_angles_per_layer * self.layers]
        NN_counter = 0

        if self.with_next_nearest_neighbors:
            # Setting aside next (N-2)*L angles for NNN-interactions
            NNN_angles_per_layer = len(self.next_nearest_neighbor_pairs)
            NNN_angles = angles[NN_angles_per_layer * self.layers:][:NNN_angles_per_layer * self.layers]
            NNN_counter = 0

        if self.with_z_phase:
            # Setting aside last N*L angles for z-phase
            Z_Phase_angles_per_layer = self.n_qubits
            Z_Phase_angles = angles[-Z_Phase_angles_per_layer * self.layers:]
            Z_Phase_counter = 0

        for layer in range(self.layers):
            # Nearest Neighbor
            for (qubit_i, qubit_j) in self.nearest_neighbor_pairs:
                theta_ij = NN_angles[NN_counter]

                # Define the Hamiltonian for XX and YY interactions
                xx_term = theta_ij * (X ^ X)
                yy_term = theta_ij * (Y ^ Y)
                hamiltonian = xx_term + yy_term

                # Create the time-evolved operator
                time_evolved_operator = PauliEvolutionGate(hamiltonian, time=1.0)
                qcircuit.append(time_evolved_operator, [qubit_i, qubit_j])

                # Increment counter for angles
                NN_counter += 1

            # Next Nearest Neighbor
            if self.with_next_nearest_neighbors:
                for qubit_i in range(self.n_qubits - 2):
                    theta_ij = NNN_angles[NNN_counter]
                    qubit_j = qubit_i + 2

                    # Define the Hamiltonian for XX and YY interactions
                    xx_term = theta_ij * (X ^ X)
                    yy_term = theta_ij * (Y ^ Y)
                    hamiltonian = xx_term + yy_term

                    # Create the time-evolved operator
                    time_evolved_operator = PauliEvolutionGate(hamiltonian, time=1.0)
                    qcircuit.append(time_evolved_operator, [qubit_i, qubit_j])

                    # Increment counter for angles
                    NNN_counter += 1

            if self.with_z_phase:
                for qubit_i in range(self.n_qubits):
                    theta_i = Z_Phase_angles[Z_Phase_counter]
                    qcircuit.rz(phi=2 * theta_i, qubit=qubit_i)

                    # Increment counter for angles
                    Z_Phase_counter += 1

        return qcircuit

    def get_cost(self, angles) -> float:
        circuit = self.set_circuit(angles=angles)
        self.counts = execute(circuit, self.simulator).result().get_counts()
        return np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO_matrix) for
                        bitstring, probability in self.counts.items()])

    def get_state_probabilities(self, flip_states: bool = True) -> Dict:
        counts = self.counts
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
