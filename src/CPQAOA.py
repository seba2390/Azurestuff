from typing import List, Dict

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import X, Y
import numpy as np

from src.Tools import qubo_cost, string_to_array
from src.Grid import Grid

class CP_QAOA:
    def __init__(self,
                 N_qubits,
                 cardinality,
                 layers,
                 QUBO_matrix,
                 grid: Grid = None,
                 initialization_strategy: np.ndarray = None,
                 with_z_phase: bool = False,
                 with_initialization_strategy: bool = False,
                 with_next_nearest_neighbors: bool = False):

        self.n_qubits = N_qubits
        self.cardinality = cardinality
        self.layers = layers
        self.QUBO_matrix = QUBO_matrix

        self.with_initialization_strategy = with_initialization_strategy
        self.with_next_nearest_neighbors = with_next_nearest_neighbors
        self.with_z_phase = with_z_phase

        self.counts = None

        if grid is None:
            # Normal 1D chain Nearest Neighbors
            self.nearest_neighbor_pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
        else:
            # Grid Nearest Neighbors
            self.nearest_neighbor_pairs = grid.get_NN_indices()

        if self.with_initialization_strategy:
            if grid is None and initialization_strategy is None:
                raise ValueError(f'if "with_initialization_strategy" is set true, either a grid w. an initialization '
                                 f'strategy or a initialization strategy should be provided')
            elif grid is not None and initialization_strategy is not None:
                raise ValueError(
                    f'Either "grid" (which has init. strat.) or "initialization_strategy" should be provided - not both.')
            elif grid is None and initialization_strategy is not None:
                if np.any(initialization_strategy < 0) or np.any(initialization_strategy > self.n_qubits - 1):
                    raise ValueError(f'"initialization_strategy" should only contain integers in range [0;N qubits[')
                self.initialization_strategy = initialization_strategy
            else:
                self.initialization_strategy = grid.get_initialization_indices()

            if len(self.initialization_strategy) != self.cardinality:
                raise ValueError(f'Provided initialization strategy that does not match provided cardinality')

            if len(list(set(self.initialization_strategy.tolist()))) != self.cardinality:
                raise ValueError(f'Provided initialization strategy that holds multiple of same qubit idx.')

        elif not self.with_initialization_strategy and grid is None:
            print(" # === N.B. - no initialization strategy was provided === #")

        self.simulator = Aer.get_backend('statevector_simulator')

    def set_circuit(self, angles):

        qcircuit = QuantumCircuit(self.n_qubits)

        # Initial state
        if self.with_initialization_strategy:
            # Distributing x-gates across string evenly
            for qubit_index in self.initialization_strategy:
                qcircuit.x(qubit_index)
        else:
            # Setting 'k' first with x-gates
            for qubit_index in range(self.cardinality):
                qcircuit.x(qubit_index)

        # Setting aside first (N-1)*L angles for NN-interactions
        NN_angles_per_layer = len(self.nearest_neighbor_pairs)
        NN_angles = angles[:NN_angles_per_layer * self.layers]
        NN_counter = 0

        if self.with_next_nearest_neighbors:
            # Setting aside next (N-2)*L angles for NNN-interactions
            NNN_angles_per_layer = self.n_qubits - 2
            NNN_angles = angles[NN_angles_per_layer * self.layers: NNN_angles_per_layer * self.layers]
            NNN_counter = 0

        if self.with_z_phase:
            # Setting aside last N*L angles for z-phase
            Z_Phase_angles_per_layer = self.n_qubits - 1
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

    def get_state_probabilities(self, angles, flip_states: bool = True) -> Dict:
        #circuit = self.set_circuit(angles=angles)
        #counts = execute(circuit, self.simulator).result().get_counts()
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
