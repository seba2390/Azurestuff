from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator
from scipy.linalg import expm
import numpy as np

from src.Tools import get_ising, qubo_cost, string_to_array, get_full_hamiltonian



class QAOA:
    def __init__(self,
                 N_qubits,
                 layers,
                 QUBO_matrix,
                 QUBO_offset,
                 constraining_mixer: bool = False,
                 mixer_qubit_indices: List[Tuple[int, int]] = None):
        self.n_qubits = N_qubits
        self.layers = layers
        self.QUBO_matrix = QUBO_matrix
        self.J_list, self.h_list = get_ising(Q=QUBO_matrix, offset=QUBO_offset)
        self.simulator = Aer.get_backend('statevector_simulator')
        self.constraining_mixer = constraining_mixer
        if constraining_mixer:
            if mixer_qubit_indices is None:
                raise ValueError(f'"mixer_qubit_indices" should be provided when "constraining_mixer" is True...')
        self.mixer_qubit_indices = mixer_qubit_indices

        self.counts = None

    def set_circuit(self, angles):

        gamma = angles[self.layers:]
        beta = angles[:self.layers]

        qcircuit = QuantumCircuit(self.n_qubits)

        mixer_angles = None
        if not self.constraining_mixer:
            # Initial state: Hadamard gate on each qubit
            for qubit_index in range(self.n_qubits):
                qcircuit.h(qubit_index)
        else:
            mixer_angles = angles[self.layers:]

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
            if not self.constraining_mixer:
                # Mixer unitary: Weighted X rotation on each qubit
                for qubit_i in range(self.n_qubits):
                    qcircuit.rx(2 * beta[layer], qubit_i)
            else:
                H = get_full_hamiltonian(indices=self.mixer_qubit_indices,
                                         angles=[mixer_angles[layer] for idx in self.mixer_qubit_indices],
                                         N_qubits=self.n_qubits,
                                         with_z_phase=False)
                time = 1.0
                U_H = Operator(expm(-1j * time * H.data))
                qcircuit.append(U_H, list(range(self.n_qubits)))

        return qcircuit

    def get_cost(self, angles) -> float:
        circuit = self.set_circuit(angles=angles)
        self.counts = execute(circuit, self.simulator).result().get_counts()
        return np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO_matrix) for
                        bitstring, probability in self.counts.items()])

    def get_state_probabilities(self, flip_states: bool = True) -> dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
