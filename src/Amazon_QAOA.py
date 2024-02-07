from time import time

import numpy as np
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator

from src.Tools import get_ising, qubo_cost, string_to_array


class Amazon_QAOA:
    def __init__(self,
                 N_qubits,
                 layers,
                 QUBO_matrix,
                 QUBO_offset,
                 normalize_cost: bool = False):
        self.n_qubits = N_qubits
        self.layers = layers
        self.QUBO_matrix = QUBO_matrix
        self.J_list, self.h_list = get_ising(Q=QUBO_matrix, offset=QUBO_offset)

        self.normalize_cost = normalize_cost
        self.device = LocalSimulator('braket_sv')

        self.circuit = self.set_circuit()
        self.counts = None
        self.cost_time, self.circuit_time = 0.0, 0.0

    @staticmethod
    def _int_to_fixed_length_binary_array_(number: int, num_bits: int) -> str:
        # Convert the number to binary and remove the '0b' prefix
        binary_str = bin(number)[2:]
        # Pad the binary string with zeros if necessary
        return binary_str.zfill(num_bits)

    def get_counts(self, state_vector: np.ndarray) -> dict[str, float]:
        n_qubits = int(np.log2(len(state_vector)))
        return {self._int_to_fixed_length_binary_array_(number=idx, num_bits=n_qubits): np.abs(state_vector[idx]) ** 2
                for idx in range(len(state_vector))}

    def set_circuit(self):

        circ = Circuit()
        H_on_all = Circuit().h(range(0, self.n_qubits))
        circ.add(H_on_all)

        # setup two parameter families
        gammas = [FreeParameter(f'gamma_{layer}') for layer in range(self.layers)]
        betas = [FreeParameter(f'beta_{layer}') for layer in range(self.layers)]

        # add QAOA circuit layer blocks
        for layer in range(self.layers):
            # Cost part
            for q_i, h_i in self.h_list:
                R_z = Circuit().rz(q_i, 2 * gammas[layer] * h_i)
                circ.add(R_z)
            for q_i, q_j, j_ij in self.J_list:
                R_zz = Circuit().zz(q_i, q_j, 2 * gammas[layer])
                circ.add(R_zz)
            # Mixer part
            RX_on_all = Circuit().rx(range(0, self.n_qubits), 2 * betas[layer])
            circ.add(RX_on_all)
        circ.state_vector()
        return circ

    def get_cost(self, angles):
        __start__ = time()
        gamma_vals, beta_vals = angles[self.layers:], angles[:self.layers]
        param_val_dict = {**{f'gamma_{i}': gamma_vals[i] for i in range(len(gamma_vals))},
                          **{f'beta_{i}': beta_vals[i] for i in range(len(beta_vals))}}
        state_vector = self.device.run(self.circuit, shots=0, inputs=param_val_dict).result().values[0]
        __end__ = time()
        self.circuit_time += __end__ - __start__

        __start__ = time()
        self.counts = self.get_counts(state_vector=np.array(state_vector))
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO_matrix) for
                        bitstring, probability in self.counts.items()])
        __end__ = time()
        self.cost_time += __end__ - __start__

        return cost

    def get_state_probabilities(self, flip_states: bool = True) -> dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
