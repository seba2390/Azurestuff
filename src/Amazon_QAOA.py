from typing import Sequence, Tuple, List
from time import time

import sympy
import numpy as np
from braket.aws import AwsDevice
from braket.circuits import Circuit
from braket.circuits.circuit import subroutine
from braket.devices import LocalSimulator
from braket.parametric import FreeParameter

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
        # Read bottom of page: https://quantumai.google/cirq/simulate/simulation
        # and consider: https://github.com/quantumlib/qsim/blob/master/docs/tutorials/qsimcirq.ipynb
        self.device = LocalSimulator('braket_sv')

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

    def set_circuit(self, angles):

        circ = Circuit()
        H_on_all = Circuit().h(range(0, self.n_qubits))
        circ.add(H_on_all)

        # setup two parameter families
        gammas = angles[self.layers:]
        betas = angles[:self.layers]

        # add QAOA circuit layer blocks
        for layer in range(self.layers):
            # Cost part
            for q_i, h_i in self.h_list:
                R_z = Circuit().rz(q_i, 2 * gammas[layer] * h_i)
                circ.add(R_z)
            for q_i, q_j, j_ij in self.J_list:
                R_zz = Circuit().cnot(q_i, q_j).rz(q_j, gammas[layer]).cnot(q_i, q_j)
                circ.add(R_zz)

            # Mixer part
            for q_i in range(self.n_qubits):
                R_x = Circuit().rx(q_i, 2 * betas[layer])
                circ.add(R_x)
        circ.state_vector()
        return circ

    def get_cost(self, angles):
        __start__ = time()
        circuit = self.set_circuit(angles)
        state_vector = self.device.run(circuit,shots=100).result()
        """state_vector = self.simulator.simulate(circuit, param_resolver=params).final_state_vector
        self.counts = self.get_counts(state_vector=np.array(state_vector))
        __end__ = time()
        self.circuit_time += __end__ - __start__
        __start__ = time()
        cost = np.mean([probability * qubo_cost(state=string_to_array(bitstring), QUBO_matrix=self.QUBO_matrix) for
                        bitstring, probability in self.counts.items()])
        __end__ = time()
        self.cost_time += __end__ - __start__
        return cost"""
        return state_vector

    def get_state_probabilities(self, flip_states: bool = True) -> dict:
        counts = self.counts
        if flip_states:
            return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
        return {bitstring[::-1]: probability for bitstring, probability in counts.items()}
