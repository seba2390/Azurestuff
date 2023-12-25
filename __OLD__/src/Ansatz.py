import sys
import os
import warnings

import numpy as np

from src.SYMQCircuit import *
from src.cp_QAOATools import *
from src.Hamiltonian import (MixerHamiltonian,
                             CostHamiltonian)


class CPQAOAansatz:
    def __init__(self, n_qubits: int,
                 with_cost: bool,
                 with_z_phase: bool,
                 classic_QAOA: bool,
                 n_layers: int,
                 cardinality: int,
                 w_edges: Union[List[Tuple[int, int, float]], None],
                 precision: int = 64):

        if not isinstance(n_qubits, int):
            raise ValueError(f"n_qubits should be integer.")
        if n_qubits <= 0:
            raise ValueError(f'n_qubits should be > 0, but is: {n_qubits}.')
        if not isinstance(n_layers, int):
            raise ValueError(f"n_layers should be integer.")
        if n_qubits < 1:
            raise ValueError(f'n_layers should be >= 1, but is: {n_layers}.')
        if not 0 <= cardinality < n_qubits:
            raise ValueError(f'cardinality should be in range: 0 <= cardinality < n_qubits.')

        self.n_qubits = n_qubits
        self.cardinality = cardinality
        self.n_layers = n_layers
        self.w_edges = w_edges
        self.precision = precision

        self.with_cost = with_cost
        self.classic_QAOA = classic_QAOA
        self.with_z_phase = with_z_phase

        # --------- QUBO --------- #
        QUBO_dict = {}
        self.QUBO_mat = None
        if self.w_edges is not None:
            self.QUBO_mat = get_qubo(size=self.n_qubits, edges=self.w_edges)
            for row in range(self.QUBO_mat.shape[0]):
                for col in range(self.QUBO_mat.shape[1]):
                    if self.QUBO_mat[row, col] != 0.0:
                        QUBO_dict[(row, col)] = self.QUBO_mat[row, col]

        # --------- Ising --------- #
        self.h_vec = np.zeros(shape=(self.n_qubits, 1), dtype=float)
        self.J_mat = np.zeros(shape=(self.n_qubits, self.n_qubits), dtype=float)
        self.ising_offset = 0
        self.J_list, self.h_list = None, None
        self.h_dict, self.J_dict = None, None
        if self.w_edges is not None:
            h_dict, J_dict, offset = qubo_to_ising(Q=QUBO_dict)
            self.h_list = [(key, h_dict[key]) for key in h_dict.keys()]
            self.J_list = [(key[0], key[1], J_dict[key]) for key in J_dict.keys()]
            for i, val in self.h_list:
                self.h_vec[i] = val
            for i, j, val in self.J_list:
                self.J_mat[i, j] = val

        if self.w_edges is not None:
            self.nr_cost_terms = self.get_nr_cost_terms()
        self.nr_mixer_terms = self.n_qubits  # For standard QAOA

    def set_ising_model(self, J: np.ndarray[float], h: np.ndarray[float], offset: float) -> None:
        if not isinstance(J, np.ndarray):
            raise ValueError(f' "J" should be a numpy array.')
        if not isinstance(h, np.ndarray):
            raise ValueError(f' "h" should be a numpy array.')
        if J.shape[0] != J.shape[1]:
            raise ValueError(f'J should be a square matrix.')
        if len(h) != self.n_qubits:
            raise ValueError(f'Dimensionality of provided "h" dosent match n_qubits of instance.')
        if J.shape[0] != self.n_qubits:
            raise ValueError(f'Dimensionality of provided "J" dosent match n_qubits of instance.')

        self.h_vec = h
        self.J_mat = J
        self.ising_offset = offset

        J_dict = {}
        J_list = []
        for q1 in range(J.shape[0]):
            for q2 in range(q1 + 1, J.shape[1]):
                J_list.append((q1, q2, J[q1, q1]))
                J_dict[(q1, q2)] = J[q1, q2]

        h_dict = {}
        h_list = []
        for q in range(len(h)):
            h_list.append((q, h[q]))
            h_dict[q] = h[q]

        self.J_list, self.J_dict = J_list, J_dict
        self.h_list, self.h_dict = h_list, h_dict

        self.nr_cost_terms = self.get_nr_cost_terms()

        Q_dict, offset = ising_to_qubo(J=J_dict, h=h_dict)
        for key, val in Q_dict.items():
            Q_dict[key] *= 0.5
        self.QUBO_mat = get_qubo(size=self.n_qubits, edges=Q_dict)

    def get_nr_cost_terms(self):
        nr_terms = 0
        # ------ Cost unitary: ------ #
        # Weighted RZZ gate for each edge
        for qubit_i, qubit_j, weight in self.J_list:
            nr_terms += 1
        # Weighted RZ gate for each qubit
        for qubit_i, weight in self.h_list:
            nr_terms += 1
        return nr_terms

    def set_circuit(self, theta: List[float]):
        """
        Construct a QAOA circuit with weighted terms.

        Parameters:
        -----------
        n_qubits : int
            Number of qubits in the circuit.
        edges : Tuple[List[Tuple[int, int, float]], List[Tuple[int, float]]]
            Tuple containing lists of edge information: (J, h).
            J: List of tuples (qubit_i, qubit_j, weight) representing weighted edges.
            h: List of tuples (qubit_i, weight) representing weighted terms.
        theta : List[float]
            List of optimization parameters.

        Returns:
        --------
        SYMQCircuit
            Constructed QAOA circuit with weighted terms.

        Notes:
        ------
        This function constructs a QAOA circuit with weighted terms for the optimization problem.
        The QAOA circuit includes alternating cost and mixer unitaries based on the given parameters.
        """
        # Initializing Q circuit
        qcircuit = SYMQCircuit(nr_qubits=self.n_qubits, precision=self.precision)

        if not self.classic_QAOA:
            # Only Mixer
            if not self.with_z_phase and not self.with_cost:
                _theta_ = theta
            # Mixer + z-phase terms
            elif self.with_z_phase and not self.with_cost:
                _theta_ = theta[:(self.n_qubits - 1) * self.n_layers]
                _phi_ = theta[(self.n_qubits - 1) * self.n_layers:]
            # Mixer + z-phase term + cost
            elif self.with_z_phase and self.with_cost:
                _theta_ = theta[:(self.n_qubits - 1) * self.n_layers]
                _phi_ = theta[(self.n_qubits - 1) * self.n_layers : 2 * ((self.n_qubits - 1) * self.n_layers)]
                _lmbda_ = theta[2 * ((self.n_qubits - 1) * self.n_layers):]

            # Initial_state: k ones set
            for qubit_index in range(self.cardinality):
                qcircuit.add_x(target_qubit=qubit_index)

            # For each Cost,Mixer repetition
            for irep in range(0, self.n_layers):

                if self.with_cost:
                    # ------ Cost unitary: ------ #
                    H_C = CostHamiltonian(hilbert_space_size=self.n_qubits,
                                          backend='SYMQCircuit',
                                          precision=self.precision)
                    J_N, h_N = len(self.J_list), len(self.h_list)
                    theta_values = _lmbda_[(J_N+h_N) * irep: (J_N+h_N) * (irep+1) - h_N]
                    phi_values   = _lmbda_[(J_N+h_N) * irep + J_N: (J_N+h_N) * (irep+1)]
                    cost_circ_mat_rep = H_C.get_unitary_matrix_representation(theta_values=theta_values,
                                                                              phi_values=phi_values,
                                                                              ising_J=self.J_list,
                                                                              ising_h=self.h_list)
                    qcircuit.add_gate(matrix_representation=cost_circ_mat_rep)

                # ------ Mixer unitary: ------ #
                H_M = MixerHamiltonian(hilbert_space_size=self.n_qubits,
                                       connectivity='NearestNeighbor',
                                       backend='SYMQCircuit',
                                       precision=self.precision,
                                       with_z_phase=self.with_z_phase)
                theta_values = _theta_[(self.n_qubits - 1) * irep: (self.n_qubits - 1) * (irep + 1)]
                if self.with_z_phase:
                    phi_values = _phi_[(self.n_qubits - 1) * irep: (self.n_qubits - 1) * (irep + 1)]
                else:
                    phi_values = np.zeros_like(theta_values)
                mixer_circ_mat_rep = H_M.get_unitary_matrix_representation(theta_values=theta_values,
                                                                           phi_values=phi_values)
                qcircuit.add_gate(matrix_representation=mixer_circ_mat_rep)
        else:

            # Number of alternating (Cost,Mixer) unitaries
            p = len(theta) // 2

            # Gamma opt param for cost unitaries as last p vals.
            gamma = theta[p:]

            # Beta opt param for mixing unitaries as first p vals.
            beta = theta[:p]

            # Initial_state: Hadamard gate on each qbit
            for qubit_index in range(self.n_qubits):
                qcircuit.add_h(target_qubit=qubit_index)

            # For each Cost,Mixer repetition
            for irep in range(0, p):

                # ------ Cost unitary: ------ #
                # Weighted RZZ gate for each edge
                for qubit_i, qubit_j, weight in self.J_list:
                    angle = 2 * gamma[irep] * weight
                    # print(f"Layer: {irep}, RZZ angle: {angle}, gamma: {gamma[irep]}, weight: {weight}")
                    qcircuit.add_rzz(qubit_1=qubit_i, qubit_2=qubit_j, angle=angle)
                # Weighted RZ gate for each qubit
                for qubit_i, weight in self.h_list:
                    angle = 2 * gamma[irep] * weight
                    # print(f"Layer: {irep}, RZ angle: {angle}, gamma: {gamma[irep]}, weight: {weight}")
                    qcircuit.add_rz(target_qubit=qubit_i, angle=angle)

                # ------ Mixer unitary: ------ #
                # Mixer unitary: Weighted X rotation on each qubit
                for qubit_i in range(self.n_qubits):
                    angle = 2 * beta[irep]
                    # print(f"Layer: {irep}, RX angle: {angle}, beta: {beta[irep]}, weight: {weight}")
                    qcircuit.add_rx(target_qubit=qubit_i, angle=angle)

        return qcircuit

    def Ising_cost(self, state: np.ndarray) -> float:
        """
        Calculate the Ising cost for a given spin configuration.

        Parameters:
        state (np.ndarray): Array representing the spin configuration.

        Returns:
        float: The computed Ising cost.
        """
        return float(state.T @ (self.J_mat @ state) + state.T @ self.h_vec) + self.ising_offset

    def compute_expectation(self, counts: dict) -> float:
        """
        Compute the expectation value based on measurement results.

        Parameters:
        counts (dict): A dictionary containing bitstring measurement outcomes as keys and their corresponding
        probabilities as values.

        Returns:
        float: The computed expectation value.
        """

        def to_ising_state(qubo_state: np.ndarray) -> np.ndarray:
            return 2 * qubo_state - 1

        _state_ = np.zeros(shape=self.h_vec.shape, dtype=float)
        _result_ = 0.0
        for bitstring, probability in counts.items():
            _state_ = np.array(list(bitstring)).reshape((self.n_qubits, 1)).astype(int)
            # N.B. Q has been multiplied with -1, therefore we are doing += instead of -= here.
            _state_ = to_ising_state(_state_)
            _result_ += self.Ising_cost(state=_state_) * probability
        return _result_

    def evaluate_circuit(self, theta: List[float]):
        """
        Execute a quantum circuit with the given parameters and compute the expectation value.

        Parameters:
        theta (List[float]): List of parameters for the quantum circuit.

        Returns:
        float: The computed expectation value.
        """

        prob_distribution = self.set_circuit(theta=theta).get_state_probabilities(eps=0)
        return self.compute_expectation(counts=prob_distribution)
