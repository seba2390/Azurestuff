from typing import *

import scipy as sc
import numpy as np

from src.SYMQCircuit import SYMQCircuit


class MixerHamiltonian:

    def __init__(self,
                 hilbert_space_size: int,
                 with_z_phase: bool,
                 backend: str = 'SYMQCircuit',
                 connectivity: str = 'NearestNeighbor',
                 precision: int = 64) -> None:
        self.__supported_backends__ = ['SYMQCircuit']
        self.__supported_connectivity__ = ['NearestNeighbor', 'NextNearestNeighbor', 'AllToAll']
        self.__dtypes__ = {64: np.complex64, 128: np.complex128}
        if not isinstance(hilbert_space_size, int):
            raise TypeError(f'"hilbert_space_size" should be type "int", but is: {type(hilbert_space_size)}.')
        if hilbert_space_size <= 0:
            raise ValueError(f'"hilbert_space_size" should be positive > 0, but is: {hilbert_space_size}.')

        if precision not in list(self.__dtypes__.keys()):
            raise ValueError(
                f'Unrecognized nr. of bits for precision, should be either of: {list(self.__dtypes__.keys())}.')
        if backend not in self.__supported_backends__:
            raise ValueError(f'Unrecognized backend, should be either of: {self.__supported_backends__}.')
        if connectivity not in self.__supported_connectivity__:
            raise ValueError(f'Unrecognized connectivity, should be either of: {self.__supported_connectivity__}.')

        self.connectivity = connectivity
        self.backend = backend
        self.precision = precision
        self.hilbert_space_size = hilbert_space_size
        self.with_z_phase = with_z_phase

    def _get_xx_matrix_rep_(self, q_i: int, q_j: int) -> np.ndarray:
        if self.backend == 'SYMQCircuit':
            xx_circ = SYMQCircuit(nr_qubits=self.hilbert_space_size, precision=self.precision)
            xx_circ.add_x(target_qubit=q_i)
            xx_circ.add_x(target_qubit=q_j)
            return xx_circ.get_circuit_unitary(as_sparse=False)

    def _get_yy_matrix_rep_(self, q_i: int, q_j: int) -> np.ndarray:
        if self.backend == 'SYMQCircuit':
            yy_circ = SYMQCircuit(nr_qubits=self.hilbert_space_size, precision=self.precision)
            yy_circ.add_y(target_qubit=q_i)
            yy_circ.add_y(target_qubit=q_j)
            return yy_circ.get_circuit_unitary(as_sparse=False)

    def _get_z_matrix_rep_(self, q_i: int) -> np.ndarray:
        if self.backend == 'SYMQCircuit':
            z_circ = SYMQCircuit(nr_qubits=self.hilbert_space_size, precision=self.precision)
            z_circ.add_z(target_qubit=q_i)
            return z_circ.get_circuit_unitary(as_sparse=False)

    def _get_term_(self, q_i: int, q_j: int, theta_ij: float, phi_i: float = 0.0) -> np.ndarray:
        mat_repr_1 = theta_ij * self._get_xx_matrix_rep_(q_i=q_i, q_j=q_j)
        mat_repr_2 = theta_ij * self._get_yy_matrix_rep_(q_i=q_i, q_j=q_j)
        if self.with_z_phase:
            mat_repr_3 = phi_i * self._get_z_matrix_rep_(q_i=q_i)
            return sc.linalg.expm(-1j * (mat_repr_1 + mat_repr_2 + mat_repr_3))
        else:
            return sc.linalg.expm(-1j * (mat_repr_1 + mat_repr_2))

    def get_unitary_matrix_representation(self, theta_values: Union[list, np.ndarray],
                                          phi_values: Union[list, np.ndarray]):

        final_representation = np.eye(N=2 ** self.hilbert_space_size, dtype=self.__dtypes__[self.precision])
        if self.connectivity == 'NearestNeighbor':
            if not (len(theta_values) == len(phi_values) == self.hilbert_space_size - 1):
                raise ValueError(
                    f'Expected "N-1" theta values and phi values, for NearestNeighbor connectivity, but got {len(theta_values)} and {len(phi_values)}.')
            for q_1 in range(self.hilbert_space_size - 1):
                q_2 = q_1 + 1
                final_representation = self._get_term_(q_i=q_1,
                                                       q_j=q_2,
                                                       theta_ij=theta_values[q_1],
                                                       phi_i=phi_values[q_1]) @ final_representation

        elif self.connectivity == 'NextNearestNeighbor':
            if self.hilbert_space_size <= 2:
                raise ValueError(f' NextNearestNeighbor connectivity requires at least 3 qubits.')
            if not (len(theta_values) == len(phi_values) == 2 * self.hilbert_space_size - 3):
                raise ValueError(
                    f'Expected "2N-3" theta values and phi values, for NextNearestNeighbor connectivity, but got {len(theta_values)} and {len(phi_values)}.')
            angle_counter = 0
            for q_1 in range(self.hilbert_space_size - 2):
                for neighbour in range(1, 3):
                    q_2 = q_1 + neighbour
                    final_representation = self._get_term_(q_i=q_1,
                                                           q_j=q_2,
                                                           theta_ij=theta_values[angle_counter],
                                                           phi_i=phi_values[angle_counter]) @ final_representation
                    angle_counter += 1
        else:
            if not (len(theta_values) == len(phi_values) == int(
                    self.hilbert_space_size * (self.hilbert_space_size - 1) / 2)):
                raise ValueError(
                    f'Expected "N(N-1)/2" theta values and phi values, for AllToAll connectivity, but got {len(theta_values)} and {len(phi_values)}.')
            angle_counter = 0
            for q_1 in range(self.hilbert_space_size):
                for q_2 in range(q_1 + 1, self.hilbert_space_size):
                    final_representation = self._get_term_(q_i=q_1,
                                                           q_j=q_2,
                                                           theta_ij=theta_values[angle_counter],
                                                           phi_i=phi_values[angle_counter]) @ final_representation
                    angle_counter += 1
        return final_representation


class CostHamiltonian:

    def __init__(self,
                 hilbert_space_size: int,
                 backend: str = 'SYMQCircuit',
                 precision: int = 64) -> None:
        __supported_backends__ = ['SYMQCircuit']
        if not isinstance(hilbert_space_size, int):
            raise TypeError(f'"hilbert_space_size" should be type "int", but is: {type(hilbert_space_size)}.')
        if hilbert_space_size <= 0:
            raise ValueError(f'"hilbert_space_size" should be positive > 0, but is: {hilbert_space_size}.')
        _dtypes_ = {64: np.complex64, 128: np.complex128}
        if precision not in list(_dtypes_.keys()):
            raise ValueError(f'Unrecognized nr. of bits for precision, should be either of: {list(_dtypes_.keys())}.')
        if backend not in __supported_backends__:
            raise ValueError(f'Unrecognized backend, should be either of: {__supported_backends__}.')

        self.backend = backend
        self.precision = precision
        self.hilbert_space_size = hilbert_space_size

    def get_unitary_matrix_representation(self,
                                          theta_values: Union[List[float], np.ndarray],
                                          phi_values: Union[List[float], np.ndarray],
                                          ising_J: List[Tuple],
                                          ising_h: List[Tuple]) -> np.ndarray:
        if self.backend == 'SYMQCircuit':

            circ = SYMQCircuit(nr_qubits=self.hilbert_space_size, precision=self.precision)

            angle_counter = 0
            # Weighted RZZ gate for each relevant qubit pair
            for qubit_i, qubit_j, weight in ising_J:
                rzz_angle = 2 * theta_values[angle_counter] * weight
                circ.add_rzz(qubit_1=qubit_i, qubit_2=qubit_j, angle=rzz_angle)
                angle_counter += 1

            angle_counter = 0
            # Weighted RZ gate for each relevant qubit
            for qubit_i, weight in ising_h:
                angle = 2 * phi_values[angle_counter] * weight
                circ.add_rz(target_qubit=qubit_i, angle=angle)
                angle_counter += 1

            return circ.get_circuit_unitary(as_sparse=False)
