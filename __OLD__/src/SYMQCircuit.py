from typing import *

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, kron, identity

from src.SYMQState import *
from src.Tools import represent_integer_with_bits, generate_bit_string_permutations

class SYMQCircuit:
    def __init__(self, nr_qubits: int, precision: int = 64):

        # N.B. 64 bit = 32 bit real and 32 bit imag, whereas 128 = 64 bit real and 64 bit imag.
        _dtypes_ = {64: np.complex64, 128: np.complex128}
        if precision not in list(_dtypes_.keys()):
            raise ValueError('Unrecognized nr. of bits for precision, should be either of: {16,32,64,128}.')
        self.precision = precision
        self.dtype = _dtypes_[self.precision]

        if nr_qubits <= 0:
            raise ValueError(f"Circuit size: '{nr_qubits}' should be positive int.")

        self.circuit_size = nr_qubits
        self.__circuit_unitary__ = identity(2 ** nr_qubits, format='csr', dtype=self.dtype)
        self.__state_vector__ = csr_matrix(np.array([1.0] + [0.0 for _ in range(2 ** nr_qubits - 1)], dtype=self.dtype))

        self._identity_ = identity(2, format='csr', dtype=self.dtype)

        self._x_gate_ = csr_matrix(np.array([[0, 1], [1, 0]], dtype=self.dtype))
        self._y_gate_ = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=self.dtype))
        self._z_gate_ = csr_matrix(np.array([[1, 0], [0, -1]], dtype=self.dtype))

        self._h_gate_ = csr_matrix((1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=self.dtype))

    ###############################################################
    ####################### GENERIC METHODS #######################
    ###############################################################

    def __add__(self, other):
        """
        Overloads the '+' operator to combine two SYMQCircuit objects.

        This method allows adding two SYMQCircuit objects by composing their unitary representations.
        The two circuits must have the same circuit size for addition to be valid. If the dtype of the two circuits
        differs, the resulting circuit will inherit the dtype of the first circuit.

        Args:
            other (SYMQCircuit): The SYMQCircuit object to be added to the current circuit.

        Returns:
            SYMQCircuit: A new SYMQCircuit object representing the composition of the two input circuits.

        Raises:
            ValueError: If the circuit sizes of the two input circuits are not equal.
            TypeError: If the 'other' operand is not of type SYMQCircuit.

        Example:
            circuit1 = SYMQCircuit(nr_qubits=3)
            circuit2 = SYMQCircuit(nr_qubits=3)
            result = circuit1 + circuit2  # Combines the two circuits into a new circuit.
        """
        if isinstance(other, SYMQCircuit):
            if self.circuit_size != other.circuit_size:
                raise ValueError("Circuits are not of equal size.")
            if self.dtype != other.dtype:
                print("Circuits not of same dtype - defaulting to dtype of circuit1 in circuit1 + circuit2.")
            resulting_circuit = SYMQCircuit(nr_qubits=self.circuit_size)
            resulting_circuit.__circuit_unitary__ = other.__circuit_unitary__ @ self.__circuit_unitary__
            resulting_circuit.dtype = self.dtype
            return resulting_circuit
        else:
            raise TypeError("Unsupported operand type for +")

    def __iadd__(self, other):
        """
        Overloads the '+=' operator to combine the current SYMQCircuit object with another.

        This method combines the current SYMQCircuit object with another SYMQCircuit object by composing their unitary
        representations. The two circuits must have the same circuit size for addition to be valid. If the dtype of the two
        circuits differs, the resulting circuit will inherit the dtype of the current circuit.

        Args:
            other (SYMQCircuit): The SYMQCircuit object to be added to the current circuit.

        Returns:
            None

        Raises:
            ValueError: If the circuit sizes of the two input circuits are not equal.
            TypeError: If the 'other' operand is not of type SYMQCircuit.

        Example:
            circuit1 = SYMQCircuit(nr_qubits=3)
            circuit2 = SYMQCircuit(nr_qubits=3)
            circuit1 += circuit2  # Combines the two circuits into the current circuit.
        """
        return self.__add__(other)

    def __eq__(self, other):
        """
        Compare two SYMQCircuit objects for equality.

        This method checks whether two SYMQCircuit objects are equal by comparing their circuit sizes and data types.
        If the two circuits have the same circuit size and data type, their unitary representations are compared element-wise
        for numerical closeness using NumPy's `np.allclose()` function.

        Args:
            other (SYMQCircuit): The SYMQCircuit object to compare with.

        Returns:
            bool: True if the two SYMQCircuit objects are equal, False otherwise.

        Example:
            circuit1 = SYMQCircuit(nr_qubits=3)
            circuit2 = SYMQCircuit(nr_qubits=3)
            if circuit1 == circuit2:
                print("The two circuits are equal.")
        """
        if isinstance(other, SYMQCircuit):
            if self.circuit_size == other.circuit_size and self.dtype == other.dtype:
                return np.allclose(self.get_circuit_unitary(), other.get_circuit_unitary())
        return False

    def __str__(self):
        """
        Generate a human-readable string representation of the SYMQCircuit object.

        Returns:
            str: A string describing the SYMQCircuit object, including the number of qubits.

        Example:
            circuit = SYMQCircuit(nr_qubits=3)
            print(str(circuit))  # Output: "SYMQCircuit with 3 qubits."
        """
        return f"SYMQCircuit with {self.circuit_size} qubits."

    def __repr__(self):
        """
        Generate an unambiguous string representation of the SYMQCircuit object.

        Returns:
            str: A string that can be used to recreate the SYMQCircuit object.

        Example:
            circuit = SYMQCircuit(circuit_size=4, dtype=np.complex64)
            print(repr(circuit))  # Output: "SYMQCircuit(circuit_size=4, dtype=<class 'numpy.complex64'>)"
        """
        return f"SYMQCircuit(circuit_size={self.circuit_size}, dtype={self.dtype})"

    ###############################################################
    ####################### UTILITY METHODS #######################
    ###############################################################

    def _update_circuit_unitary_(self, gate: csr_matrix):
        """
        Update the circuit's unitary representation by performing matrix multiplication with a gate.

        Args:
            gate (np.ndarray): The gate's unitary matrix representation.

        Returns:
            None: This function updates the internal circuit's unitary representation in place.
        """
        self.__circuit_unitary__ = gate @ self.__circuit_unitary__

    def _single_qubit_tensor_prod_matrix_rep_(self, target_qubit: int, gate_mat_rep: csr_matrix) -> csr_matrix:
        """
        Calculate the tensor product of a gate's matrix representation with the identity matrix
        for all qubits except the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.
            gate_mat_rep (np.ndarray): The matrix representation of the gate.

        Returns:
            np.ndarray: The tensor product of the gate's matrix representation with identity matrices.
        """

        if target_qubit == self.circuit_size - 1:
            _mat_rep_ = gate_mat_rep
            _after_I_ = identity(2 ** (self.circuit_size - 1), format='csr')
            _mat_rep_ = kron(_mat_rep_, _after_I_)
        else:
            _before_I_ = identity(2 ** (self.circuit_size - target_qubit - 1), format='csr')
            _mat_rep_ = kron(_before_I_, gate_mat_rep)
            _after_I_ = identity(2 ** target_qubit, format='csr')
            _mat_rep_ = kron(_mat_rep_, _after_I_)

        return csr_matrix(_mat_rep_)

    def _validity_(self, target_qubit: int, control_qubit=None) -> None:
        """
        Check if the target qubit index is valid for the current quantum circuit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) to be checked.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        if target_qubit >= self.circuit_size or target_qubit < 0:
            raise ValueError(f"Target qubit: '{target_qubit}', must be in 0 <= target qubit < circuit size.")

        if control_qubit is not None:
            if control_qubit >= self.circuit_size or control_qubit < 0:
                raise ValueError(f"Control qubit: '{control_qubit}', must be in 0 <= control qubit < circuit size.")
            if control_qubit == target_qubit:
                raise ValueError(f"Control qubit should be different from target qubit.")

    ###############################################################
    ######################## 1 QUBIT GATES ########################
    ###############################################################

    def add_id(self, target_qubit: int) -> None:
        """
        Apply the identity gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._identity_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_x(self, target_qubit: int) -> None:
        """
        Apply the Pauli-X gate (NOT gate) to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._x_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_y(self, target_qubit: int) -> None:
        """
        Apply the Pauli-Y gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._y_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_z(self, target_qubit: int) -> None:
        """
        Apply the Pauli-Z gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._z_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_h(self, target_qubit: int) -> None:
        """
        Apply the Hadamard gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=self._h_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_rx(self, target_qubit: int, angle: float) -> None:
        """
        Apply the rotation around the X-axis gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.
            angle (float): The rotation angle in radians.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _rx_gate_ = np.cos(angle / 2) * self._identity_ - 1j * np.sin(angle / 2) * self._x_gate_
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_rx_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_ry(self, target_qubit: int, angle: float) -> None:
        """
        Apply the rotation around the Y-axis gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.
            angle (float): The rotation angle in radians.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _ry_gate_ = np.cos(angle / 2) * self._identity_ + 1j * np.sin(angle / 2) * self._y_gate_.T
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_ry_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_rz(self, target_qubit: int, angle: float) -> None:
        """
        Apply the rotation around the Z-axis gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.
            angle (float): The rotation angle in radians.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _rz_gate_ = csr_matrix(
            np.array([[np.exp(-1j * angle / 2), 0.0], [0.0, np.exp(1j * angle / 2)]], dtype=self.dtype))
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_rz_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_t(self, target_qubit: int) -> None:
        """
        Add a T gate to the circuit.

        The T gate is a single-qubit gate that introduces a phase of π/4 to the qubit state.

        Args:
            target_qubit (int): The index of the target qubit.

        Returns:
            None
        """
        # Ensure that the target qubit index is valid
        self._validity_(target_qubit=target_qubit)

        # Define the T gate matrix representation
        _t_gate_ = csr_matrix(np.array([[1.0, 0.0], [0.0, np.exp(1j * np.pi / 4)]], dtype=self.dtype))

        # Compute the tensor product matrix representation for the T gate on the target qubit
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_t_gate_)

        # Update the circuit's unitary representation with the T gate's matrix representation
        self._update_circuit_unitary_(_mat_rep_)

    def add_s(self, target_qubit: int) -> None:
        """
        Add an S gate to the circuit.

        The S gate is a single-qubit gate that introduces a phase of π/2 to the qubit state.

        Args:
            target_qubit (int): The index of the target qubit.

        Returns:
            None
        """
        # Ensure that the target qubit index is valid
        self._validity_(target_qubit=target_qubit)

        # Define the S gate matrix representation
        _s_gate_ = csr_matrix(np.array([[1.0, 0.0], [0.0, 1j]], dtype=self.dtype))

        # Compute the tensor product matrix representation for the S gate on the target qubit
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_s_gate_)

        # Update the circuit's unitary representation with the S gate's matrix representation
        self._update_circuit_unitary_(_mat_rep_)

    def add_p(self, target_qubit: int, angle: float) -> None:
        """
        Add a P gate to the circuit.

        The P gate is a single-qubit gate that introduces a phase factor to the qubit state.

        Args:
            target_qubit (int): The index of the target qubit.
            angle (float): The rotation angle in radians by which the qubit state is phase-shifted.

        Returns:
            None
        """
        # Ensure that the target qubit index is valid
        self._validity_(target_qubit=target_qubit)

        # Define the P gate matrix representation
        _p_gate_ = csr_matrix(np.array([[1.0, 0.0], [0.0, np.exp(1j * angle)]], dtype=self.dtype))

        # Compute the tensor product matrix representation for the P gate on the target qubit
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_p_gate_)

        # Update the circuit's unitary representation with the P gate's matrix representation
        self._update_circuit_unitary_(_mat_rep_)

    def add_u(self, target_qubit: int, angle_1: float, angle_2: float, angle_3: float) -> None:
        """
        Add a U3 (Universal Single-Qubit) gate to the circuit.

        The U3 gate is a general single-qubit gate that allows arbitrary rotations
        around the X, Y, and Z axes. It can be used to construct any single-qubit gate.

        Args:
            target_qubit (int): The index of the target qubit.
            angle_1 (float): The rotation angle around the X-axis in radians.
            angle_2 (float): The rotation angle around the Y-axis in radians.
            angle_3 (float): The rotation angle around the Z-axis in radians.

        Returns:
            None
        """
        # Ensure that the target qubit index is valid
        self._validity_(target_qubit=target_qubit)

        # Define the U gate matrix representation
        _u_gate_ = csr_matrix(np.array([[np.cos(angle_1 / 2), -np.exp(1j * angle_3) * np.sin(angle_1 / 2)],
                                        [np.exp(1j * angle_2) * np.sin(angle_1 / 2),
                                         np.exp(1j * (angle_2 + angle_3)) * np.cos(angle_1 / 2)]],
                                       dtype=self.dtype))

        # Compute the tensor product matrix representation for the U gate on the target qubit
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_u_gate_)

        # Update the circuit's unitary representation with the U gate's matrix representation
        self._update_circuit_unitary_(_mat_rep_)

    ###############################################################
    ######################## 2 QUBIT GATES ########################
    ###############################################################

    def add_cnot_OLD(self, target_qubit: int, control_qubit: int) -> None:
        """
        Add a controlled-NOT (CNOT) gate to the quantum circuit.

        Args:
            target_qubit (int): The index of the target qubit (the qubit whose state is flipped if the control qubit is in state 1).
            control_qubit (int): The index of the control qubit.

        Returns:
            None
        """
        self._validity_(target_qubit=target_qubit, control_qubit=control_qubit)

        _flip_ = {'0': '1', '1': '0'}

        # Create a matrix representation of the circuit unitary
        _mat_rep_ = csr_matrix(self.__circuit_unitary__.shape, dtype=self.dtype)

        # Iterate over all possible basis states (bit string permutations)
        for basis_state in generate_bit_string_permutations(n=self.circuit_size):
            # Reversing to match qiskit convention (least significant bit on the right)
            _reversed_state_ = basis_state[::-1]

            if _reversed_state_[control_qubit] == '1':
                # If the control qubit is in state 1, apply the CNOT operation to the basis state
                _rs_basis_state_ = list(basis_state)
                # Note reverse indexing to match qiskit convention
                _rs_basis_state_[-(target_qubit + 1)] = _flip_[basis_state[-(target_qubit + 1)]]
                _rs_basis_state_ = ''.join(_rs_basis_state_)
                _rs_basis_state_vector_ = SYMQState(state=_rs_basis_state_).get_statevector()

                _ls_basis_state_vector_ = SYMQState(state=basis_state).get_statevector()
                _mat_rep_ += csr_matrix(np.outer(_ls_basis_state_vector_, _rs_basis_state_vector_))
            else:
                # If the control qubit is in state 0, apply an identity operation to the basis state
                _basis_state_vector_ = SYMQState(state=basis_state).get_statevector()
                _mat_rep_ += csr_matrix(np.outer(_basis_state_vector_, _basis_state_vector_))

        # Update the circuit unitary with the CNOT operation
        self._update_circuit_unitary_(_mat_rep_)

    def add_cnot(self, target_qubit: int, control_qubit: int) -> None:
        """
        Add a controlled-NOT (CNOT) gate to the quantum circuit.

        Args:
            target_qubit (int): The index of the target qubit (the qubit whose state is flipped if the control qubit is in state 1).
            control_qubit (int): The index of the control qubit.

        Returns:
            None
        """
        self._validity_(target_qubit=target_qubit, control_qubit=control_qubit)

        _flip_ = {'0': '1', '1': '0'}

        # Create a matrix representation of the circuit unitary
        _mat_rep_ = lil_matrix(self.__circuit_unitary__.shape, dtype=self.dtype)

        # Iterate over all possible basis states (bit string permutations)
        for basis_state in generate_bit_string_permutations(n=self.circuit_size):
            # Reversing to match qiskit convention (least significant bit on the right)
            _reversed_state_ = basis_state[::-1]

            if _reversed_state_[control_qubit] == '1':
                # If the control qubit is in state 1, apply the CNOT operation to the basis state
                _rs_basis_state_ = list(basis_state)
                # Note reverse indexing to match qiskit convention
                _rs_basis_state_[-(target_qubit + 1)] = _flip_[basis_state[-(target_qubit + 1)]]
                _row_index_, _col_index_ = int(''.join(_rs_basis_state_), 2), int(basis_state, 2)
                _mat_rep_[_row_index_, _col_index_] = 1
            else:
                # If the control qubit is in state 0, apply an identity operation to the basis state
                _mat_rep_[int(basis_state, 2), int(basis_state, 2)] = 1

        # Update the circuit unitary with the CNOT operation
        self._update_circuit_unitary_(_mat_rep_.tocsr())

    def _get_cnot_mat(self, target_qubit: int, control_qubit: int) -> csr_matrix:
        """
        Get the matrix representation of the controlled-NOT (CNOT) gate for the given target and control qubits.

        Args:
            target_qubit (int): The index of the target qubit (the qubit whose state is flipped if the control qubit is in state 1).
            control_qubit (int): The index of the control qubit.

        Returns:
            np.ndarray: The matrix representation of the CNOT gate.
        """

        self._validity_(target_qubit=target_qubit, control_qubit=control_qubit)

        _flip_ = {'0': '1', '1': '0'}

        # Create a matrix representation of the circuit unitary
        _mat_rep_ = lil_matrix(self.__circuit_unitary__.shape, dtype=self.dtype)

        # Iterate over all possible basis states (bit string permutations)
        for basis_state in generate_bit_string_permutations(n=self.circuit_size):
            # Reversing to match qiskit convention (least significant bit on the right)
            _reversed_state_ = basis_state[::-1]

            if _reversed_state_[control_qubit] == '1':
                # If the control qubit is in state 1, apply the CNOT operation to the basis state
                _rs_basis_state_ = list(basis_state)
                # Note reverse indexing to match qiskit convention
                _rs_basis_state_[-(target_qubit + 1)] = _flip_[basis_state[-(target_qubit + 1)]]
                _row_index_, _col_index_ = int(''.join(_rs_basis_state_), 2), int(basis_state, 2)
                _mat_rep_[_row_index_, _col_index_] = 1
            else:
                # If the control qubit is in state 0, apply an identity operation to the basis state
                _mat_rep_[int(basis_state, 2), int(basis_state, 2)] = 1

        return _mat_rep_.tocsr()

    def add_cx(self, target_qubit: int, control_qubit: int) -> None:
        """
        Add a controlled-X (CNOT) gate to the quantum circuit.

        Args:
            target_qubit (int): The index of the target qubit (the qubit whose state is flipped if the control qubit is in state 1).
            control_qubit (int): The index of the control qubit.

        Returns:
            None
        """

        self.add_cnot(target_qubit=target_qubit, control_qubit=control_qubit)

    def add_cy(self, target_qubit: int, control_qubit: int) -> None:
        """
        Add a controlled-Y gate to the quantum circuit.

        Args:
            target_qubit (int): The index of the target qubit.
            control_qubit (int): The index of the control qubit.

        Returns:
            None
        """
        # Add an Rz gate with angle -pi/2 to implement the controlled-Y gate
        self.add_rz(target_qubit=target_qubit, angle=-np.pi / 2)
        # Add a CNOT gate to perform the controlled-X operation
        self.add_cx(target_qubit=target_qubit, control_qubit=control_qubit)
        # Add another Rz gate with angle pi/2 to complete the controlled-Y operation
        self.add_rz(target_qubit=target_qubit, angle=np.pi / 2)

    def add_cz(self, target_qubit: int, control_qubit: int) -> None:
        """
        Add a controlled-Z gate to the quantum circuit.

        Args:
            target_qubit (int): The index of the target qubit.
            control_qubit (int): The index of the control qubit.

        Returns:
            None
        """
        # Add an H gate (Hadamard) to the target qubit
        self.add_h(target_qubit=target_qubit)
        # Add a CNOT gate to perform the controlled-X operation
        self.add_cx(target_qubit=target_qubit, control_qubit=control_qubit)
        # Add another H gate to the target qubit
        self.add_h(target_qubit=target_qubit)

    def add_swap(self, qubit_1: int, qubit_2: int) -> None:
        """
        Add a SWAP gate to the quantum circuit, exchanging two qubits.

        Args:
            qubit_1 (int): The index of the first qubit to be exchanged.
            qubit_2 (int): The index of the second qubit to be exchanged.

        Returns:
            None
        """
        self._validity_(target_qubit=qubit_1, control_qubit=qubit_2)
        _cnot1_mat_rep_ = self._get_cnot_mat(target_qubit=qubit_1, control_qubit=qubit_2)
        _cnot2_mat_rep_ = self._get_cnot_mat(target_qubit=qubit_2, control_qubit=qubit_1)
        # Apply the CNOT gate with control qubit as qubit_2 and target qubit as qubit_1
        self._update_circuit_unitary_(_cnot1_mat_rep_)
        # Apply the CNOT gate with control qubit as qubit_1 and target qubit as qubit_2
        self._update_circuit_unitary_(_cnot2_mat_rep_)
        # Apply the CNOT gate again with control qubit as qubit_2 and target qubit as qubit_1
        self._update_circuit_unitary_(_cnot1_mat_rep_)

    def add_rzz(self, qubit_1: int, qubit_2: int, angle: float) -> None:
        """
        Add a controlled phase shift gate (Rzz) to the quantum circuit.

        Args:
            qubit_1 (int): The index of the first qubit (control qubit).
            qubit_2 (int): The index of the second qubit (target qubit).
            angle (float): The angle of rotation in radians.

        Returns:
            None
        """
        # Apply a CNOT gate with control qubit as qubit_1 and target qubit as qubit_2
        self.add_cnot(target_qubit=qubit_1, control_qubit=qubit_2)
        # Apply an Rz gate to the target qubit with the specified angle
        self.add_rz(target_qubit=qubit_1, angle=angle)
        # Apply another CNOT gate with control qubit as qubit_1 and target qubit as qubit_2
        self.add_cnot(target_qubit=qubit_1, control_qubit=qubit_2)

    def add_rxx(self, qubit_1: int, qubit_2: int, angle: float) -> None:
        """
        Add a controlled rotation around the XX-axis (Rxx) to the quantum circuit.

        Args:
            qubit_1 (int): The index of the first qubit (control qubit).
            qubit_2 (int): The index of the second qubit (target qubit).
            angle (float): The angle of rotation in radians.

        Returns:
            None
        """
        # Apply Hadamard gate (H) to qubit_1
        self.add_h(target_qubit=qubit_1)
        # Apply Hadamard gate (H) to qubit_2
        self.add_h(target_qubit=qubit_2)
        # Apply Rzz gate with the specified angle to qubit_1 and qubit_2
        self.add_rzz(qubit_1=qubit_1, qubit_2=qubit_2, angle=angle)
        # Apply Hadamard gate (H) to qubit_1
        self.add_h(target_qubit=qubit_1)
        # Apply Hadamard gate (H) to qubit_2
        self.add_h(target_qubit=qubit_2)

    def add_ryy(self, qubit_1: int, qubit_2: int, angle: float) -> None:
        """
        Add a controlled rotation around the YY-axis (Ryy) to the quantum circuit.

        Args:
            qubit_1 (int): The index of the first qubit (control qubit).
            qubit_2 (int): The index of the second qubit (target qubit).
            angle (float): The angle of rotation in radians.

        Returns:
            None
        """
        # Apply Rx gate with angle pi/2 to qubit_1
        self.add_rx(target_qubit=qubit_1, angle=np.pi / 2)
        # Apply Rx gate with angle pi/2 to qubit_2
        self.add_rx(target_qubit=qubit_2, angle=np.pi / 2)
        # Apply Rzz gate with the specified angle to qubit_1 and qubit_2
        self.add_rzz(qubit_1=qubit_1, qubit_2=qubit_2, angle=angle)
        # Apply Rx gate with angle -pi/2 to qubit_1
        self.add_rx(target_qubit=qubit_1, angle=-np.pi / 2)
        # Apply Rx gate with angle -pi/2 to qubit_2
        self.add_rx(target_qubit=qubit_2, angle=-np.pi / 2)

    def add_crz(self, target_qubit: int, control_qubit: int, angle: float) -> None:
        """
        Add a controlled-RZ gate to the circuit.

        Args:
            target_qubit (int): The target qubit index.
            control_qubit (int): The control qubit index.
            angle (float): The rotation angle in radians.

        Returns:
            None
        """
        # Apply RZ gate to the target qubit controlled by the control qubit
        self.add_rz(target_qubit=target_qubit, angle=angle / 2)

        # Apply CX (CNOT) gate with control_qubit controlling the target_qubit
        self.add_cx(target_qubit=target_qubit, control_qubit=control_qubit)

        # Apply RZ gate to the target qubit controlled by the control qubit
        self.add_rz(target_qubit=target_qubit, angle=-angle / 2)

        # Apply CX (CNOT) gate with control_qubit controlling the target_qubit again
        self.add_cx(target_qubit=target_qubit, control_qubit=control_qubit)

    def add_crx(self, target_qubit: int, control_qubit: int, angle: float) -> None:
        """
        Add a controlled-RX gate to the circuit.

        Args:
            target_qubit (int): The target qubit index.
            control_qubit (int): The control qubit index.
            angle (float): The rotation angle in radians.

        Returns:
            None
        """
        # Apply Hadamard gate to the target qubit
        self.add_h(target_qubit=target_qubit)

        # Apply controlled-RZ gate to the target qubit controlled by the control qubit
        self.add_crz(target_qubit=target_qubit, control_qubit=control_qubit, angle=angle)

        # Apply Hadamard gate to the target qubit again
        self.add_h(target_qubit=target_qubit)

    def add_cry(self, target_qubit: int, control_qubit: int, angle: float) -> None:
        """
        Add a controlled-RY gate to the circuit.

        Args:
            target_qubit (int): The target qubit index.
            control_qubit (int): The control qubit index.
            angle (float): The rotation angle in radians.

        Returns:
            None
        """
        # Apply RY gate to the target qubit controlled by the control qubit
        self.add_ry(target_qubit=target_qubit, angle=angle / 2)

        # Apply CX (CNOT) gate with control_qubit controlling the target_qubit
        self.add_cx(target_qubit=target_qubit, control_qubit=control_qubit)

        # Apply RY gate to the target qubit controlled by the control qubit
        self.add_ry(target_qubit=target_qubit, angle=-angle / 2)

        # Apply CX (CNOT) gate with control_qubit controlling the target_qubit again
        self.add_cx(target_qubit=target_qubit, control_qubit=control_qubit)

    def add_exp_of_pauli_string(self, pauli_string: str, theta: float) -> None:
        """
        Adds the exponential of a Pauli string to the quantum circuit.

        Args:
            pauli_string (str): Pauli string containing only 'I', 'X', 'Y', or 'Z'.
            theta (float): Parameter for the exponential.

        Raises:
            ValueError: If pauli_string has an inappropriate length or contains invalid characters.

        The exponential of a Pauli string P is calculated as:
        e^(-i*P*theta) = cos(theta) * I - i * sin(theta) * P

        The resulting unitary gate is added to the quantum circuit.

        """
        if len(pauli_string) != self.circuit_size or not all(c in 'IXYZ' for c in pauli_string):
            raise ValueError(
                f"Invalid Pauli string: {pauli_string} - should have appropriate length and only contain I,X,Y,Z")

        final_circuit = SYMQCircuit(nr_qubits=self.circuit_size, precision=self.precision)
        _target_qubit_ = 0
        for P_i in pauli_string:
            circuit = SYMQCircuit(nr_qubits=self.circuit_size, precision=self.precision)
            if P_i == 'X':
                circuit.add_x(target_qubit=_target_qubit_)
            elif P_i == 'Y':
                circuit.add_y(target_qubit=_target_qubit_)
            elif P_i == 'Z':
                circuit.add_z(target_qubit=_target_qubit_)
            final_circuit = final_circuit + circuit
            _target_qubit_ += 1
        _I_ = identity(2 ** self.circuit_size, format='csr', dtype=self.dtype)
        _P_ = final_circuit.get_circuit_unitary(as_sparse=True)

        final_mat_rep = np.cos(theta) * _I_ - 1j * np.sin(theta) * _P_
        self.__circuit_unitary__ = final_mat_rep @ self.__circuit_unitary__

    def add_pauli_gate(self, pauli: str, target_qubit: int) -> None:
        """
        Adds a general Pauli gate (X, Y, or Z) to the quantum circuit.

        Args:
            pauli (str): The Pauli operator to apply. Should be one of 'X', 'Y', or 'Z'.
            target_qubit (int): The target qubit index.

        Raises:
            ValueError: If the provided Pauli string is invalid.
        """
        if not isinstance(pauli, str) or pauli not in ['X', 'Y', 'Z']:
            raise ValueError(f"Invalid Pauli string: {pauli} - should be str and either 'X', 'Y', or 'Z'")

        if pauli == 'X':
            self.add_x(target_qubit=target_qubit)
        elif pauli == 'Y':
            self.add_y(target_qubit=target_qubit)
        else:
            self.add_z(target_qubit=target_qubit)

    def add_cu(self):
        # TODO: add impl of this.
        pass

    def get_circuit_unitary(self, as_sparse: bool = False) -> Union[np.ndarray, csr_matrix]:
        """
        Get the unitary representation of the quantum circuit.

        Returns:
            np.ndarray: The unitary representation as a numpy array.
        """
        if as_sparse:
            return self.__circuit_unitary__
        return self.__circuit_unitary__.todense()

    def add_gate(self, matrix_representation: Union[csr_matrix, np.ndarray]):
        if isinstance(matrix_representation, csr_matrix):
            if not (2**self.circuit_size == matrix_representation.shape[0] == matrix_representation.shape[1]):
                raise ValueError("Given matrix are not of same size as circuit. For circuit of 'N' qubits"
                                 " the matrix should be (2^N) x (2^N).")

            resulting_circuit = SYMQCircuit(nr_qubits=self.circuit_size)
            resulting_circuit.__circuit_unitary__ = matrix_representation @ self.__circuit_unitary__
            resulting_circuit.dtype = self.dtype
            self.__circuit_unitary__ = resulting_circuit.__circuit_unitary__

        elif isinstance(matrix_representation, np.ndarray):
            if not (2**self.circuit_size == matrix_representation.shape[0] == matrix_representation.shape[1]):
                raise ValueError("Given matrix are not of same size as circuit. For circuit of 'N' qubits"
                                 " the matrix should be (2^N) x (2^N).")

            resulting_circuit = SYMQCircuit(nr_qubits=self.circuit_size)
            resulting_circuit.__circuit_unitary__ = csr_matrix(matrix_representation) @ self.__circuit_unitary__
            resulting_circuit.dtype = self.dtype
            self.__circuit_unitary__ = resulting_circuit.__circuit_unitary__

        else:
            raise ValueError(f' Provided matrix should be either a "csr_matrix" or "numpy array". ')

    def get_state_vector(self):
        """
        Get the state vector of the quantum circuit.

        Returns:
            np.ndarray: The state vector as a numpy array.
        """
        __state_vector__ = np.array([1.0] + [0.0 for _ in range(2 ** self.circuit_size - 1)], dtype=self.dtype)
        return self.get_circuit_unitary() @ __state_vector__

    def get_state_probabilities(self, reverse_states: bool = False, eps: float = 1e-6) -> dict:
        """
        Calculate the probabilities of each basis state in a quantum state.
        N.B. as circuit

        Returns:
            dict: A dictionary containing the basis state as keys and their respective probabilities as values.
        """
        _probs_ = {}
        for n, c_n in enumerate(self.get_state_vector().tolist()[0]):
            _state_string_ = represent_integer_with_bits(number=n, nr_bits=int(self.circuit_size))
            if reverse_states:
                _state_string_ = _state_string_[::-1]
            p = np.power(np.linalg.norm(c_n), 2)
            if p > eps:
                _probs_[_state_string_] = np.power(np.linalg.norm(c_n), 2)
        return _probs_
