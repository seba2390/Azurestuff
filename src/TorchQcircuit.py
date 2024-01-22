import torch
import numpy as np


class TorchQcircuit:
    def __init__(self, n_qubits: int):
        self.dtype = torch.cfloat
        self.circuit_size = n_qubits
        self._identity_ = self.I(n=2)
        self._x_gate_ = torch.tensor(np.array([[0, 1], [1, 0]]), dtype=self.dtype)
        self._h_gate_ = torch.tensor((1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]]), dtype=self.dtype)
        self.__circuit_unitary__ = self.I(n=2 ** n_qubits)
        self.__circuit_unitary__.requires_grad = True

        self.__state_vector__ = torch.tensor(np.array([1.0] + [0.0 for _ in range(2 ** n_qubits - 1)]),
                                             dtype=self.dtype)

    def I(self, n: int) -> torch.Tensor:
        return torch.eye(n=n, dtype=self.dtype)

    def _single_qubit_tensor_prod_matrix_rep_(self, target_qubit: int, gate_mat_rep: torch.Tensor) -> torch.Tensor:
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
            _after_I_ = self.I(n=2 ** (self.circuit_size - 1))
            _mat_rep_ = torch.kron(_mat_rep_, _after_I_)
        else:
            _before_I_ = self.I(n=2 ** (self.circuit_size - target_qubit - 1))
            _mat_rep_ = torch.kron(_before_I_, gate_mat_rep)
            _after_I_ = self.I(n=2 ** target_qubit)
            _mat_rep_ = torch.kron(_mat_rep_, _after_I_)
        return _mat_rep_

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

    def generate_bit_string_permutations(self, n: int) -> str:
        """
        A 'generator' type function that calculates all 2^n-1
        possible bitstring of a 'n-length' bitstring one at a time.
        (All permutations are not stored in memory simultaneously).

        :param n: length of bit-string
        :return: i'th permutation.
        """
        num_permutations = 2 ** n
        for i in range(num_permutations):
            _binary_string_ = bin(i)[2:].zfill(n)
            yield _binary_string_

    def _update_circuit_unitary_(self, gate: torch.Tensor):
        """
        Update the circuit's unitary representation by performing matrix multiplication with a gate.

        Args:
            gate (np.ndarray): The gate's unitary matrix representation.

        Returns:
            None: This function updates the internal circuit's unitary representation in place.
        """
        self.__circuit_unitary__ = gate @ self.__circuit_unitary__

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
        _mat_rep_ = torch.zeros_like(self.__circuit_unitary__)

        # Iterate over all possible basis states (bit string permutations)
        for basis_state in self.generate_bit_string_permutations(n=self.circuit_size):
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

    def add_rx(self, target_qubit: int, angle) -> None:
        """
        Apply the rotation around the X-axis gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.
            angle (float): The rotation angle in radians.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _rx_gate_ = torch.cos(torch.tensor(angle / 2)) * self._identity_ - 1j * torch.sin(torch.tensor(angle / 2)) * self._x_gate_
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_rx_gate_)
        self._update_circuit_unitary_(_mat_rep_)

    def add_rz(self, target_qubit: int, angle) -> None:
        """
        Apply the rotation around the Z-axis gate to the target qubit.

        Args:
            target_qubit (int): The index of the target qubit (0-based) in the quantum circuit.
            angle (float): The rotation angle in radians.

        Raises:
            ValueError: If the target qubit index is out of range for the circuit size.
        """
        self._validity_(target_qubit=target_qubit)
        _rz_gate_ = torch.tensor([[torch.exp(-1j * angle / 2.0), 0.0], [0.0, torch.exp(1j * angle / 2.0)]], dtype=self.dtype)
        _mat_rep_ = self._single_qubit_tensor_prod_matrix_rep_(target_qubit=target_qubit, gate_mat_rep=_rz_gate_)
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
        self.add_rx(target_qubit=qubit_1, angle=torch.pi / 2)
        # Apply Rx gate with angle pi/2 to qubit_2
        self.add_rx(target_qubit=qubit_2, angle=torch.pi / 2)
        # Apply Rzz gate with the specified angle to qubit_1 and qubit_2
        self.add_rzz(qubit_1=qubit_1, qubit_2=qubit_2, angle=angle)
        # Apply Rx gate with angle -pi/2 to qubit_1
        self.add_rx(target_qubit=qubit_1, angle=-torch.pi / 2)
        # Apply Rx gate with angle -pi/2 to qubit_2
        self.add_rx(target_qubit=qubit_2, angle=-torch.pi / 2)

    def get_circuit_unitary(self) -> torch.Tensor:
        """
        Get the unitary representation of the quantum circuit.

        Returns:
            np.ndarray: The unitary representation as a numpy array.
        """
        return self.__circuit_unitary__
