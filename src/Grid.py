from typing import List, Tuple
import numpy as np


class Grid:
    def __init__(self,
                 N_qubits: int = None,
                 Rows: int = None,
                 Cols: int = None):

        self.using_N_qubits = False
        if N_qubits is None:
            if Rows is None or Cols is None:
                raise ValueError(
                    'When the grid is not initialized using "N_qubits", it should be initialized, using "Rows" and '
                    '"Cols".')
        else:
            if int(np.sqrt(N_qubits)) - np.sqrt(N_qubits) != 0:
                raise ValueError(
                    f'When grid is initialized w. "N_qubits" it is assumed to be a square grid, and therefore '
                    f'"N_qubits" must be a perfect square integer. ')
            self.N_qubits = N_qubits
            self.using_N_qubits = True
        if Rows is not None or Cols is not None:
            if N_qubits is not None:
                raise ValueError(f'When specifying the grid using "Rows" & "Cols", "N_qubits" should not be specified.')
            if Rows is None or Cols is None:
                raise ValueError(
                    f'When specifying the grid without "N_qubits", both "Rows" & "Cols" has to be specified.')
            self.rows, self.cols = Rows, Cols
            self.initialization_strategy = None

    def get_grid_indexing(self) -> np.ndarray:
        if self.using_N_qubits:
            root = int(np.sqrt(self.N_qubits))
            return np.array([[col + row * root for col in range(root)] for row in range(root)])
        else:
            return np.array([[col + row * self.cols for col in range(self.cols)] for row in range(self.rows)])

    def get_NN_indices(self) -> List[Tuple[int, int]]:
        """ Returns pairs of indices corresponding to
        Nearest Neighbor interactions in the grid structure """
        grid_indices = self.get_grid_indexing()
        rows, cols = grid_indices.shape
        NN_pairs = []
        for row in range(rows):
            for col in range(cols):
                if row == rows - 1:
                    if col < cols - 1:
                        NN_pairs.append((grid_indices[row, col], grid_indices[row, col + 1]))
                elif col == cols - 1:
                    NN_pairs.append((grid_indices[row, col], grid_indices[row + 1, col]))
                else:
                    NN_pairs.append((grid_indices[row, col], grid_indices[row + 1, col]))
                    NN_pairs.append((grid_indices[row, col], grid_indices[row, col + 1]))
        return NN_pairs

    def set_initialization_strategy(self, strategy: np.ndarray) -> None:
        if strategy.shape != self.get_grid_indexing().shape:
            raise ValueError('Strategy should be of same dimensions as the grid.')
        if np.any((strategy != 0) & (strategy != 1)):
            raise ValueError('Strategy should binary grid.')
        self.initialization_strategy = strategy

    def get_initialization_strategy(self) -> np.ndarray:
        if self.initialization_strategy is None:
            raise RuntimeError('Initialization strategy not yet defined.')
        return self.initialization_strategy

    def get_initialization_indices(self) -> List[int]:
        if self.initialization_strategy is None:
            raise RuntimeError('Initialization strategy not yet defined.')
        return self.get_grid_indexing()[np.where(self.initialization_strategy == 1)].flatten().tolist()


