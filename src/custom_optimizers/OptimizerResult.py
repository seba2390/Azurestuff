from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptResult:
    nfev: int
    x: np.ndarray
    fun: float
