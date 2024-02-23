from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass(frozen=True)
class SimResult:
    N: int
    k: int
    L: int
    alpha: float
    w_nnn: bool
    CP_VQA: dict
    QAOA: dict
    QAOA_HYBRID: dict
