"""CPU address-only translation for the opt-in UP8 physical projection rail."""
import numpy as np


def physical_inverse(ui, inverse, projection_halves):
    ui = np.asarray(ui, dtype=np.int32).reshape(-1, 512)
    inverse = np.asarray(inverse, dtype=np.int32).reshape(-1, 2)
    physical = ui[:, 0].astype(np.int64) // 16
    if np.any(ui[:, 0] < 0) or np.any(ui[:, 0] % 16):
        raise ValueError('UP8 UI does not identify aligned physical planar rows')
    if len(np.unique(physical)) != len(physical) or np.any(
        (physical + 1) * 256 > projection_halves):
        raise ValueError('UP8 projection row ownership or extent is invalid')
    pi = inverse[:, 0].astype(np.int64)
    if np.any(pi < 0) or np.any(pi // 256 >= len(physical)):
        raise ValueError('UP8 compact inverse is outside UI')
    result = inverse.copy()
    result[:, 0] = physical[pi // 256] * 256 + pi % 256
    return np.ascontiguousarray(result)
