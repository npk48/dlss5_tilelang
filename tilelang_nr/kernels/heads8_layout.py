"""CPU-only immutable completion ownership for EightJoint, from native metadata."""
import numpy as np
from tilelang_nr.common.up8_layout import physical_inverse


def pool_rows(pi):
    pi = np.asarray(pi, dtype=np.int32).reshape(-1, 16, 4, 256)
    rows = np.where(pi[..., 0] >= 0, pi[..., 0] // 256, -1).astype(np.int32)
    expected = np.where(rows[..., None] >= 0, rows[..., None] * 256 + np.arange(256), -1)
    if not np.array_equal(pi, expected):
        raise ValueError('DS8 non-row-separable pool metadata')
    active = rows >= 0
    lo = np.min(np.where(active, rows // 32, 2), axis=-1)
    hi = np.max(np.where(active, rows // 32, -1), axis=-1)
    if np.any((hi >= 0) & (lo != hi)):
        raise ValueError('DS8 pool crosses the native M32 completion boundary')
    return np.ascontiguousarray(rows)


def up_rows(ui, inverse, raw_map, projection_halves):
    ui = np.asarray(ui, dtype=np.int32).reshape(-1, 512)
    inverse = np.asarray(inverse, dtype=np.int32).reshape(-1, 2)
    raw_map = np.asarray(raw_map, dtype=np.int32).reshape(-1, 64, 256)
    mapped = physical_inverse(ui, inverse, projection_halves)
    projected = np.where(
        raw_map[..., 0] >= 0, inverse[np.maximum(raw_map[..., 0], 0), 0] // 256, -1
    )
    compact = np.full((len(projected), 16), -1, np.int32)
    physical = compact.copy()
    local = np.full_like(projected, -1)
    for tile, row in enumerate(projected):
        owned = np.unique(row[row >= 0])
        if len(owned) > 16:
            raise ValueError('UP8 projection has more than M16 owned rows')
        compact[tile, :len(owned)] = owned
        physical[tile, :len(owned)] = ui[owned, 0] // 16
        for i, n in enumerate(owned):
            local[tile, row == n] = i
    # A raw pair consumes the same locally computed Half as the corrected physical inverse.
    t, r, c = np.indices(raw_map.shape)
    valid = raw_map >= 0
    expected = physical[t, np.maximum(local[t, r], 0)] * 256 + c
    if not np.array_equal(mapped[raw_map[valid], 0], expected[valid]):
        raise ValueError('UP8 local consumer differs from corrected physical inverse')
    if not np.array_equal(mapped[raw_map[valid], 1], c[valid]):
        raise ValueError('UP8 gate ownership differs from native metadata')
    return tuple(np.ascontiguousarray(x) for x in (compact, physical, local))
