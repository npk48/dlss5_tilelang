"""Bounded vector-packet routing; distinct from the strict scalar route."""
import tilelang.language as T
from tilelang_nr.common.wide import op, permute


def route_groups(bits):
    """Static destination bytes grouped by source (word,lane), not source byte."""
    groups = {}
    for byte in range(4):
        source = sum(((byte >> j) & 1) << k for j, k in enumerate(bits))
        groups.setdefault(source & ~3, []).append(byte)
    return tuple(tuple(group) for group in groups.values())


def _prmt(a, b, selector):
    return T.call_extern('uint32', 'wide_packet_prmt', a, b, selector)


def local_route_word(P, word, lane, bits):
    groups = route_groups(bits)
    bank_bits = tuple(k - 7 for j, k in enumerate(bits) if 2 <= j < 7 and k >= 7)
    fetched = []
    for group in groups:
        byte = group[0]
        ix = permute(word * 128 + lane * 4 + byte, bits)
        base = permute(word * 128 + byte, bits) // 128
        if bank_bits:
            assert len(bank_bits) == 1
            # Original helper: TWO unconditional full-mask shuffles, then select.
            value = op(
                'shfl_pair',
                P[base],
                P[base | (1 << bank_bits[0])], (ix // 4) % 32,
                ix // 128 != base
            )
        elif all(bits[j] == j for j in range(2, 7)):
            value = P[base]
        else:
            value = op('shfl', P[base], (ix // 4) % 32)
        fetched.append(T.alloc_var('uint32', init=value)[0])
    assert len(groups) in (2, 4)
    pieces = []
    for pair in range(len(groups) // 2):
        selector = T.uint32(0)
        for b in range(4):
            for half in range(2):
                if b in groups[pair * 2 + half]:
                    ix = permute(word * 128 + lane * 4 + b, bits)
                    selector = selector | (T.cast(ix % 4 + half * 4, 'uint32') << (b * 4))
        pieces.append(
            T.alloc_var('uint32', init=_prmt(fetched[pair * 2], fetched[pair * 2 + 1], selector))[0]
        )
    if len(pieces) == 1:
        return pieces[0]
    selector = sum((b + (4 if b in groups[2] + groups[3] else 0)) << (b * 4) for b in range(4))
    return _prmt(pieces[0], pieces[1], selector)


def shared_packed(S, head, word, lane):
    # Inverse hidden packet: two existing words, one PRMT, no byte reloads.
    base = (word // 2) * 2
    x = T.reinterpret(S[head, (base // 4) * 128 + lane * 4 + base % 4], 'uint32')
    y = T.reinterpret(S[head, (base // 4) * 128 + lane * 4 + base % 4 + 1], 'uint32')
    return _prmt(x, y, T.if_then_else(word % 2 == 0, 0x5410, 0x7632))
