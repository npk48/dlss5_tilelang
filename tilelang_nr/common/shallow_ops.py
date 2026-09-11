"""Intrinsic expressions only; packet algorithms live in shallow.py TileLang IR."""
import tilelang.language as T


def call(name, *args, dtype="uint32"):
    return T.call_extern(dtype, "nr_tl_shallow::" + name, *args)


def pack(a, b):
    return call("pack", a, b)


def unpack(a, i):
    return T.Cast("float16", call("unpack", a, i, dtype="float32"))


def splat(a):
    return pack(T.float16(a), T.float16(a))


def add(a, b):
    return call("add", a, b)


def mul(a, b):
    return call("mul", a, b)


def fma(a, b, c):
    return call("fma", a, b, c)


def e4four(a, b):
    return call("e4pair", a) | (call("e4pair", b) << 16)


def shfl(a, lane):
    return call("shfl", a, lane)


def une4(a):
    return T.Cast("float16", call("une4", a, dtype="float32"))


def hm(a, b):
    return T.Cast("float16", call("hmul", a, b, dtype="float32"))


def ha(a, b):
    return T.Cast("float16", call("hadd", a, b, dtype="float32"))


def hf(a, b, c):
    return T.Cast("float16", call("hfma", a, b, c, dtype="float32"))
