import collections


def x_if_xp_is_none(x, xp):
    return x if xp is None else xp


def unpack(result):
    return result[0] if isinstance(result, tuple) and len(result) == 1 else result


def pack(result):
    if isinstance(result, collections.abc.Sequence):
        return result
    return (result, )