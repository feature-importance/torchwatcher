import collections


def x_if_xp_is_none(x, xp):
    """returns x if xp is None, otherwise returns xp"""
    return x if xp is None else xp


def unpack(result):
    """unpacks a single-item tuple, and returns the input if its not a single
    item tuple"""
    return result[0] if (isinstance(result, tuple) and
                         len(result) == 1) else result


def pack(result):
    """packs an item into a single-item tuple unless its already a tuple"""
    if isinstance(result, collections.abc.Sequence):
        return result
    return (result,)


def true(*_):
    """Always returns True. Useful for selecting nodes in a graph."""
    return True


def false(*_):
    """Always returns False. Useful for (not) selecting nodes in a graph."""
    return False
