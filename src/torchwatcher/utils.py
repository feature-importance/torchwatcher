import collections
from contextlib import contextmanager

from torch import nn


@contextmanager
def preserve_training_modes(*roots: nn.Module):
    """Restore exact per-module training modes after a temporary change.

    Modules are restored after all of their parents, including when a child is
    shared, so each module's final ``train`` call receives its saved mode.
    Direct flag restoration then covers custom ``train`` implementations which
    do not delegate to :class:`torch.nn.Module`.
    """
    modules: list[nn.Module] = []
    modes: dict[nn.Module, bool] = {}
    children: dict[nn.Module, set[nn.Module]] = {}

    def discover(module: nn.Module):
        if module in modes:
            return
        modules.append(module)
        modes[module] = module.training
        children[module] = set()
        for child in module._modules.values():
            if child is None:
                continue
            children[module].add(child)
            discover(child)

    for root in roots:
        discover(root)

    indegrees = {module: 0 for module in modules}
    for module_children in children.values():
        for child in module_children:
            indegrees[child] += 1

    pending = collections.deque(
        module for module in modules if indegrees[module] == 0
    )
    restore_order = []
    while pending:
        module = pending.popleft()
        restore_order.append(module)
        for child in children[module]:
            indegrees[child] -= 1
            if indegrees[child] == 0:
                pending.append(child)

    # Module.train() cannot handle cyclic module registration either, but keep
    # restoration best-effort if a custom module permits such a structure.
    restore_order.extend(
        module for module in modules if module not in restore_order
    )

    try:
        yield
    finally:
        for module in restore_order:
            module.train(modes[module])
        for module in modules:
            module.training = modes[module]


def x_if_xp_is_none(x, xp):
    """returns x if xp is None, otherwise returns xp"""
    return x if xp is None else xp


def unpack(result):
    """unpacks a single-item tuple, and returns the input if it's not a single
    item tuple
    """
    return result[0] if (isinstance(result, tuple) and
                         len(result) == 1) else result


def unpack_call(args, kwargs):
    """Represent positional and keyword call inputs without changing the
    existing positional-only convention.
    """
    positional = unpack(args)
    if not kwargs:
        return positional
    if not args:
        return kwargs
    return positional, kwargs


def pack(result):
    """packs an item into a single-item tuple unless it's already a tuple"""
    if isinstance(result, collections.abc.Sequence):
        return result
    return (result,)


def true(*_):
    """Always returns True. Useful for selecting nodes in a graph."""
    return True


def false(*_):
    """Always returns False. Useful for (not) selecting nodes in a graph."""
    return False

