import torch.nn as nn
from torch.fx import GraphModule
from typing import Optional, Dict, Any, Callable
from .node_selector import NodeSelector
from ..nn import GradientIdentity
from .tracing import symbolic_trace


def replace_module(model: nn.Module, selector: NodeSelector,
                   create_replacement: Callable[[nn.Module], nn.Module],
                   tracer_kwargs: Optional[Dict[str, Any]] = None,
                   class_name_prefix="Replaced_") -> GraphModule:
    """Replace a module with another module within the graph.

    Args:
        model: the model
        selector: node selector. This must only match 'call_module' nodes.
        create_replacement: a function to create the replacement module. It will
                            be called with the original module as an argument.
        tracer_kwargs: extra keyword arguments to pass to the tracing module.
                       The 'leaf_modules' key is particularly useful when you
                       want to replace a custom module without tracing inside
                       it.

    Returns:
        the new model with replacements made.
    """
    if tracer_kwargs is None:
        tracer_kwargs = {}
    if 'leaf_modules' not in tracer_kwargs:
        tracer_kwargs['leaf_modules'] = []
    tracer_kwargs['leaf_modules'].append(GradientIdentity)

    traced = symbolic_trace(model, tracer_kwargs=tracer_kwargs)
    for node in traced.graph.nodes:
        if selector.fn((traced, node)):
            assert node.op == 'call_module'
            replacement = create_replacement(traced.get_submodule(node.target))
            traced.add_submodule(node.target, replacement)

    traced.graph.eliminate_dead_code()
    traced.delete_all_unused_submodules()
    traced.recompile()

    traced.__class__.__name__ = class_name_prefix + model.__class__.__name__

    return traced