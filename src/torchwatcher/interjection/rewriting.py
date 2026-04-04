import copy
import torch.nn as nn

from torch.fx import GraphModule
from typing import Optional, Dict, Any, Callable, Type

from .node_selector import NodeSelector
from ..nn import GradientIdentity
from .tracing import symbolic_trace, DualGraphModule, _get_name


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


def replace(model: nn.Module, selector: NodeSelector,
            create_replacement: Callable[[nn.Module], nn.Module],
            tracer_kwargs: Optional[Dict[str, Any]] = None,
            class_name_prefix="Replaced_") -> DualGraphModule:
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
    is_training = model.training

    if tracer_kwargs is None:
        tracer_kwargs = {}
    if 'leaf_modules' not in tracer_kwargs:
        tracer_kwargs['leaf_modules'] = []
    tracer_kwargs['leaf_modules'].append(GradientIdentity)

    graphs = {}
    graphmodules = {}
    for mode in ["train", "eval"]:
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eval()

        traced = replace_module(model, selector, create_replacement, tracer_kwargs, class_name_prefix)
        graphs[mode] = traced.graph
        graphmodules[mode] = traced

    for mode in ["train", "eval"]:
        # inserting the replacements might have added modules to the traced
        # GraphModule, but not the underlying model. We copy those added modules
        # over.
        # We do this in a separate loop in case changes to val influence train
        model_modules = dict(model.named_modules())
        for name, module in traced.named_children():
            if name not in model_modules:
                model.add_module(name, module)

    # Build the final graph module
    graph_module = DualGraphModule(graphmodules["train"], graphmodules["eval"],
                                   class_name="DualGraphModule_" + _get_name(model))

    # Restore original training mode
    model.train(is_training)
    graph_module.train(is_training)

    return graph_module


def replace_module_native(model, target_class: Type[nn.Module],
                          create_replacement: Callable[[nn.Module], nn.Module],
                          clone=True) -> nn.Module:
    """
    Recursively replaces all instances of target_class with a new module
    created by replacement_factory.

    Args:
        model (nn.Module): The model to modify.
        target_class (type): The class type to look for (e.g., nn.ReLU).
        create_replacement (callable): A function/lambda that returns
                                       a new instance of the replacement module.
        clone: If True, a deep copy of the model is made before modifying.

    Returns:
        model with replacements made.
    """
    if clone:
        model = copy.deepcopy(model)

    # We convert to a list and reverse it to perform a bottom-up replacement.
    # This prevents issues where replacing a parent module makes its
    # children unreachable during iteration.
    modules = list(model.named_modules())

    for name, module in reversed(modules):
        if isinstance(module, target_class):
            # Identify the parent and the attribute name
            if name == "":
                # This is the root model itself
                # Replacing the root is tricky; we can't setattr 'self'
                # easily within this loop, but we can replace its components.
                continue

            # Split the name into path and the final attribute name
            # e.g., 'features.0.conv' -> ['features', '0'], 'conv'
            parts = name.split('.')
            attr_name = parts[-1]

            # Traverse to the parent module
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)

            # Replace the module
            new_module = create_replacement(module)
            setattr(parent, attr_name, new_module)

    return model
