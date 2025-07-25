import copy
from typing import Type, Optional, Dict, Any, List, Union, Callable

import torch
from torch import fx
from torch import nn
from torchvision.models.feature_extraction import NodePathTracer, \
    _set_default_tracer_kwargs
from torchwatcher.interjection.node_selector import node_selector

from .interjection import Interjection
from .node_selector import NodeSelector, matches_module_class
from ..nn import GradientIdentity


class DualGraphModule(nn.Module):
    def __init__(self, train_module: nn.Module, eval_module: nn.Module,
                 class_name="DualGraphModule"):
        super().__init__()

        self.__class__.__name__ = class_name

        self.train_module = train_module
        self.eval_module = eval_module

    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_module(*args, **kwargs)
        else:
            return self.eval_module(*args, **kwargs)


def _get_name(model: Union[torch.nn.Module, Callable[..., Any]]) -> str:
    """Get the name of a model; used for naming the wrapped GraphModule.

    Args:
        model: the model

    Returns:
        the name
    """
    if isinstance(model, nn.Module):
        return model.__class__.__name__
    else:
        return model.__name__


def symbolic_trace(model: Union[torch.nn.Module, Callable[..., Any]],
                   tracer_kwargs: Optional[Dict[str, Any]] = None,
                   concrete_args: Optional[Dict[str, Any]] = None
                   ) -> fx.GraphModule:
    """Custom symbolic tracing using functionality in torchvision's feature
    extraction framework.

    Args:
        model: the model to trace
        tracer_kwargs: extra arguments for the tracer
        concrete_args
    """
    # Instantiate our NodePathTracer and use that to trace the model
    tracer_kwargs = _set_default_tracer_kwargs(tracer_kwargs)
    tracer = NodePathTracer(**tracer_kwargs)
    graph = tracer.trace(model, concrete_args=concrete_args)

    graph_module = fx.GraphModule(tracer.root, graph, "traced_"+_get_name(model))

    # We store the qualified names as an extra attribute of each node,
    # allowing a NodeSelector to access
    # without extra external information
    for node in graph_module.graph.nodes:
        if node in tracer.node_to_qualname:
            node.qualified_name = tracer.node_to_qualname[node]

    return graph_module


def get_fresh_qualname(traced: fx.GraphModule, prefix: str) -> str:
    """Generate a unique qualified name for a module that will be added to the
    graph.

    Args:
        traced: the traced module
        prefix: prefix for qualified names

    Returns:
        the generated name
    """

    i = 0
    while True:
        qualname = f"{prefix}{i}"
        i += 1
        if not hasattr(traced, qualname):
            break

    return qualname


def find_nodes(graph: fx.Graph, op) -> List[fx.Node]:
    """Find nodes in the graph matching a given operation.

    Args:
        graph: graph to search
        op: operation to search for

    Returns:
        matching nodes
    """

    return list(filter(lambda n: n.op == op, graph.nodes))


def extract_node(graph_module: fx.GraphModule,
                 target_node: fx.Node) -> fx.GraphModule:
    """Extract a node into its own module/graph, re-writing inputs as appropriate.

    This trims all extraneous bits from the graph, just leaving the node of
    interest.

    Args:
        graph_module: the graph module to extract from
        target_node: the node to extract

    Returns:
        the new GraphModule with just that single node in it
    """
    gm = copy.deepcopy(graph_module)

    # Set new inputs with placeholders
    for node in gm.graph.nodes:
        if node.name == target_node.all_input_nodes[0].name:
            # FIXME: need to deal with multiple inputs
            node.op, node.target, node.args, node.kwargs = ('placeholder',
                                                            node.name, (), {})

    output_node = find_nodes(gm.graph, op='output')[0]
    output_node.args = tuple(
        [node for node in gm.graph.nodes if node.name == target_node.name])
    gm.graph.eliminate_dead_code()

    # Remove unused placeholders
    for node in find_nodes(gm.graph, op='placeholder'):
        if node.name != target_node.all_input_nodes[0].name:
            # FIXME: need to deal with multiple inputs
            gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
    gm.delete_all_unused_submodules()
    gm.recompile()

    return gm


def add_interjection(graph_module: fx.GraphModule,
                     interjection: Interjection) -> str:
    """Add (or find if it already exists) the given interjection and return its
    name. A Unique name is generated for the cases it's not in the traced
    module already.

    Any given interjection is added as a module just once; when it is
    called, the name of the node is passed along so that if it's reused
    across multiple points the user can tell where it's coming from.

    Args:
        graph_module: the graph module to add the interjection to
        interjection: the interjection to add

    Returns:
        the name of the interjection module within the graph_module
    """
    for name, module in graph_module.named_modules():
        if module == interjection:
            return name

    interjection_name = get_fresh_qualname(graph_module,
                                           type(interjection).__name__)

    graph_module.add_module(interjection_name, interjection)

    return interjection_name


def handle_inplace(graph_module: fx.GraphModule, node: fx.Node) -> fx.Node:
    # handle modules with 'inplace' flags
    if (node.op == 'call_module' and
            hasattr(graph_module.get_submodule(node.target), 'inplace')):
        graph_module.get_submodule(node.target).inplace = False
        return node

    # TODO: call_function with a kwarg called 'inplace'

    # TODO: call_function with a name ending in underscore

    return node


def preferred_name(node: fx.Node) -> str:
    """Get the preferred name for a given node. Preferred name
     is the qualified name if it exists. Otherwise, it's the regular
     node name.

    Args:
        node: the node

    Returns:
        the preferred name
    """
    if hasattr(node, 'qualified_name'):
        return node.qualified_name
    return node.name


def insert_interjection(graph_module: fx.GraphModule, node: fx.Node,
                        interjection: Interjection):
    """Insert an interjection into the traced module at a given node. If the
    interjection is of the wrapped type, then the node is replaced with
    interjection (which wraps the original node); otherwise the new node is
    inserted immediately after the node.

    Args:
        graph_module: the traced module
        node: the insertion point
        interjection: the interjection to insert
    """

    interjection_name = add_interjection(graph_module, interjection)

    node = handle_inplace(graph_module, node)

    if hasattr(interjection, '_wrapped'):
        with graph_module.graph.inserting_after(node):
            extracted = extract_node(graph_module, node)
            # new node to represent the call to the interjection
            args = (preferred_name(node), node,)
            new_node = graph_module.graph.call_module(interjection_name, args)
            # register the extracted node that we wrap
            interjection.register(preferred_name(node), extracted)
            # clean everything up by replacing uses and inputs, then removing
            # the original node
            node.replace_all_uses_with(new_node)
            # FIXME: need to deal with multiple inputs
            new_node.replace_input_with(new_node, node.all_input_nodes[0])
        graph_module.graph.erase_node(node)
    else:
        with graph_module.graph.inserting_after(node):
            if node.op == 'call_module':
                modules = dict(graph_module.named_modules())
                module = modules[node.target]
            else:
                module = None
            interjection.register(preferred_name(node), module)

            # create the interjection node after the current one
            new_node = graph_module.graph.call_module(interjection_name,
                                                      (preferred_name(node), node,))
            # and hook it to the graph
            node.replace_all_uses_with(new_node)
            new_node.replace_input_with(new_node, node)


def interject_by_module_class(model: nn.Module,
                              target_module_class: Type[nn.Module],
                              interjection: Interjection,
                              tracer_kwargs: Optional[Dict[str, Any]] = None) -> DualGraphModule:
    """Adds an interjection to all nodes that represent a particular nn.Module
    within the provided model.

    Args:
        model: the model to add the interjection(s) to
        target_module_class: the module to match for the insertion point
        interjection: the interjection to insert
        tracer_kwargs: extra keyword arguments to pass to the tracing module.
                       The 'leaf_modules' key is particularly useful when you
                       want to interject a custom module without tracing inside
                       it.

    Returns:
        the interjected model
    """

    selector = matches_module_class(target_module_class)
    return interject_by_match(model, selector, interjection, tracer_kwargs=tracer_kwargs)


def interject_by_match(model: nn.Module, selector: NodeSelector,
                       interjection: Interjection,
                       tracer_kwargs: Optional[Dict[str, Any]] = None) -> DualGraphModule:
    """Adds an interjection to all nodes that represent a particular nn.Module

    Args:
        model: the model
        selector: node selector
        interjection: the interjection
        tracer_kwargs: extra keyword arguments to pass to the tracing module.
                       The 'leaf_modules' key is particularly useful when you
                       want to interject a custom module without tracing inside
                       it.

    Returns:
        the interjected model
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

        traced = symbolic_trace(model, tracer_kwargs=tracer_kwargs)
        for node in traced.graph.nodes:
            if selector.fn((traced, node)):
                insert_interjection(traced, node, interjection)

        traced.graph.eliminate_dead_code()
        traced.delete_all_unused_submodules()
        traced.recompile()
        graphs[mode] = traced.graph
        graphmodules[mode] = traced

    for mode in ["train", "eval"]:
        # inserting the interjections will have added modules to the traced
        # GraphModule, but not the underlying model. We copy those added modules
        # over.
        # We do this in a separate loop in case changes to val influence train
        model_modules = dict(model.named_modules())
        for name, module in traced.named_children():
            if name not in model_modules:
                model.add_module(name, module)

    # Build the final graph module
    graph_module = DualGraphModule(graphmodules["train"], graphmodules["eval"],
                                   class_name="DualGraphModule_"+_get_name(model))

    # Restore original training mode
    model.train(is_training)
    graph_module.train(is_training)

    return graph_module


def interject_by_name(model: nn.Module, name: str,
                      interjection: Interjection,
                      tracer_kwargs: Optional[Dict[str, Any]] = None) -> DualGraphModule:
    """Adds an interjection to the particular named node

    Args:
        model: the model
        name: the node name
        interjection: the interjection
        tracer_kwargs: extra keyword arguments to pass to the tracing module.
                       The 'leaf_modules' key is particularly useful when you
                       want to interject a custom module without tracing inside
                       it.

    Returns:
        the interjected model
    """
    return interject_by_match(model,
                              node_selector.matches_qualified_name(name),
                              interjection, tracer_kwargs=tracer_kwargs)


def trim(network: fx.GraphModule):
    # TODO: implement. This should trim the tail of the graph so that it
    #  stops after the last interjection - but only if needed (like when not
    #  training!)
    pass
