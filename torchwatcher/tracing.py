import copy

import torch
from torch import nn
from torch.fx import symbolic_trace
from torchvision.models import resnet18

from torchwatcher.interjection import ForwardInterjection, WrappedForwardInterjection, \
    WrappedForwardBackwardInterjection


def get_fresh_qualname(traced: torch.fx.GraphModule, prefix: str) -> str:
    """
    Generate a unique qualified name for a module that will be added to the graph.
    """
    i = 0
    while True:
        qualname = f"{prefix}{i}"
        i += 1
        if not hasattr(traced, qualname):
            break

    return qualname


def find_nodes(graph: torch.fx.Graph, op):
    """
    Find nodes matching a given operation.
    """
    return list(filter(lambda n: n.op == op, graph.nodes))


def extract_node(traced: torch.fx.GraphModule, target_node: torch.fx.Node) -> torch.fx.GraphModule:
    """
    Extract a node into its own module/graph, re-writing inputs as appropriate.
    """
    gm = copy.deepcopy(traced)

    # Set new inputs with placeholders
    for node in gm.graph.nodes:
        if node.name == target_node.all_input_nodes[0].name:  # FIXME: need to deal with multiple inputs
            node.op, node.target, node.args, node.kwargs = 'placeholder', node.name, (), {}

    output_node = find_nodes(gm.graph, op='output')[0]
    output_node.args = tuple([node for node in gm.graph.nodes if node.name == target_node.name])
    gm.graph.eliminate_dead_code()

    # Remove unused placeholders
    for node in find_nodes(gm.graph, op='placeholder'):
        if node.name != target_node.all_input_nodes[0].name:  # FIXME: need to deal with multiple inputs
            gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
    gm.delete_all_unused_submodules()
    gm.recompile()

    return gm


def add_interjection(traced, interjection):
    """
    Add (or find if it already exists) the given interjection and return its name. A Unique name is generated for the
    cases it's not in the traced module already.
    """
    # Any given interjection is added as a module just once; when it is called, the name of the node is passed along so
    # that if it's reused across multiple points the user can tell where it's coming from.
    for name, module in traced.named_modules():
        if module == interjection:
            return name

    interjection_name = get_fresh_qualname(traced, type(interjection).__name__)
    traced.add_module(interjection_name, interjection)
    return interjection_name


def insert_interjection(traced, node, interjection):
    """
    Insert an interjection into the traced module at a given node. If the interjection is of the wrapped type, then the
    node is replaced with interjection (which wraps the original node); otherwise the node is inserted immediately after
    the node.
    """
    interjection_name = add_interjection(traced, interjection)

    if hasattr(interjection, '_wrapped'):
        with traced.graph.inserting_after(node):
            extracted = extract_node(traced, node)
            # new node to represent the call to the interjection
            new_node = traced.graph.call_module(interjection_name, (node.name, node,))
            # register the extracted node that we wrap
            interjection.wrap(node.name, extracted)
            # clean everything up by replacing uses and inputs, then removing the original node
            node.replace_all_uses_with(new_node)
            new_node.replace_input_with(new_node, node.all_input_nodes[0])  # FIXME: need to deal with multiple inputs
        traced.graph.erase_node(node)
    else:
        with traced.graph.inserting_after(node):
            # create the interjection node after the current one
            new_node = traced.graph.call_module(interjection_name, (node.name, node,))
            # and hook it to the graph
            node.replace_all_uses_with(new_node)
            new_node.replace_input_with(new_node, node)


def interject_by_module_class(network, target_module_class, interjection):
    """Adds an interjection to all nodes that represent a particular nn.Module"""
    traced = symbolic_trace(network)
    modules = dict(traced.named_modules())

    for node in traced.graph.nodes:
        if node.target not in modules:
            continue

        if type(modules[node.target]) == target_module_class:
            insert_interjection(traced, node, interjection)

    traced.recompile()
    return traced


class MyForwardInterjection(ForwardInterjection):
    def process(self, name, input):
        print(name, input.shape)
        # return args[0]


net = resnet18()
net2 = interject_by_module_class(net, nn.Conv2d, MyForwardInterjection())
r = net2(torch.zeros(1, 3, 224, 244))


class MyWrappedForwardInterjection(WrappedForwardInterjection):
    def process(self, name, input, output):
        print(name, input.shape)


net = resnet18()
net2 = interject_by_module_class(net, nn.Conv2d, MyWrappedForwardInterjection())
net2(torch.zeros(1, 3, 224, 244))


class MyWrappedForwardBackwardInterjection(WrappedForwardBackwardInterjection):
    def process(self, name, input, output):
        print("forward", name, input.shape)

    def process_backward(self, name, grad_input, grad_output):
        print("backward", name, grad_input.shape if grad_input is not None else None)


net = resnet18()
net2 = interject_by_module_class(net, nn.Conv2d, MyWrappedForwardBackwardInterjection())
r = net2(torch.zeros(1, 3, 224, 244))
loss = torch.nn.functional.cross_entropy(r, torch.tensor([0], dtype=torch.long))
loss.backward()

