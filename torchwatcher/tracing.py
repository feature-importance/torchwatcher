import copy

import torch
from torch import nn
from torch.fx import symbolic_trace
from torchvision.models import resnet18

from torchwatcher.interjection import ForwardInterjection, WrappedForwardInterjection


def get_fresh_qualname(traced: torch.fx.GraphModule, prefix: str) -> str:
    """
    Generate a unique qualified name for a module that will be added to the graph.

    Args:
        traced ():
        prefix ():

    Returns:

    """
    i = 0
    while True:
        qualname = f"{prefix}{i}"
        i += 1
        if not hasattr(traced, qualname):
            break

    return qualname

def find_nodes(graph: torch.fx.Graph, op):
    return list(filter(lambda n: n.op == op ,graph.nodes))


def extract_node(traced: torch.fx.GraphModule, target_node: torch.fx.Node) -> torch.fx.GraphModule:
    gm = copy.deepcopy(traced)

    # Set new inputs with placeholders
    for node in gm.graph.nodes:
        if node.name == target_node.prev.name:
            node.op, node.target, node.args, node.kwargs = 'placeholder', node.name, (), {}

    output_node = find_nodes(gm.graph, op='output')[0]
    output_node.args = tuple([node for node in gm.graph.nodes if node.name == target_node.name])
    gm.graph.eliminate_dead_code()

    # Remove unused placeholders
    for node in find_nodes(gm.graph, op='placeholder'):
        if node.name != target_node.prev.name:
            gm.graph.erase_node(node)
    new_module = torch.fx.GraphModule(gm, gm.graph)

    return new_module

def insert_interjection(traced, node, interjection, interjection_name):
    if hasattr(interjection, '_wrapped'):
        with traced.graph.inserting_after(node):
            # new node to represent the call to the interjection
            new_node = traced.graph.call_module(interjection_name, (node.name, node,))
            # register the extracted node that we wrap
            interjection.wrap(node.name, extract_node(traced, node))
            # clean everything up by replacing uses and inputs, then removing the original node
            node.replace_all_uses_with(new_node)
            new_node.replace_input_with(new_node, node.prev)
        traced.graph.erase_node(node)
    else:
        with traced.graph.inserting_after(node):
            new_node = traced.graph.call_module(interjection_name, (node.name, node,))
            node.replace_all_uses_with(new_node)
            new_node.replace_input_with(new_node, node)


def interject_by_module_class(network, target_module_class, interjection):
    traced = symbolic_trace(network)
    modules = dict(traced.named_modules())

    # Any given interjection is added as a module just once; when it is called, the name of the node is passed along so
    # that if it's reused across multiple points the user can tell where its coming from.
    interjection_name = get_fresh_qualname(traced, type(interjection).__name__)
    traced.add_module(interjection_name, interjection)

    for node in traced.graph.nodes:
        if node.target not in modules:
            continue

        if type(modules[node.target]) == target_module_class:
            insert_interjection(traced, node, interjection, interjection_name)

    traced.recompile()
    return traced


class MyForwardInterjection(ForwardInterjection):
    def process(self, name, *args, **kwargs):
        print(name)
        # return args[0]

class MyWrappedForwardInterjection(WrappedForwardInterjection):
    def process(self, name, *args, **kwargs):
        print(name)


net = resnet18()
net2 = interject_by_module_class(net, nn.Conv2d, MyForwardInterjection())
# print(net2.code)
r = net2(torch.zeros(1, 3, 224, 244))

net = resnet18()
net2 = interject_by_module_class(net, nn.Conv2d, MyWrappedForwardInterjection())
# print(net2.code)
net2(torch.zeros(1, 3, 224, 244))

