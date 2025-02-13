import torch
from torch import nn
from torch.fx import symbolic_trace
from torchvision.models import resnet18

from torchwatcher.interjection import ForwardInterjection


def get_fresh_qualname(graph: torch.fx.Graph, prefix: str) -> str:
    i = 0
    while True:
        qualname = f"{prefix}{i}"
        i += 1
        if not hasattr(graph, qualname):
            break

    return qualname


def insert_interjection(traced, node, interjection):
    if hasattr(interjection, 'wrapped'):
        raise RuntimeError("Not yet implemented")
    else:
        with traced.graph.inserting_after(node):
            name = get_fresh_qualname(traced.graph, type(interjection).__name__)
            # TODO: how do we reuse the interjections so we just have one module instance and feed in the position in
            #  the graph?
            traced.add_module(name, interjection)
            new_node = traced.graph.call_module(name, (node,))
            node.replace_all_uses_with(new_node)
            new_node.replace_input_with(new_node, node)


def interject_by_module_class(network, target_module_class, interjection):
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
    def process(self, *args, **kwargs):
        print(args[0])
        # return args[0]


net = resnet18()
net2 = interject_by_module_class(net, nn.Conv2d, MyForwardInterjection())

r = net2(torch.zeros(1, 3, 224, 244))
