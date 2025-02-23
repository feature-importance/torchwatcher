import torch
from torchvision.models import resnet18
from torchvision.models.feature_extraction import get_graph_node_names

from torchwatcher.analysis.analysis import PerClassAnalyzer
from torchwatcher.analysis.basic_statistics import FeatureStats
from torchwatcher.analysis.dead_relus import DeadReLU
from torchwatcher.interjection import interject_by_match, ForwardInterjection, \
    WrappedForwardBackwardInterjection
from torchwatcher.interjection.node_selector import node_types
import torchwatcher.interjection.node_selector as node_selector


class MyForwardInterjection(ForwardInterjection):
    def process(self, name, input):
        print(name, input.shape)
        # return args[0]


# net = resnet18()
# net2 = interject_by_module_class(net, nn.Conv2d, MyForwardInterjection())
# r = net2(torch.zeros(1, 3, 224, 244))
#
#
# class MyWrappedForwardInterjection(WrappedForwardInterjection):
#     def process(self, name, input, output):
#         print(name, input.shape)
#
#
# net = resnet18()
# net2 = interject_by_module_class(net, nn.Conv2d,
# MyWrappedForwardInterjection())
# net2(torch.zeros(1, 3, 224, 244))
#
#
class MyWrappedForwardBackwardInterjection(WrappedForwardBackwardInterjection):
    def process(self, name, module, input, output):
        print("forward", name, input.shape)
        print(dict(module.named_parameters()))

    def process_backward(self, name, module, grad_input, grad_output):
        print("backward", name,
              grad_input.shape if grad_input is not None else None)
        print(dict(module.named_parameters()))


#
# net = resnet18()
# net2 = interject_by_module_class(net, nn.Conv2d,
# MyWrappedForwardBackwardInterjection())
# r = net2(torch.zeros(1, 3, 224, 244))
# loss = torch.nn.functional.cross_entropy(r, torch.tensor([0],
# dtype=torch.long))
# loss.backward()

# net = resnet18()
# print(get_graph_node_names(net))
# # net2 = interject_by_match(net, node_selector.is_activation,
# # MyWrappedForwardBackwardInterjection())
# net2 = interject_by_match(net, node_selector.matches_qualified_name(
#     "layer1.0.conv1"),
#                           MyWrappedForwardBackwardInterjection())
# r = net2(torch.zeros(1, 3, 224, 244))
# loss = torch.nn.functional.cross_entropy(r, torch.tensor([0], dtype=torch.long))
# loss.backward()
#
# dr = DeadReLU()
# net = resnet18()
# net2 = interject_by_match(net, node_types.Activations.is_relu, dr)
# net2(torch.rand(10, 3, 224, 224))
# print(dr.to_dict())


fs = PerClassAnalyzer(FeatureStats())
net = resnet18()
net2 = interject_by_match(net, node_types.Activations.is_relu, fs)
fs.targets = torch.randint(0, 3, (10,))
net2(torch.rand(10, 3, 224, 224))
for k, v in fs.to_dict().items():
    print(k, v['channel_sparsity_mean'])