import torch
from model_utilities.models.cifar_resnet import resnet18_3x3, ResNet18_3x3_Weights
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from torchwatcher.analysis.analysis import Analyzer
from torchwatcher.interjection import interject_by_match, node_selector
from torchwatcher.nn import GradientIdentity

# Load a pre-trained model and set up some data
model = resnet18_3x3(weights=ResNet18_3x3_Weights.CIFAR10_s0)

data = CIFAR10(root="/Users/jsh2/data", train=False,
               transform=ResNet18_3x3_Weights.CIFAR10_s0.transforms())
loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=0)


# Create a class to record feature maps
class FeatureMapTracker(Analyzer):
    def __init__(self):
        super().__init__(gradient=False)

    def process_batch_state(self, name, state, working_results):
        if working_results is None:
            working_results = []
        fm = state.outputs.detach() # capture layer output (could swap to input)
        working_results.append(fm)
        return working_results

    def finalise_result(self, name, result):
        return torch.cat(result, dim=0)


fmt = FeatureMapTracker()

# interject the model. We'll use node_selector.Activations.is_relu to grab
# all featuremaps at every relu, but you can just swap this with a
# different selector or combination of selectors
model2 = interject_by_match(nn.Sequential(GradientIdentity(), model),
                            node_selector.Activations.is_relu,
                            fmt)
fmt.reset()

# we then just need to perform forward passes against the network outputs
model2.eval()
for _X, _y in loader:
    _pred = model2(_X)
    break  # comment this to get more than just the first batch

# fmt.to_dict()[...] contains the gradient maps for each layer
print(fmt.to_dict().keys())
print(fmt.to_dict()['1.relu'].shape)
