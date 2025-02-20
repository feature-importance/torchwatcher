import torch
from torch import nn

from torchwatcher.interjection import ForwardInterjection


class DeadReLU(ForwardInterjection):
    def __init__(self):
        super().__init__()
        self.masks = {}

    def process(self, name: str, module: [None | nn.Module], x: torch.Tensor):
        if name not in self.masks:
            self.masks[name] = torch.zeros(x.shape[1:])

        with torch.no_grad():
            self.masks[name] += (x.detach().sum(dim=0) > 0).float()

    def print_summary(self):
        for name, mask in self.masks.items():
            print(name, mask.numel() - mask.sum())


