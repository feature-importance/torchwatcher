import torch
import torch.nn as nn


class GradientIdentity(nn.Identity):
    """
    Subclass of nn.Identity that enables gradient tracking of the input. Useful
    when you want to interject a network's input gradients as you can just wrap
    with nn.Sequential(GradientIdentity(), net) and add the interjection to the
    GradientIdentity.
    """
    # noinspection PyMethodMayBeStatic
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.requires_grad = True
        return x