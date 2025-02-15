import abc
import copy
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Analyzer(abc.ABC):
    def __call__(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str) -> None:
        self.process_batch(features, classes, layer, name)

    @abc.abstractmethod
    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str) -> None:
        pass

    @abc.abstractmethod
    def get_result(self) -> dict:
        pass


class NameAnalyser(Analyser):
    """
    Just logs the layer name
    """

    def __init__(self):
        super().__init__()
        self.name = None

    def process_batch(self, features: torch.Tensor, classes: torch.Tensor, layer: nn.Module, name: str) -> None:
        if self.name is None:
            self.name = name

    def get_result(self) -> dict:
        return {'name': self.name}
