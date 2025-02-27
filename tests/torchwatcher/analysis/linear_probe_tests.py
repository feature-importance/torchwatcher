import unittest

import torch
from torch import nn

from torchwatcher.analysis.linear_probe import LinearProbe
from torchwatcher.interjection import interject_by_match
from torchwatcher.interjection.node_selector import node_types


class TestLinearProbes(unittest.TestCase):
    def setUp(self):
        self.net = nn.Sequential(nn.Linear(1, 2),
                                 nn.ReLU(),
                                 nn.Linear(2, 2),
                                 nn.ReLU(),
                                 nn.Linear(2, 1))
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam

    def test_linear_probes_untrained(self):
        lp = LinearProbe(1,
                         partial_optim=self.optimizer,
                         criterion=self.loss)

        net = interject_by_match(self.net, node_types.Activations.is_relu, lp)

        lp.targets = torch.ones(10, 1)
        net(torch.randn(10, 1))
        print(lp.to_dict())

        lp.train()
        for i in range(10):
            net(torch.randn(10, 1))
            lp.train_step()
            print(lp.probes["1_probe"].weight)
        lp.eval()
        net(torch.randn(10, 1))
        print(lp.to_dict())


if __name__ == '__main__':
    unittest.main()
