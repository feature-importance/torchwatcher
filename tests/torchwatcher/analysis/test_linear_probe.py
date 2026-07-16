import unittest

import torch
import torchbearer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
        self.optimiser = torch.optim.Adam

    def test_linear_probes_untrained(self):
        lp = LinearProbe(1,
                         partial_optim=self.optimiser,
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

    def test_callback_trains_from_torchbearer_targets(self):
        net = nn.Sequential(nn.Linear(2, 4),
                            nn.ReLU(),
                            nn.Linear(4, 2))
        lp = LinearProbe(2,
                         partial_optim=lambda params: torch.optim.SGD(params, lr=0.1),
                         criterion=nn.CrossEntropyLoss())
        net = interject_by_match(net, node_types.Activations.is_relu, lp)
        loader = DataLoader(
            TensorDataset(torch.randn(8, 2), torch.randint(0, 2, (8,))),
            batch_size=4,
        )

        trial = torchbearer.Trial(
            net,
            metrics=['acc'],
            callbacks=[lp.callback(keep_model_eval=True)],
        ).with_generators(train_generator=loader)
        trial.run(1, verbose=0)

        self.assertTrue(lp._targets_set)
        self.assertEqual(lp.targets.shape, (4,))
        self.assertIn("1", lp.working_results)
        self.assertFalse(net.training)
        self.assertTrue(lp.training)

    def test_callback_accumulates_eval_metrics(self):
        net = nn.Sequential(nn.Linear(2, 4),
                            nn.ReLU(),
                            nn.Linear(4, 2))
        lp = LinearProbe(2,
                         partial_optim=lambda params: torch.optim.SGD(params, lr=0.1),
                         criterion=nn.CrossEntropyLoss())
        net = interject_by_match(net, node_types.Activations.is_relu, lp)
        loader = DataLoader(
            TensorDataset(torch.randn(8, 2), torch.randint(0, 2, (8,))),
            batch_size=4,
        )

        trial = torchbearer.Trial(
            net,
            metrics=['acc'],
            callbacks=[lp.callback(train_probes=False)],
        ).with_generators(val_generator=loader)
        trial.evaluate(verbose=0)

        results = lp.to_dict()
        self.assertIn("1", results)
        self.assertIn("acc", results["1"])


if __name__ == '__main__':
    unittest.main()
