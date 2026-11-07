import unittest

import torch
import torchbearer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchwatcher.analysis import PerClassAnalyser
from torchwatcher.analysis.linear_probe import LinearProbe
from torchwatcher.interjection import interject_by_match, interject_by_name
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
                         partial_optimiser=self.optimiser,
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
                         partial_optimiser=lambda params: torch.optim.SGD(params, lr=0.1),
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
                         partial_optimiser=lambda params: torch.optim.SGD(params, lr=0.1),
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

    def test_accepts_non_contiguous_activations(self):
        lp = LinearProbe(
            2,
            partial_optimiser=lambda params: torch.optim.SGD(params, lr=0.1),
            criterion=nn.CrossEntropyLoss(),
        )
        net = interject_by_name(nn.Sequential(nn.Identity()), "0", lp)
        features = torch.randn(2, 2, 2, 2).to(
            memory_format=torch.channels_last
        )
        self.assertFalse(features.is_contiguous())
        lp.targets = torch.tensor([0, 1])

        net(features)

        self.assertEqual(lp.working_results["0"].shape, (2, 2))

    def test_per_class_wrapper_replays_probe_registration(self):
        analyser = PerClassAnalyser(LinearProbe(
            2,
            partial_optimiser=lambda params: torch.optim.SGD(params, lr=0.1),
            criterion=nn.CrossEntropyLoss(),
        ))
        net = interject_by_name(nn.Sequential(nn.Identity()), "0", analyser)
        analyser.targets = torch.tensor([0, 1])

        net(torch.randn(2, 3))

        for child in analyser.analysers.values():
            self.assertIn("0_probe", child.probes)
            self.assertEqual(child.working_results["0"].shape, (1, 2))
            self.assertEqual(child.interjection._wrapped, {})

    def test_qualified_name_collisions_create_distinct_probes(self):
        class Nested(nn.Module):
            def __init__(self):
                super().__init__()
                self.b = nn.Identity()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.a_b = nn.Identity()
                self.a = Nested()

            def forward(self, inputs):
                return self.a.b(self.a_b(inputs))

        probe = LinearProbe(
            2,
            partial_optimiser=lambda params: torch.optim.SGD(params, lr=0.1),
            criterion=nn.CrossEntropyLoss(),
        )
        watched_model = interject_by_match(
            Model(),
            node_types.Linear.is_identity,
            probe,
        )

        self.assertEqual(set(probe._probe_names), {"a_b", "a.b"})
        self.assertEqual(len(set(probe._probe_names.values())), 2)
        self.assertEqual(len(probe.probes), 2)
        for name, optimiser in probe.optimisers.items():
            registered_probe = probe.probes[probe._probe_names[name]]
            self.assertIs(
                optimiser.param_groups[0]["params"][0],
                registered_probe.weight,
            )

        probe.targets = torch.tensor([0, 1])
        watched_model(torch.randn(2, 3))
        self.assertEqual(set(probe.working_results), {"a_b", "a.b"})


if __name__ == '__main__':
    unittest.main()
