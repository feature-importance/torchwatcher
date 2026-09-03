import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from torchwatcher.analysis import Analyser, NameAnalyser
from torchwatcher.interjection import interject_by_name


class TestAnalyserRun(unittest.TestCase):
    def test_runs_a_regular_analyser_over_a_dataset(self):
        analyser = NameAnalyser()
        watched = interject_by_name(
            nn.Sequential(nn.Linear(3, 4), nn.ReLU()),
            "1",
            analyser,
        )
        watched.train()
        analyser.enabled = False

        result = analyser.run(
            watched,
            TensorDataset(torch.randn(10, 3)),
            batch_size=4,
        )

        self.assertEqual(result, {"1": "1"})
        self.assertTrue(watched.training)
        self.assertFalse(analyser.enabled)

    def test_sets_targets_from_the_second_batch_item(self):
        class TargetSumAnalyser(Analyser):
            def process_batch_state(self, name, state, working_results):
                total = 0 if working_results is None else working_results
                return total + state.targets.sum().item()

        analyser = TargetSumAnalyser()
        watched = interject_by_name(
            nn.Sequential(nn.Linear(3, 4), nn.ReLU()),
            "1",
            analyser,
        )
        targets = torch.arange(8)

        result = analyser.run(
            watched,
            TensorDataset(torch.randn(8, 3), targets),
            batch_size=4,
        )

        self.assertEqual(result, {"1": targets.sum().item()})

    def test_accepts_custom_input_preparation(self):
        analyser = NameAnalyser()
        watched = interject_by_name(
            nn.Sequential(nn.Linear(3, 4), nn.ReLU()),
            "1",
            analyser,
        )
        dataset = [
            {"features": torch.randn(3)}
            for _ in range(8)
        ]

        result = analyser.run(
            watched,
            dataset,
            batch_size=4,
            prepare_inputs=lambda batch, device: batch["features"].to(device),
        )

        self.assertEqual(result, {"1": "1"})


if __name__ == "__main__":
    unittest.main()
