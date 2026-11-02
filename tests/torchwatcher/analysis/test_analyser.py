import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from torchwatcher.analysis import (
    Analyser,
    AnalyserList,
    NameAnalyser,
    PerClassAnalyser,
    PerClassVersusAnalyser,
)
from torchwatcher.interjection import interject_by_name


class _OutputGradientAnalyser(Analyser):
    def __init__(self):
        super().__init__(gradient=True)

    def process_batch_state(self, name, state, working_results):
        return state.output_gradients.detach().clone()


class _BatchSizeAnalyser(Analyser):
    def process_batch_state(self, name, state, working_results):
        count = 0 if working_results is None else working_results
        return count + state.outputs.shape[0]


class _BufferedAnalyser(_BatchSizeAnalyser):
    def __init__(self):
        super().__init__()
        self.register_buffer("marker", torch.ones(1))


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

    def test_rejects_analyser_list_containing_gradient_analyser(self):
        analyser = AnalyserList(NameAnalyser(), _OutputGradientAnalyser())
        watched = interject_by_name(
            nn.Sequential(nn.Linear(3, 4), nn.ReLU()),
            "1",
            analyser,
        )

        self.assertTrue(analyser.gradient)
        with self.assertRaisesRegex(ValueError, "explicit backward pass"):
            analyser.run(
                watched,
                TensorDataset(torch.randn(4, 3)),
                batch_size=2,
            )

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


class TestAnalyserList(unittest.TestCase):
    def test_mixed_list_processes_gradient_child_once(self):
        name_analyser = NameAnalyser()
        gradient_analyser = _OutputGradientAnalyser()
        analyser = AnalyserList(name_analyser, gradient_analyser)
        watched = interject_by_name(
            nn.Sequential(nn.Linear(2, 2), nn.ReLU()),
            "1",
            analyser,
        )
        inputs = torch.ones(1, 2, requires_grad=True)

        watched(inputs).sum().backward()

        self.assertEqual(name_analyser.to_dict(), {"1": "1"})
        torch.testing.assert_close(
            gradient_analyser.to_dict()["1"],
            torch.ones(1, 2),
        )
        self.assertEqual(gradient_analyser.current_states, {})


class TestAnalyserEnablement(unittest.TestCase):
    def test_disabled_forward_analyser_does_not_retain_activations(self):
        analyser = NameAnalyser()
        watched = interject_by_name(
            nn.Sequential(nn.Identity()),
            "0",
            analyser,
        )
        analyser.enabled = False

        watched(torch.ones(2, 3, requires_grad=True))

        self.assertEqual(analyser.current_states, {})
        self.assertEqual(analyser.to_dict(), {})

    def test_disabled_gradient_analyser_does_not_retain_autograd_state(self):
        analyser = _OutputGradientAnalyser()
        watched = interject_by_name(
            nn.Sequential(nn.Identity()),
            "0",
            analyser,
        )
        analyser.enabled = False
        inputs = torch.ones(2, 3, requires_grad=True)

        watched(inputs).sum().backward()

        self.assertEqual(analyser.current_states, {})
        self.assertEqual(analyser.to_dict(), {})

    def test_enablement_change_between_forward_and_backward_is_safe(self):
        analyser = _OutputGradientAnalyser()
        watched = interject_by_name(
            nn.Sequential(nn.Identity()),
            "0",
            analyser,
        )
        inputs = torch.ones(2, 3, requires_grad=True)

        outputs = watched(inputs)
        self.assertIn("0", analyser.current_states)
        analyser.enabled = False
        self.assertEqual(analyser.current_states, {})
        analyser.enabled = True
        outputs.sum().backward()

        self.assertEqual(analyser.current_states, {})
        self.assertEqual(analyser.to_dict(), {})


class TestPerClassAnalyser(unittest.TestCase):
    def test_children_follow_module_lifecycle(self):
        analyser = PerClassAnalyser(_BufferedAnalyser())
        watched = interject_by_name(
            nn.Sequential(nn.Identity()),
            "0",
            analyser,
        )
        analyser.targets = torch.tensor([0])

        watched(torch.ones(1, 1))
        child = analyser.analysers[0]
        analyser.eval()
        analyser.to(dtype=torch.float64)

        self.assertFalse(child.training)
        self.assertEqual(child.marker.dtype, torch.float64)
        self.assertIn("_class_analysers.0.marker", analyser.state_dict())

        analyser.reset()

        self.assertEqual(len(analyser._class_analysers), 0)
        self.assertNotIn("_class_analysers.0.marker", analyser.state_dict())

    def test_reset_discards_child_results(self):
        analyser = PerClassAnalyser(_BatchSizeAnalyser())
        watched = interject_by_name(
            nn.Sequential(nn.Identity()),
            "0",
            analyser,
        )
        analyser.targets = torch.tensor([0, 1])

        watched(torch.ones(2, 1))
        self.assertEqual(analyser.to_dict(), {0: {"0": 1}, 1: {"0": 1}})

        analyser.reset()

        self.assertEqual(analyser.to_dict(), {})

    def test_versus_reuses_scalar_class_keys_across_batches(self):
        analyser = PerClassVersusAnalyser(_BatchSizeAnalyser())
        watched = interject_by_name(
            nn.Sequential(nn.Identity()),
            "0",
            analyser,
        )

        for _ in range(2):
            analyser.targets = torch.tensor([0, 1, 0])
            watched(torch.ones(3, 1))

        self.assertEqual(
            analyser.to_dict(),
            {
                0: {"0": 4},
                "~0": {"0": 2},
                1: {"0": 2},
                "~1": {"0": 4},
            },
        )

    def test_versus_uses_class_complement_for_backward(self):
        analyser = PerClassVersusAnalyser(_OutputGradientAnalyser())
        watched = interject_by_name(
            nn.Sequential(nn.Identity()),
            "0",
            analyser,
        )
        analyser.targets = torch.tensor([0, 1, 0])
        inputs = torch.ones(3, 1, requires_grad=True)
        weights = torch.tensor([[1.0], [2.0], [3.0]])

        (watched(inputs) * weights).sum().backward()

        result = analyser.to_dict()
        torch.testing.assert_close(result[0]["0"], weights[[0, 2]])
        torch.testing.assert_close(result["~0"]["0"], weights[[1]])
        torch.testing.assert_close(result[1]["0"], weights[[1]])
        torch.testing.assert_close(result["~1"]["0"], weights[[0, 2]])


if __name__ == "__main__":
    unittest.main()
