import copy
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
from torchwatcher.interjection import (
    interject_by_match,
    interject_by_module_class_native,
    interject_by_name,
)
from torchwatcher.interjection.node_selector import NodeSelector


class _OutputGradientAnalyser(Analyser):
    def __init__(self):
        super().__init__(gradient=True)

    def process_batch_state(self, name, state, working_results):
        return state.output_gradients.detach().clone()


class _BatchSizeAnalyser(Analyser):
    def process_batch_state(self, name, state, working_results):
        count = 0 if working_results is None else working_results
        return count + state.outputs.shape[0]


class _GradientCallCountAnalyser(Analyser):
    def __init__(self):
        super().__init__(gradient=True)

    def process_batch_state(self, name, state, working_results):
        count = 0 if working_results is None else working_results
        return count + 1


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

    def test_reset_run_does_not_reuse_targets_from_a_previous_run(self):
        analyser = PerClassAnalyser(NameAnalyser())
        watched = interject_by_name(
            nn.Sequential(nn.Linear(3, 4), nn.ReLU()),
            "1",
            analyser,
        )
        labelled = TensorDataset(
            torch.randn(4, 3),
            torch.tensor([0, 0, 1, 1]),
        )
        unlabelled = TensorDataset(torch.randn(2, 3))

        self.assertEqual(
            analyser.run(watched, labelled, batch_size=2),
            {0: {"1": "1"}, 1: {"1": "1"}},
        )

        self.assertEqual(
            analyser.run(watched, unlabelled, batch_size=2),
            {},
        )

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

    def test_duplicate_analyser_types_have_distinct_result_prefixes(self):
        first = NameAnalyser()
        second = NameAnalyser()
        first.working_results["layer"] = "first"
        second.working_results["layer"] = "second"

        result = AnalyserList(first, second).to_dict()

        self.assertEqual(
            result,
            {
                "NameAnalyser[0].layer": "first",
                "NameAnalyser[1].layer": "second",
            },
        )


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


class TestGradientAnalyserState(unittest.TestCase):
    def test_records_repeated_native_module_calls(self):
        class ReusedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, inputs):
                return self.relu(inputs) + self.relu(inputs)

        analyser = _GradientCallCountAnalyser()
        watched = interject_by_module_class_native(
            ReusedModule(),
            nn.ReLU,
            analyser,
        )
        inputs = torch.ones(2, requires_grad=True)

        watched(inputs).sum().backward()

        self.assertEqual(analyser.to_dict(), {"relu": 2})
        self.assertEqual(analyser.current_states, {})

    def test_records_multiple_forwards_before_combined_backward(self):
        analyser = _GradientCallCountAnalyser()
        watched = interject_by_name(
            nn.Sequential(nn.Identity()),
            "0",
            analyser,
        )
        first = torch.ones(2, requires_grad=True)
        second = torch.ones(2, requires_grad=True)

        outputs = watched(first) + watched(second)
        outputs.sum().backward()

        self.assertEqual(analyser.to_dict(), {"0": 2})
        self.assertEqual(analyser.current_states, {})


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

    def test_state_dict_reconstructs_dynamic_children(self):
        analyser = PerClassAnalyser(_BufferedAnalyser())
        watched = interject_by_name(
            nn.Sequential(nn.Identity()),
            "0",
            analyser,
        )
        analyser.targets = torch.tensor([0, 1])
        watched(torch.ones(2, 1))
        analyser.analysers[0].marker.fill_(2)
        analyser.analysers[1].marker.fill_(3)
        state = copy.deepcopy(analyser.state_dict())

        restored = PerClassAnalyser(_BufferedAnalyser())
        interject_by_name(
            nn.Sequential(nn.Identity()),
            "0",
            restored,
        )
        restored.load_state_dict(state)

        self.assertEqual(list(restored.analysers), [0, 1])
        torch.testing.assert_close(
            restored.analysers[0].marker,
            torch.tensor([2.0]),
        )
        torch.testing.assert_close(
            restored.analysers[1].marker,
            torch.tensor([3.0]),
        )

    def test_supports_multi_input_wrapped_nodes(self):
        class AddInputs(nn.Module):
            def forward(self, left, right):
                return torch.add(left, right)

        selector = NodeSelector(
            lambda state: (
                state[1].op == "call_function"
                and state[1].target is torch.add
            )
        )
        analyser = PerClassAnalyser(_BatchSizeAnalyser())
        watched = interject_by_match(AddInputs(), selector, analyser)
        analyser.targets = torch.tensor([0, 1])

        watched(torch.ones(2, 1), torch.ones(2, 1))

        self.assertEqual(analyser.to_dict(), {0: {"add": 1}, 1: {"add": 1}})

    def test_supports_multi_input_gradients(self):
        class AddInputs(nn.Module):
            def forward(self, left, right):
                return torch.add(left, right)

        selector = NodeSelector(
            lambda state: (
                state[1].op == "call_function"
                and state[1].target is torch.add
            )
        )
        analyser = PerClassAnalyser(_OutputGradientAnalyser())
        watched = interject_by_match(AddInputs(), selector, analyser)
        analyser.targets = torch.tensor([0, 1])
        left = torch.ones(2, 1, requires_grad=True)
        right = torch.ones(2, 1, requires_grad=True)
        weights = torch.tensor([[1.0], [2.0]])

        (watched(left, right) * weights).sum().backward()

        torch.testing.assert_close(
            analyser.to_dict()[0]["add"],
            weights[[0]],
        )
        torch.testing.assert_close(
            analyser.to_dict()[1]["add"],
            weights[[1]],
        )

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
