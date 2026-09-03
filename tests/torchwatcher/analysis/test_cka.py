import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchwatcher.analysis import LinearCKAAnalyser
from torchwatcher.interjection import interject_by_match, interject_by_name
from torchwatcher.interjection.node_selector import node_types


def _linear_cka(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_gram = left.flatten(1) @ left.flatten(1).t()
    right_gram = right.flatten(1) @ right.flatten(1).t()

    def center(gram):
        return (
            gram
            - gram.mean(dim=0, keepdim=True)
            - gram.mean(dim=1, keepdim=True)
            + gram.mean()
        )

    left_gram = center(left_gram)
    right_gram = center(right_gram)
    return (left_gram * right_gram).sum() / torch.sqrt(
        left_gram.square().sum() * right_gram.square().sum()
    )


class TestLinearCKAAnalyser(unittest.TestCase):
    def test_compares_layers_in_the_same_model(self):
        torch.manual_seed(1)
        model = nn.Sequential(
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU(),
        )
        analyser = LinearCKAAnalyser()
        watched = interject_by_match(
            model,
            node_types.Activations.is_relu,
            analyser.watch("model"),
        )

        for _ in range(2):
            with analyser.batch():
                watched(torch.randn(8, 4))

        result = analyser.result()
        self.assertEqual(result.row_names, ("model.1", "model.3"))
        self.assertEqual(result.column_names, result.row_names)
        torch.testing.assert_close(result.values, result.values.t())
        torch.testing.assert_close(
            result.values.diag(),
            torch.ones(2, dtype=result.values.dtype),
        )

    def test_compares_models_with_overlapping_layer_names(self):
        torch.manual_seed(2)
        model_a = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
        model_b = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
        inputs = torch.randn(10, 4)

        expected_a = model_a(inputs)
        expected_b = model_b(inputs)
        activation_a = model_a[1](model_a[0](inputs))
        activation_b = model_b[1](model_b[0](inputs))

        analyser = LinearCKAAnalyser()
        watched_a = interject_by_name(model_a, "1", analyser.watch("a"))
        watched_b = interject_by_name(model_b, "1", analyser.watch("b"))

        with analyser.batch():
            actual_b = watched_b(inputs)
            actual_a = watched_a(inputs)

        torch.testing.assert_close(actual_a, expected_a)
        torch.testing.assert_close(actual_b, expected_b)
        result = analyser.result("a", "b")
        self.assertEqual(result.row_names, ("a.1",))
        self.assertEqual(result.column_names, ("b.1",))
        torch.testing.assert_close(
            result.values[0, 0],
            _linear_cka(activation_a, activation_b).double(),
        )

    def test_accumulates_batches_and_resets(self):
        analyser = LinearCKAAnalyser()
        model = interject_by_name(
            nn.Sequential(nn.Linear(3, 3), nn.ReLU()),
            "1",
            analyser.watch("model"),
        )

        with analyser.batch():
            model(torch.randn(6, 3))
        self.assertTrue(analyser.to_dict())

        analyser.reset()
        self.assertEqual(analyser.to_dict(), {})
        with self.assertRaisesRegex(RuntimeError, "no complete CKA batches"):
            analyser.result()

    def test_requires_an_explicit_batch_boundary(self):
        analyser = LinearCKAAnalyser()
        model = interject_by_name(
            nn.Sequential(nn.Linear(3, 3), nn.ReLU()),
            "1",
            analyser.watch("model"),
        )

        with self.assertRaisesRegex(RuntimeError, "analyser.batch"):
            model(torch.randn(6, 3))

        analyser.enabled = False
        model(torch.randn(6, 3))

    def test_rejects_missing_and_mismatched_sources(self):
        analyser = LinearCKAAnalyser()
        watched_a = interject_by_name(
            nn.Sequential(nn.Linear(3, 3), nn.ReLU()),
            "1",
            analyser.watch("a"),
        )
        watched_b = interject_by_name(
            nn.Sequential(nn.Linear(3, 3), nn.ReLU()),
            "1",
            analyser.watch("b"),
        )

        with self.assertRaisesRegex(RuntimeError, "produced no observations"):
            with analyser.batch():
                watched_a(torch.randn(6, 3))

        with self.assertRaisesRegex(ValueError, "different batch sizes"):
            with analyser.batch():
                watched_a(torch.randn(6, 3))
                watched_b(torch.randn(5, 3))

        self.assertEqual(analyser.to_dict(), {})

    def test_debiased_estimator_requires_four_samples(self):
        analyser = LinearCKAAnalyser(debiased=True)
        model = interject_by_name(
            nn.Sequential(nn.Linear(3, 3), nn.ReLU()),
            "1",
            analyser.watch("model"),
        )

        with self.assertRaisesRegex(ValueError, "at least four"):
            with analyser.batch():
                model(torch.randn(3, 3))

    def test_run_handles_a_within_model_dataset(self):
        analyser = LinearCKAAnalyser()
        watched = interject_by_match(
            nn.Sequential(
                nn.Linear(3, 4),
                nn.ReLU(),
                nn.Linear(4, 2),
                nn.ReLU(),
            ),
            node_types.Activations.is_relu,
            analyser.watch("model"),
        )
        watched.train()
        analyser.enabled = False

        output = analyser.run(
            watched,
            TensorDataset(torch.randn(12, 3)),
            batch_size=6,
        )

        self.assertIn("model:model", output)
        self.assertEqual(analyser.result().values.shape, (2, 2))
        self.assertTrue(watched.training)
        self.assertFalse(analyser.enabled)

    def test_run_handles_cross_model_dataloader(self):
        analyser = LinearCKAAnalyser()
        watched_a = interject_by_name(
            nn.Sequential(nn.Linear(3, 4), nn.ReLU()),
            "1",
            analyser.watch("a"),
        )
        watched_b = interject_by_name(
            nn.Sequential(nn.Linear(3, 5), nn.ReLU()),
            "1",
            analyser.watch("b"),
        )
        loader = DataLoader(
            TensorDataset(torch.randn(12, 3)),
            batch_size=6,
            shuffle=False,
        )

        analyser.run({"a": watched_a, "b": watched_b}, loader)

        self.assertEqual(analyser.result("a", "b").values.shape, (1, 1))
        with self.assertRaisesRegex(ValueError, "model sources"):
            analyser.run({"a": watched_a}, loader)


if __name__ == "__main__":
    unittest.main()
