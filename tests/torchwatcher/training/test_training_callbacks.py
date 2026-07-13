import torch
import torchbearer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchwatcher.training import AnalyzerEvaluation


class FakeAnalyzer:
    def __init__(self, result=None):
        self.enabled = False
        self.reset_count = 0
        self.result = result or {"layer": {"value": 1}}

    def reset(self):
        self.reset_count += 1

    def to_dict(self):
        return self.result


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.calls = []

    def forward(self, x):
        self.calls.append((self.training, torch.is_grad_enabled(), x.device))
        return self.linear(x)


def test_analyzer_evaluation_records_and_restores_state():
    model = RecordingModel()
    model.train()
    analyzer = FakeAnalyzer()
    loader = DataLoader(
        TensorDataset(torch.randn(2, 2), torch.ones(2)),
        batch_size=1,
    )
    evaluation = AnalyzerEvaluation(analyzer, loader)
    state = {
        torchbearer.MODEL: model,
        torchbearer.EPOCH: 0,
        torchbearer.BATCH: 1,
    }

    record = evaluation.record(state, event="batch", global_step=2)

    assert model.training
    assert not analyzer.enabled
    assert analyzer.reset_count == 1
    assert model.calls == [
        (False, False, model.linear.weight.device),
        (False, False, model.linear.weight.device),
    ]
    assert record == {
        "event": "batch",
        "epoch": 1,
        "batch": 2,
        "global_step": 2,
        "result": {"layer": {"value": 1}},
    }


def test_analyzer_evaluation_composes_with_model_utilities_schedule():
    model = RecordingModel()
    train_loader = DataLoader(
        TensorDataset(torch.randn(4, 2), torch.zeros(4, 1)),
        batch_size=2,
    )
    eval_loader = DataLoader(
        TensorDataset(torch.randn(1, 2), torch.tensor([0])),
        batch_size=1,
    )
    evaluation = AnalyzerEvaluation(FakeAnalyzer(), eval_loader)
    trial = torchbearer.Trial(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        nn.BCEWithLogitsLoss(),
        metrics=["loss"],
        callbacks=evaluation.callbacks(
            every_n_batches=1,
            include_start=True,
            include_end=True,
        ),
    )

    trial.with_generators(train_generator=train_loader).to("cpu")
    trial.run(1, verbose=0)

    assert [record["event"] for record in evaluation.records] == [
        "start",
        "batch",
        "batch",
    ]
    assert [record["global_step"] for record in evaluation.records] == [0, 1, 2]


def test_analyzer_evaluation_accepts_arbitrary_training_iteration_schedule():
    model = RecordingModel()
    train_loader = DataLoader(
        TensorDataset(torch.randn(6, 2), torch.zeros(6, 1)),
        batch_size=2,
    )
    eval_loader = DataLoader(
        TensorDataset(torch.randn(1, 2), torch.tensor([0])),
        batch_size=1,
    )
    evaluation = AnalyzerEvaluation(FakeAnalyzer(), eval_loader)
    trial = torchbearer.Trial(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        nn.BCEWithLogitsLoss(),
        metrics=["loss"],
        callbacks=evaluation.callbacks(
            schedule=[0, 2],
            include_start=False,
            include_end=False,
        ),
    )

    trial.with_generators(train_generator=train_loader).to("cpu")
    trial.run(1, verbose=0)

    assert [record["event"] for record in evaluation.records] == [
        "batch",
        "batch",
    ]
    assert [record["global_step"] for record in evaluation.records] == [1, 3]
