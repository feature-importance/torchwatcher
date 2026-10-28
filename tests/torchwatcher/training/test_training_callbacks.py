import torch
import torchbearer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchwatcher.parameters import selectors
from torchwatcher.training import (
    AnalyserEvaluation,
    SelectedParametersCallback,
)


class FakeAnalyser:
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


def test_selected_parameters_callback_resolves_selector_at_start():
    model = RecordingModel()
    calls = []
    callback = SelectedParametersCallback(
        selectors.matches_full_name("linear.weight"),
        lambda parameter_info, state: calls.append((parameter_info, state)),
    )
    state = {torchbearer.MODEL: model}

    callback.on_start(state)
    callback.on_backward(state)

    assert len(calls) == 1
    assert calls[0][0] is callback.selected[0]
    assert calls[0][1] is state
    assert len(callback.selected) == 1
    assert callback.selected[0].full_name == "linear.weight"
    assert callback.selected[0].parameter is model.linear.weight


def test_selected_parameters_callback_resolves_explicit_parameters():
    model = RecordingModel()
    calls = []
    callback = SelectedParametersCallback(
        [model.linear.bias, model.linear.weight, model.linear.bias],
        lambda parameter_info, _: calls.append(parameter_info),
    )
    state = {torchbearer.MODEL: model}

    callback.on_start(state)
    callback.on_backward(state)

    assert [info.parameter for info in calls] == [
        model.linear.bias,
        model.linear.weight,
    ]


def test_selected_parameters_callback_supports_explicit_events():
    model = RecordingModel()
    state = {torchbearer.MODEL: model}

    for event in ("on_start", "on_backward", "on_step_training"):
        calls = []
        callback = SelectedParametersCallback(
            selectors.matches_full_name("linear.weight"),
            lambda _, __: calls.append(event),
            event=event,
        )

        callback.on_start(state)
        callback.on_backward(state)
        callback.on_step_training(state)

        assert calls == [event]


def test_selected_parameters_callback_runs_before_optimizer_step():
    model = nn.Linear(1, 1, bias=False)
    original_weight = model.weight.detach().clone()
    loader = DataLoader(
        TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)),
        batch_size=1,
    )

    def clear_gradient(parameter_info, _):
        parameter_info.parameter.grad.zero_()

    callback = SelectedParametersCallback(
        selectors.matches_full_name("weight"),
        clear_gradient,
    )
    trial = torchbearer.Trial(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        nn.MSELoss(),
        callbacks=[callback],
    )

    trial.with_generators(train_generator=loader).to("cpu")
    trial.run(1, verbose=0)

    assert torch.equal(model.weight, original_weight)


def test_selected_parameters_callback_rejects_empty_selection():
    callback = SelectedParametersCallback(
        selectors.matches_full_name("missing"),
        lambda *_: None,
    )

    try:
        callback.on_start({torchbearer.MODEL: RecordingModel()})
    except ValueError as exc:
        assert "matched no parameters" in str(exc)
    else:
        raise AssertionError("callback accepted an empty parameter selection")


def test_selected_parameters_callback_rejects_unregistered_parameter():
    callback = SelectedParametersCallback(
        [nn.Parameter(torch.ones(1))],
        lambda *_: None,
    )

    try:
        callback.on_start({torchbearer.MODEL: RecordingModel()})
    except ValueError as exc:
        assert "not registered" in str(exc)
    else:
        raise AssertionError("callback accepted an unregistered parameter")


def test_selected_parameters_callback_validates_constructor_arguments():
    try:
        SelectedParametersCallback([], None)
    except TypeError as exc:
        assert "action" in str(exc)
    else:
        raise AssertionError("callback accepted a non-callable action")

    try:
        SelectedParametersCallback([], lambda *_: None, event="unknown")
    except ValueError as exc:
        assert "event" in str(exc)
    else:
        raise AssertionError("callback accepted an unknown event")

    try:
        SelectedParametersCallback([torch.ones(1)], lambda *_: None)
    except TypeError as exc:
        assert "Parameter" in str(exc)
    else:
        raise AssertionError("callback accepted a non-parameter tensor")


def test_analyser_evaluation_records_and_restores_state():
    model = RecordingModel()
    model.train()
    analyser = FakeAnalyser()
    loader = DataLoader(
        TensorDataset(torch.randn(2, 2), torch.ones(2)),
        batch_size=1,
    )
    evaluation = AnalyserEvaluation(analyser, loader)
    state = {
        torchbearer.MODEL: model,
        torchbearer.EPOCH: 0,
        torchbearer.BATCH: 1,
    }

    record = evaluation.record(state, event="batch", global_step=2)

    assert model.training
    assert not analyser.enabled
    assert analyser.reset_count == 1
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


def test_analyser_evaluation_can_enable_gradient_computation():
    model = RecordingModel()
    analyser = FakeAnalyser()
    loader = DataLoader(
        TensorDataset(torch.randn(2, 2), torch.ones(2)),
        batch_size=1,
    )
    evaluation = AnalyserEvaluation(
        analyser,
        loader,
        compute_gradients=True,
    )
    state = {torchbearer.MODEL: model}

    evaluation.record(state, event="batch", global_step=1)

    assert model.calls == [
        (False, True, model.linear.weight.device),
        (False, True, model.linear.weight.device),
    ]
    assert model.linear.weight.grad is None


def test_analyser_evaluation_runs_optional_backward_callback():
    model = RecordingModel()
    original_weight_grad = torch.ones_like(model.linear.weight)
    model.linear.weight.grad = original_weight_grad.clone()
    analyser = FakeAnalyser()
    loader = DataLoader(
        TensorDataset(torch.randn(2, 2), torch.ones(2)),
        batch_size=1,
    )
    backward_calls = []

    def backward(outputs, batch):
        backward_calls.append(batch)
        return outputs.sum()

    evaluation = AnalyserEvaluation(
        analyser,
        loader,
        compute_gradients=True,
        backward=backward,
    )
    state = {torchbearer.MODEL: model}

    evaluation.record(state, event="batch", global_step=1)

    assert len(backward_calls) == 2
    assert torch.equal(model.linear.weight.grad, original_weight_grad)


def test_analyser_evaluation_rejects_backward_without_gradients():
    loader = DataLoader(
        TensorDataset(torch.randn(1, 2), torch.ones(1)),
        batch_size=1,
    )

    try:
        AnalyserEvaluation(
            FakeAnalyser(),
            loader,
            backward=lambda outputs, batch: outputs.sum(),
        )
    except ValueError as exc:
        assert "compute_gradients" in str(exc)
    else:
        raise AssertionError("AnalyserEvaluation accepted backward without gradients")


def test_analyser_evaluation_composes_with_model_utilities_schedule():
    model = RecordingModel()
    train_loader = DataLoader(
        TensorDataset(torch.randn(4, 2), torch.zeros(4, 1)),
        batch_size=2,
    )
    eval_loader = DataLoader(
        TensorDataset(torch.randn(1, 2), torch.tensor([0])),
        batch_size=1,
    )
    evaluation = AnalyserEvaluation(FakeAnalyser(), eval_loader)
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


def test_analyser_evaluation_accepts_arbitrary_training_iteration_schedule():
    model = RecordingModel()
    train_loader = DataLoader(
        TensorDataset(torch.randn(6, 2), torch.zeros(6, 1)),
        batch_size=2,
    )
    eval_loader = DataLoader(
        TensorDataset(torch.randn(1, 2), torch.tensor([0])),
        batch_size=1,
    )
    evaluation = AnalyserEvaluation(FakeAnalyser(), eval_loader)
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
