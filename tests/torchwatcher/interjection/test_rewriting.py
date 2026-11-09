import torch
import pytest
from torch import nn

from torchwatcher.interjection import (
    replace,
    replace_module,
    replace_module_native,
)
from torchwatcher.interjection.node_selector import matches_module_class


class _RaisesInEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if not self.training:
            raise RuntimeError("eval tracing failed")
        return outputs


def test_parameterised_replacement_is_shared_between_train_and_eval():
    model = nn.Sequential(nn.Linear(1, 1, bias=False))
    replacements = []

    def create_replacement(original):
        replacement = nn.Linear(
            original.in_features,
            original.out_features,
            bias=False,
        )
        replacement.weight.data.fill_(1.0)
        replacements.append(replacement)
        return replacement

    replaced = replace(
        model,
        matches_module_class(nn.Linear),
        create_replacement,
    )
    train_replacement = replaced.train_module.get_submodule("0")
    eval_replacement = replaced.eval_module.get_submodule("0")

    assert replacements == [train_replacement]
    assert eval_replacement is train_replacement

    optimiser = torch.optim.SGD(replaced.parameters(), lr=0.1)
    replaced.train()
    replaced(torch.ones(1, 1)).sum().backward()
    optimiser.step()
    replaced.eval()

    torch.testing.assert_close(
        replaced(torch.ones(1, 1)),
        torch.tensor([[0.9]]),
    )
    torch.testing.assert_close(
        replaced.state_dict()["0.weight"],
        torch.tensor([[0.9]]),
    )


def test_parameterised_replacement_state_dict_hooks_run_once():
    calls = []

    def create_replacement(original):
        replacement = nn.Linear(
            original.in_features,
            original.out_features,
        )
        replacement.register_state_dict_pre_hook(
            lambda _module, prefix, _keep_vars: (
                calls.append(("pre", prefix))
            )
        )
        replacement.register_state_dict_post_hook(
            lambda _module, _state, prefix, _metadata: (
                calls.append(("post", prefix))
            )
        )
        return replacement

    replaced = replace(
        nn.Sequential(nn.Linear(1, 1)),
        matches_module_class(nn.Linear),
        create_replacement,
    )

    replaced.state_dict()

    assert calls == [("pre", "0."), ("post", "0.")]


def test_parameterised_replacement_hook_renamed_key_loads_once():
    load_prefixes = []

    def create_replacement(original):
        replacement = nn.Linear(
            original.in_features,
            original.out_features,
        )

        def save_hook(_module, state, prefix, _metadata):
            state[f"{prefix}renamed_weight"] = state.pop(
                f"{prefix}weight"
            )

        def load_hook(_module, state, prefix, *_args):
            load_prefixes.append(prefix)
            renamed_key = f"{prefix}renamed_weight"
            if renamed_key in state:
                state[f"{prefix}weight"] = state.pop(renamed_key)

        replacement.register_state_dict_post_hook(save_hook)
        replacement.register_load_state_dict_pre_hook(load_hook)
        return replacement

    replaced = replace(
        nn.Sequential(nn.Linear(1, 1)),
        matches_module_class(nn.Linear),
        create_replacement,
    )
    checkpoint = replaced.state_dict()

    replaced.load_state_dict(checkpoint, strict=True)

    assert load_prefixes == ["0."]


def test_parameterised_replacement_post_hook_sees_public_diagnostics():
    diagnostics = []

    def create_replacement(original):
        replacement = nn.Linear(
            original.in_features,
            original.out_features,
        )
        def post_hook(_module, incompatible):
            diagnostics.append((
                list(incompatible.missing_keys),
                list(incompatible.unexpected_keys),
            ))
            incompatible.missing_keys.remove("0.weight")

        replacement.register_load_state_dict_post_hook(post_hook)
        return replacement

    replaced = replace(
        nn.Sequential(nn.Linear(1, 1)),
        matches_module_class(nn.Linear),
        create_replacement,
    )

    incompatible = replaced.load_state_dict({}, strict=False)

    expected_missing = ["0.weight", "0.bias"]
    assert diagnostics == [(expected_missing, [])]
    assert incompatible.missing_keys == ["0.bias"]


def test_replace_restores_source_mode_after_failure():
    model = _RaisesInEval()
    model.train()
    model.linear.eval()

    with pytest.raises(RuntimeError, match="eval tracing failed"):
        replace(
            model,
            matches_module_class(nn.Linear),
            lambda original: nn.Linear(
                original.in_features,
                original.out_features,
            ),
        )

    assert model.training
    assert not model.linear.training


def test_replace_rejects_mixed_mode_module_traced_through_by_fx():
    class ModeDependentBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, inputs):
            if self.training:
                return self.linear(inputs)
            return -self.linear(inputs)

    model = nn.Sequential(ModeDependentBlock())
    model.train()
    model[0].eval()

    with pytest.raises(ValueError, match="mixed training modes.*'0'"):
        replace(
            model,
            matches_module_class(nn.Linear),
            lambda original: nn.Linear(
                original.in_features,
                original.out_features,
            ),
        )

    assert model.training
    assert not model[0].training


def test_replace_does_not_mutate_tracer_kwargs():
    leaf_modules = [_RaisesInEval]
    tracer_kwargs = {"leaf_modules": leaf_modules}

    replace(
        nn.Sequential(nn.Linear(1, 1)),
        matches_module_class(nn.Linear),
        lambda original: nn.Linear(
            original.in_features,
            original.out_features,
        ),
        tracer_kwargs=tracer_kwargs,
    )

    assert tracer_kwargs == {"leaf_modules": [_RaisesInEval]}
    assert tracer_kwargs["leaf_modules"] is leaf_modules


def test_replace_module_uses_isolated_fx_class_names():
    first = replace_module(
        nn.Sequential(nn.ReLU()),
        matches_module_class(nn.ReLU),
        lambda _: nn.Identity(),
        class_name_prefix="First_",
    )
    second = replace_module(
        nn.Sequential(nn.ReLU()),
        matches_module_class(nn.ReLU),
        lambda _: nn.Identity(),
        class_name_prefix="Second_",
    )

    assert type(first).__name__ == "First_Sequential"
    assert type(second).__name__ == "Second_Sequential"
    assert type(first) is not type(second)


def test_native_replacement_can_replace_the_root_module():
    replacement = nn.Identity()

    replaced = replace_module_native(
        nn.ReLU(),
        nn.ReLU,
        lambda original: replacement,
    )

    assert replaced is replacement


def test_native_replacement_preserves_shared_module_aliases():
    model = nn.Module()
    shared = nn.Linear(2, 2)
    model.left = shared
    model.right = shared
    replacements = []

    def create_replacement(original):
        replacement = nn.Linear(2, 2)
        replacements.append(replacement)
        return replacement

    replaced = replace_module_native(
        model,
        nn.Linear,
        create_replacement,
        clone=False,
    )

    assert len(replacements) == 1
    assert replaced.left is replacements[0]
    assert replaced.right is replacements[0]
