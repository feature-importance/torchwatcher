import torch
from torch import nn

from torchwatcher.interjection import replace
from torchwatcher.interjection.node_selector import matches_module_class


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
