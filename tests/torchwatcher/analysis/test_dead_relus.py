import torch
from torch import nn

from torchwatcher.analysis import DeadReLU
from torchwatcher.interjection import interject_by_name


def test_dead_relu_counts_units_that_never_activate_across_batches():
    analyser = DeadReLU()
    watched_model = interject_by_name(
        nn.Sequential(nn.ReLU()),
        "0",
        analyser,
    )

    watched_model(torch.tensor([[1.0, -1.0, -1.0]]))
    watched_model(torch.tensor([[2.0, -1.0, 3.0]]))

    result = analyser.to_dict()["0"]
    assert analyser.working_results["0"].dtype == torch.bool
    assert result["dead_count"].item() == 1
    assert result["numel"] == 3
    assert result["sum"].item() == 2
