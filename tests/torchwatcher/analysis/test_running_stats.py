import torch

from torchwatcher.analysis import Variance


def test_variance_matches_torch_across_uneven_batches():
    first = torch.tensor([
        [0.0, 10.0],
        [2.0, 12.0],
    ])
    second = torch.tensor([
        [4.0, 14.0],
        [6.0, 16.0],
        [8.0, 18.0],
    ])
    values = torch.cat((first, second))
    statistic = Variance()

    statistic.add(first)
    statistic.add(second)

    torch.testing.assert_close(statistic.mean(), values.mean(dim=0))
    torch.testing.assert_close(statistic.variance(), values.var(dim=0))
    torch.testing.assert_close(
        statistic.variance(unbiased=False),
        values.var(dim=0, unbiased=False),
    )


def test_variance_accepts_non_contiguous_input():
    values = torch.arange(24.0).reshape(2, 3, 4).transpose(1, 2)
    assert not values.is_contiguous()
    statistic = Variance()

    statistic.add(values)

    torch.testing.assert_close(statistic.mean(), values.mean(dim=0))
    torch.testing.assert_close(statistic.variance(), values.var(dim=0))
