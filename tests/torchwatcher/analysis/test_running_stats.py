import torch

from torchwatcher.analysis import CrossCovariance, Variance


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


def test_cross_covariance_accepts_non_contiguous_feature_maps():
    generator = torch.Generator().manual_seed(1)
    left = torch.randn(2, 3, 4, 5, generator=generator).transpose(2, 3)
    right = torch.randn(2, 2, 4, 5, generator=generator).transpose(2, 3)
    assert not left.is_contiguous()
    assert not right.is_contiguous()
    statistic = CrossCovariance()

    statistic.add(left, right)

    left_rows = left.reshape(2, 3, -1).permute(0, 2, 1).reshape(-1, 3)
    right_rows = right.reshape(2, 2, -1).permute(0, 2, 1).reshape(-1, 2)
    left_centered = left_rows - left_rows.mean(dim=0)
    right_centered = right_rows - right_rows.mean(dim=0)
    expected_covariance = (
        left_centered.T @ right_centered / (left_rows.shape[0] - 1)
    )

    means = statistic.mean()
    torch.testing.assert_close(means[0], left_rows.mean(dim=0))
    torch.testing.assert_close(means[1], right_rows.mean(dim=0))
    torch.testing.assert_close(statistic.covariance(), expected_covariance)
