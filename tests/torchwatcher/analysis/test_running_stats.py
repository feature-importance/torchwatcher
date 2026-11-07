import pytest
import torch

from torchwatcher.analysis import (
    Bincount,
    Covariance,
    CrossCovariance,
    CrossIoU,
    IoU,
    Mean,
    Quantile,
    SecondMoment,
    Stat,
    TopK,
    Variance,
)


EMPTY_STATISTIC_TYPES = [
    Mean,
    Variance,
    Covariance,
    SecondMoment,
    Bincount,
    CrossCovariance,
    IoU,
    CrossIoU,
    Quantile,
    TopK,
]


@pytest.mark.parametrize(
    "statistic_type",
    EMPTY_STATISTIC_TYPES,
)
def test_empty_statistic_state_round_trips(statistic_type):
    statistic = statistic_type()

    state = statistic.state_dict()
    restored = statistic_type(state=state)

    assert restored.state_dict() == state


@pytest.mark.parametrize("statistic_type", EMPTY_STATISTIC_TYPES)
def test_empty_statistic_moves_to_cpu(statistic_type):
    statistic_type().cpu_()


@pytest.mark.parametrize("statistic_type", EMPTY_STATISTIC_TYPES)
def test_empty_statistic_file_round_trips(statistic_type, tmp_path):
    path = tmp_path / "statistic.npz"
    statistic = statistic_type()

    statistic.save(path)
    restored = statistic_type()
    restored.load(path)

    assert restored.state_dict() == statistic.state_dict()


def test_topk_is_a_stat_and_initialised_state_round_trips():
    statistic = TopK(k=2)
    statistic.add(torch.tensor([3.0, 1.0, 2.0]))

    restored = TopK(state=statistic.state_dict())

    assert isinstance(restored, Stat)
    expected_values, expected_indices = statistic.topk()
    actual_values, actual_indices = restored.topk()
    torch.testing.assert_close(actual_values, expected_values)
    torch.testing.assert_close(actual_indices, expected_indices)


def test_initialised_quantile_state_round_trips():
    statistic = Quantile(r=8)
    statistic.add(torch.arange(12.0).reshape(6, 2))

    restored = Quantile(state=statistic.state_dict())

    torch.testing.assert_close(
        restored.quantiles([0.25, 0.5, 0.75]),
        statistic.quantiles([0.25, 0.5, 0.75]),
    )


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
    left_centred = left_rows - left_rows.mean(dim=0)
    right_centred = right_rows - right_rows.mean(dim=0)
    expected_covariance = (
        left_centred.T @ right_centred / (left_rows.shape[0] - 1)
    )

    means = statistic.mean()
    torch.testing.assert_close(means[0], left_rows.mean(dim=0))
    torch.testing.assert_close(means[1], right_rows.mean(dim=0))
    torch.testing.assert_close(statistic.covariance(), expected_covariance)
