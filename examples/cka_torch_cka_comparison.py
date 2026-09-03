import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Validating torchwatcher against torch-cka

    This notebook performs the same two-model comparison with torchwatcher and
    [`torch-cka` 0.21](https://pypi.org/project/torch-cka/), then asserts that
    the layer-by-layer matrices agree numerically.

    The comparison is deliberately controlled:

    - both libraries receive deep copies of the same model weights;
    - both see the same unshuffled, fixed-size batches;
    - both observe exactly the two ReLU layers;
    - torchwatcher uses `debiased=True` and float32 accumulation, matching
      `torch-cka`'s unbiased minibatch HSIC calculation.

    Install the reference package before running this notebook:

    ```bash
    pip install torch-cka==0.21 matplotlib tqdm
    ```
    """)
    return


@app.cell
def _():
    import copy

    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from torch_cka import CKA as TorchCKA

    from torchwatcher.analysis import LinearCKAAnalyser
    from torchwatcher.interjection import interject_by_match, node_selector

    return (
        DataLoader,
        LinearCKAAnalyser,
        TensorDataset,
        TorchCKA,
        copy,
        interject_by_match,
        nn,
        node_selector,
        plt,
        torch,
    )


@app.cell
def _(nn):
    class TinyMLP(nn.Module):
        def __init__(self, hidden_dims):
            super().__init__()
            first, second = hidden_dims
            self.linear1 = nn.Linear(6, first)
            self.relu1 = nn.ReLU()
            self.linear2 = nn.Linear(first, second)
            self.relu2 = nn.ReLU()
            self.classifier = nn.Linear(second, 3)

        def forward(self, inputs):
            features = self.relu1(self.linear1(inputs))
            features = self.relu2(self.linear2(features))
            return self.classifier(features)

    return (TinyMLP,)


@app.cell
def _(DataLoader, TensorDataset, TinyMLP, torch):
    torch.manual_seed(19)
    inputs = torch.randn(48, 6)
    comparison_loader = DataLoader(
        TensorDataset(inputs),
        batch_size=12,
        shuffle=False,
        drop_last=True,
    )

    base_model_a = TinyMLP((10, 7)).eval()
    base_model_b = TinyMLP((14, 5)).eval()
    selected_layers = ["relu1", "relu2"]
    return base_model_a, base_model_b, comparison_loader, selected_layers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run torchwatcher

    The two source-bound observers share one relational analyser. `run` accepts
    their source-to-model mapping and creates one batch transaction around both
    forwards before updating the HSIC totals.
    """)
    return


@app.cell
def _(
    LinearCKAAnalyser,
    base_model_a,
    base_model_b,
    comparison_loader,
    copy,
    interject_by_match,
    node_selector,
    torch,
):
    torchwatcher_cka = LinearCKAAnalyser(
        debiased=True,
        accumulation_device="cpu",
        accumulation_dtype=torch.float32,
    )
    torchwatcher_model_a = interject_by_match(
        copy.deepcopy(base_model_a),
        node_selector.Activations.is_relu,
        torchwatcher_cka.watch("model_a"),
    ).eval()
    torchwatcher_model_b = interject_by_match(
        copy.deepcopy(base_model_b),
        node_selector.Activations.is_relu,
        torchwatcher_cka.watch("model_b"),
    ).eval()

    torchwatcher_cka.run(
        {
            "model_a": torchwatcher_model_a,
            "model_b": torchwatcher_model_b,
        },
        comparison_loader,
    )

    torchwatcher_result = torchwatcher_cka.result("model_a", "model_b")
    return (torchwatcher_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run torch-cka

    `torch-cka` attaches ordinary forward hooks and drives the model forwards
    itself. Passing both loader arguments avoids any hidden dataset choice and
    makes the correspondence with the torchwatcher loop explicit.
    """)
    return


@app.cell
def _(
    TorchCKA,
    base_model_a,
    base_model_b,
    comparison_loader,
    copy,
    selected_layers,
):
    reference_cka = TorchCKA(
        copy.deepcopy(base_model_a),
        copy.deepcopy(base_model_b),
        model1_name="model_a",
        model2_name="model_b",
        model1_layers=selected_layers,
        model2_layers=selected_layers,
        device="cpu",
    )
    reference_cka.compare(comparison_loader, comparison_loader)
    reference_result = reference_cka.export()
    reference_matrix = reference_result["CKA"].detach().cpu()
    return reference_matrix, reference_result


@app.cell(hide_code=True)
def _(
    mo,
    reference_matrix,
    reference_result,
    selected_layers,
    torch,
    torchwatcher_result,
):
    torchwatcher_layers_a = [
        name.rsplit(".", 1)[-1] for name in torchwatcher_result.row_names
    ]
    torchwatcher_layers_b = [
        name.rsplit(".", 1)[-1] for name in torchwatcher_result.column_names
    ]

    assert torchwatcher_layers_a == reference_result["model1_layers"]
    assert torchwatcher_layers_b == reference_result["model2_layers"]
    assert torchwatcher_layers_a == selected_layers
    torch.testing.assert_close(
        torchwatcher_result.values,
        reference_matrix,
        rtol=1e-5,
        atol=1e-6,
    )

    absolute_error = (torchwatcher_result.values - reference_matrix).abs()
    maximum_error = absolute_error.max().item()
    mo.md(f"""
    ## Numerical check passed

    Both implementations produced the same **{len(selected_layers)} ×
    {len(selected_layers)}** CKA matrix.

    Maximum absolute difference: **`{maximum_error:.3e}`**
    """)
    return (absolute_error,)


@app.cell(hide_code=True)
def _(
    absolute_error,
    plt,
    reference_matrix,
    selected_layers,
    torchwatcher_result,
):
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    matrices = (
        torchwatcher_result.values,
        reference_matrix,
        absolute_error,
    )
    titles = ("torchwatcher", "torch-cka 0.21", "absolute difference")
    for axis, matrix, title in zip(axes, matrices, titles):
        if title == "absolute difference":
            image = axis.imshow(matrix, cmap="viridis")
        else:
            image = axis.imshow(matrix, vmin=0, vmax=1, cmap="magma")
        axis.set_xticks(
            range(len(selected_layers)),
            selected_layers,
            rotation=35,
            ha="right",
        )
        axis.set_yticks(range(len(selected_layers)), selected_layers)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.75)

    axes[0].set_ylabel("Model A layers")
    for axis in axes:
        axis.set_xlabel("Model B layers")
    figure.tight_layout()
    figure
    return


if __name__ == "__main__":
    app.run()
