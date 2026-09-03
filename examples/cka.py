import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparing neural representations with CKA

    Centered Kernel Alignment (CKA) measures how similarly two layers
    represent the same examples. It can compare layers with different feature
    dimensions, and is unchanged by isotropic scaling or an orthogonal change
    of basis.

    This notebook demonstrates both modes supported by torchwatcher:

    1. compare every selected layer with every other selected layer in one
       model;
    2. compare the selected layers of two different models.

    The models and dataset are deliberately small and deterministic, so the
    example runs locally without downloading data or pretrained weights.
    """)
    return


@app.cell
def _():
    import copy

    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from torchwatcher.analysis import LinearCKAAnalyser
    from torchwatcher.interjection import interject_by_match, node_selector

    return (
        DataLoader,
        LinearCKAAnalyser,
        TensorDataset,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data and models

    CKA requires corresponding rows to describe the same examples. We
    therefore use an unshuffled loader and send each batch to both models. The
    two networks intentionally have different hidden widths: CKA does not
    require their feature dimensions to match.
    """)
    return


@app.cell
def _(DataLoader, TensorDataset, TinyMLP, torch):
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = torch.randn(48, 6)
    loader = DataLoader(
        TensorDataset(inputs),
        batch_size=12,
        shuffle=False,
    )

    model_a = TinyMLP((10, 7)).eval()
    model_b = TinyMLP((14, 5)).eval()
    return device, loader, model_a, model_b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Layers within one model

    A relational analyser is attached through `watch(source)`. With one source,
    `LinearCKAAnalyser` automatically constructs the square, layer-by-layer
    comparison. Its `run` helper resets and enables the analyser, moves each
    batch to the model's device, and supplies the relational batch boundaries.
    """)
    return


@app.cell
def _(
    LinearCKAAnalyser,
    copy,
    device,
    interject_by_match,
    model_a,
    node_selector,
):
    within_model_cka = LinearCKAAnalyser()
    within_model = interject_by_match(
        copy.deepcopy(model_a),
        node_selector.Activations.is_relu,
        within_model_cka.watch("model_a"),
    ).to(device).eval()
    return within_model, within_model_cka


@app.cell
def _(loader, within_model, within_model_cka):
    within_model_cka.run(within_model, loader)
    within_result = within_model_cka.result()
    within_result.to_dict()
    return (within_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Layers across two models

    For a cross-model comparison, each model receives its own source-bound
    observer. The source name becomes part of every result label and, more
    importantly, keeps identically named graph nodes in the two models
    separate. Passing a source-to-model mapping lets `run` execute both models
    in the same transaction for every input batch.
    """)
    return


@app.cell
def _(
    LinearCKAAnalyser,
    device,
    interject_by_match,
    model_a,
    model_b,
    node_selector,
):
    between_models_cka = LinearCKAAnalyser()
    watched_a = interject_by_match(
        model_a,
        node_selector.Activations.is_relu,
        between_models_cka.watch("model_a"),
    ).to(device).eval()
    watched_b = interject_by_match(
        model_b,
        node_selector.Activations.is_relu,
        between_models_cka.watch("model_b"),
    ).to(device).eval()
    return between_models_cka, watched_a, watched_b


@app.cell
def _(between_models_cka, loader, watched_a, watched_b):
    between_models_cka.run(
        {"model_a": watched_a, "model_b": watched_b},
        loader,
    )
    between_result = between_models_cka.result("model_a", "model_b")
    between_result.to_dict()
    return (between_result,)


@app.cell(hide_code=True)
def _(between_result, mo, plt, within_result):
    def short_names(names):
        return [name.rsplit(".", 1)[-1] for name in names]

    def draw_result(axis, result, title):
        image = axis.imshow(result.values, vmin=0, vmax=1, cmap="magma")
        axis.set_xticks(
            range(len(result.column_names)),
            short_names(result.column_names),
            rotation=35,
            ha="right",
        )
        axis.set_yticks(
            range(len(result.row_names)),
            short_names(result.row_names),
        )
        axis.set_title(title)
        for row in range(result.values.shape[0]):
            for column in range(result.values.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{result.values[row, column]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                )
        return image

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    within_image = draw_result(axes[0], within_result, "Within model A")
    between_image = draw_result(axes[1], between_result, "Model A vs model B")
    axes[0].set_ylabel("Model A layers")
    axes[0].set_xlabel("Model A layers")
    axes[1].set_ylabel("Model A layers")
    axes[1].set_xlabel("Model B layers")
    figure.colorbar(between_image, ax=axes, label="Linear CKA", shrink=0.8)

    mo.vstack([
        mo.md(r"""
        ## Results

        The within-model diagonal is exactly one because every representation
        is being compared with itself. Off-diagonal and cross-model entries
        describe similarities between different learned representations.
        """),
        figure,
    ])
    return


if __name__ == "__main__":
    app.run()
