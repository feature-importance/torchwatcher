import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reproducing the tunnel effect

    This notebook reproduces the Figure 1 workflow from [*The Tunnel Effect:
    Building Data Representations in Deep Neural Networks*](https://proceedings.neurips.cc/paper_files/paper/2023/file/f249db9ab5975586f36df46f8958c008-Paper-Conference.pdf) for a CIFAR-10
    pretrained **ResNet18**.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch
    import torchbearer
    from torch import nn

    from model_utilities.datasets import cifar10_loaders
    from model_utilities.models.cifar_resnet import resnet18_3x3, ResNet18_3x3_Weights
    from torchvision.models.resnet import BasicBlock

    from torchwatcher.analysis.analysis import AnalyserList
    from torchwatcher.analysis.linear_probe import LinearProbe
    from torchwatcher.analysis.rank import RankAnalyser
    from torchwatcher.interjection import interject_by_module_class

    return (
        AnalyserList,
        BasicBlock,
        LinearProbe,
        Path,
        RankAnalyser,
        ResNet18_3x3_Weights,
        cifar10_loaders,
        interject_by_module_class,
        nn,
        plt,
        resnet18_3x3,
        torch,
        torchbearer,
    )


@app.cell
def _(Path, torch):
    # Practical defaults. Increase the sample caps and probe epochs for a
    # closer, slower reproduction of the paper figure.
    DATA_ROOT = Path("~/data").expanduser()

    TRAIN_SAMPLE_CAP = 2048
    TEST_SAMPLE_CAP = 10000
    BATCH_SIZE = 64
    PROBE_EPOCHS = 8
    PROBE_LR = 1e-3
    RANK_FEATURE_DIM = 8000
    RANK_THRESHOLD = 1e-3
    TUNNEL_ACC_FRACTION = 0.95
    NUM_WORKERS = 0

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return (
        BATCH_SIZE,
        DATA_ROOT,
        DEVICE,
        NUM_WORKERS,
        PROBE_EPOCHS,
        PROBE_LR,
        RANK_FEATURE_DIM,
        RANK_THRESHOLD,
        TEST_SAMPLE_CAP,
        TRAIN_SAMPLE_CAP,
        TUNNEL_ACC_FRACTION,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the model and data

    We load the CIFAR-10 pretrained resnet18 (seed-0 weights) and use the
    preprocessing transforms that ship with those weights, so the inputs match
    what the network saw during training. The `cifar10_loaders` helper from
    `model-utilities` constructs capped train and test loaders for us — the
    train loader feeds probe training, the test loader is used for rank
    collection, probe evaluation, and the final accuracy check.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    DATA_ROOT,
    DEVICE,
    NUM_WORKERS,
    ResNet18_3x3_Weights,
    TEST_SAMPLE_CAP,
    TRAIN_SAMPLE_CAP,
    cifar10_loaders,
    resnet18_3x3,
    torch,
):
    torch.manual_seed(0)

    weights = ResNet18_3x3_Weights.CIFAR10_s0
    model = resnet18_3x3(weights=weights).to(DEVICE)
    transform = weights.transforms()

    train_loader, test_loader = cifar10_loaders(
        DATA_ROOT,
        batch_size=BATCH_SIZE,
        train_sample_cap=TRAIN_SAMPLE_CAP,
        eval_sample_cap=TEST_SAMPLE_CAP,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        train_transform=transform,
        eval_transform=transform,
    )
    return model, test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attach the analysers

    Here is the core of the `torchwatcher` workflow. We instantiate the two
    analysers — a `RankAnalyser` to estimate per-layer feature rank and a
    `LinearProbe` (one 10-class linear classifier per layer) — and combine them
    in an `AnalyserList` so they observe the *same* forward pass.

    `interject_by_module_class` then rewrites the model so that every `BasicBlock`
    output is observed by the analysers. The returned `watched_model` behaves
    exactly like the original ResNet on the forward pass, while exposing eight
    measurement points across its four residual stages.
    """)
    return


@app.cell
def _(
    AnalyserList,
    BasicBlock,
    DEVICE,
    LinearProbe,
    PROBE_LR,
    RANK_FEATURE_DIM,
    RANK_THRESHOLD,
    RankAnalyser,
    interject_by_module_class,
    model,
    nn,
    torch,
):
    rank_collector = RankAnalyser(
        n=RANK_FEATURE_DIM,
        threshold=RANK_THRESHOLD,
    )
    probe_trainer = LinearProbe(
        10,
        partial_optimiser=lambda params: torch.optim.Adam(params, lr=PROBE_LR),
        criterion=nn.CrossEntropyLoss(),
    )
    analyser = AnalyserList(rank_collector, probe_trainer)

    watched_model = interject_by_module_class(
        model,
        BasicBlock,
        analyser,
    ).to(DEVICE)
    return analyser, probe_trainer, rank_collector, watched_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the experiment

    With the analysers attached we can drive the whole measurement pipeline.
    The backbone is always kept frozen and in `eval` mode; the
    `set_analysers` helper just toggles which analyser is active and whether
    the probes are in train or eval mode for a given pass:

    1. **`train_probes`** — for each batch, runs a forward pass, sets the
       targets on the analyser, and takes a `train_step` so every layer's linear
       probe learns to classify from that layer's representation. Crucially the
       probes are trained on detached features, so no gradients flow back into
       the backbone.
    2. **`evaluate_final_model`** — measures each trained probe's accuracy and the representation
       rank the on the test set. Also measures the full network's test accuracy which becomes the
       reference point for the 95% tunnel threshold.

    `run_experiment` orchestrates these steps and returns the per-layer ranks
    and probe accuracies alongside the final model accuracy. The expensive
    experiment block is wrapped in Marimo's persistent cache, so subsequent
    notebook runs can restore the results without retraining the probes.
    """)
    return


@app.cell
def _(
    DEVICE,
    PROBE_EPOCHS,
    analyser,
    mo,
    probe_trainer,
    rank_collector,
    test_loader,
    torchbearer,
    train_loader,
    watched_model,
):
    def set_analysers(*, rank_enabled, probe_enabled, training):
        analyser.train(training)
        analyser.enabled = (rank_enabled or probe_enabled)
        rank_collector.enabled = rank_enabled
        probe_trainer.enabled = probe_enabled

    def train_probes():
        set_analysers(
            rank_enabled=False,
            probe_enabled=True,
            training=True,
        )

        print("Training linear probes")
        trial = torchbearer.Trial(
            watched_model,
            metrics=["acc"],
            callbacks=[probe_trainer.callback(keep_model_eval=True)],
        )
        trial.with_generators(train_generator=train_loader).to(DEVICE)
        trial.run(PROBE_EPOCHS, verbose=2)

    def evaluate_final_model():
        set_analysers(
            rank_enabled=True,
            probe_enabled=True,
            training=False,
        )
        analyser.reset()

        print("Evaluating model")
        trial = torchbearer.Trial(
            watched_model,
            metrics=["acc"],
            callbacks=[probe_trainer.callback(train_probes=False)],
        )
        trial.with_generators(val_generator=test_loader).to(DEVICE)
        final_metrics = trial.evaluate(verbose=2)

        return final_metrics["val_acc"], rank_collector.to_dict(), probe_trainer.to_dict()

    def run_experiment():
        train_probes()
        final_model_accuracy, ranks, probe_scores = evaluate_final_model()

        layer_names = list(ranks)
        results = {
            "layer_names": layer_names,
            "layers": list(range(1, len(layer_names) + 1)),
            "ranks": [ranks[name]["features_rank"] for name in layer_names],
            "probe_acc": [probe_scores[name]['acc'] for name in layer_names],
            "final_model_accuracy": final_model_accuracy,
        }
        return results

    with mo.persistent_cache(name="tunnel-effect-results-resnet18"):
        results = run_experiment()
    results
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualise the tunnel

    Finally we recreate Figure 1: numerical rank (red, dashed) and linear-probe
    accuracy (blue) plotted against layer depth on a twin axis. We locate the
    **tunnel start** as the first layer whose probe accuracy reaches 95% of the
    full model's accuracy, and shade everything from there to the output.

    Reading the plot: probe accuracy climbs steeply over the early
    *extractor* layers and then plateaus — once inside the shaded region,
    deeper layers no longer improve linear separability. Meanwhile the
    numerical rank keeps falling through the tunnel, showing that those layers
    are compressing the representation onto a lower-dimensional manifold rather
    than adding task-relevant structure. That combination of saturated accuracy
    and shrinking rank is the signature of the tunnel effect.
    """)
    return


@app.cell
def _(TUNNEL_ACC_FRACTION, plt, results):
    layers = results["layers"]
    ranks = results["ranks"]
    probe_acc = results["probe_acc"]
    final_model_accuracy = results["final_model_accuracy"]
    threshold = final_model_accuracy * TUNNEL_ACC_FRACTION

    tunnel_start = next(
        (layer for layer, acc in zip(layers, probe_acc) if acc >= threshold),
        None,
    )

    fig, rank_axis = plt.subplots(figsize=(7.5, 4.5))
    acc_axis = rank_axis.twinx()

    rank_line = rank_axis.plot(
        layers,
        ranks,
        color="tab:red",
        linestyle="--",
        marker="o",
        label="Numerical rank",
    )
    acc_line = acc_axis.plot(
        layers,
        probe_acc,
        color="tab:blue",
        marker="o",
        label="Linear probing ACC",
    )

    if tunnel_start is not None:
        rank_axis.axvspan(
            tunnel_start,
            layers[-1],
            color="0.85",
            alpha=0.7,
            zorder=0,
        )

    rank_axis.set_xlabel("Layer")
    rank_axis.set_ylabel("Numerical rank")
    acc_axis.set_ylabel("Accuracy")
    rank_axis.set_xticks(layers)
    acc_axis.set_ylim(0, 1)
    rank_axis.grid(alpha=0.2)

    lines = rank_line + acc_line
    rank_axis.legend(
        lines,
        [line.get_label() for line in lines],
        loc="lower left",
    )
    fig.tight_layout()
    fig
    return final_model_accuracy, threshold, tunnel_start


@app.cell(hide_code=True)
def _(final_model_accuracy, mo, threshold, tunnel_start):
    mo.md(f"""
    Final model accuracy on the configured test subset:
    **{final_model_accuracy:.3f}**.

    The shaded tunnel begins at layer
    **{tunnel_start if tunnel_start is not None else "not reached"}**,
    using the 95% threshold (**{threshold:.3f}**).
    """)
    return


@app.cell
def _(model):
    model
    return


if __name__ == "__main__":
    app.run()
