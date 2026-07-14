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
    # Rank evolution during training

    This notebook tracks how the numerical rank of intermediate feature
    representations changes while a CIFAR-10 model is being trained.

    The setup is intentionally close to the tunnel-effect example: we attach a
    `RankAnalyser` to every ReLU activation with `torchwatcher`, run images
    through the watched model, and plot the resulting per-layer ranks. The main
    difference is that the model is not frozen. We train a CIFAR-appropriate ResNet-18
    (`resnet18_3x3` from `model-utilities`) and take rank snapshots every few
    batches, because the representation geometry often moves fastest at the
    start of optimisation.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import functools

    import matplotlib.pyplot as plt
    import torch
    import torchbearer
    from torch import nn

    from model_utilities.datasets import cifar10_loaders
    from model_utilities.models.cifar_resnet import resnet18_3x3
    from model_utilities.training.modelfitting import get_device, set_seed

    from torchwatcher.analysis.rank import RankAnalyser
    from torchwatcher.interjection import interject_by_match, node_selector
    from torchwatcher.training import AnalyserEvaluation

    return (
        AnalyserEvaluation,
        Path,
        RankAnalyser,
        cifar10_loaders,
        functools,
        get_device,
        interject_by_match,
        nn,
        node_selector,
        plt,
        resnet18_3x3,
        set_seed,
        torch,
        torchbearer,
    )


@app.cell
def _(Path):
    DATA_ROOT = Path("~/data").expanduser()

    EPOCHS = 2
    BATCH_SIZE = 128
    TRAIN_SAMPLE_CAP = 10000
    RANK_SAMPLE_CAP = 1024
    RANK_EVERY_N_BATCHES = 50
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    RANK_FEATURE_DIM = 4096
    RANK_THRESHOLD = 1e-3
    NUM_WORKERS = 2

    DEVICE = "auto"
    return (
        BATCH_SIZE,
        DATA_ROOT,
        DEVICE,
        EPOCHS,
        LEARNING_RATE,
        MOMENTUM,
        NUM_WORKERS,
        RANK_EVERY_N_BATCHES,
        RANK_FEATURE_DIM,
        RANK_SAMPLE_CAP,
        RANK_THRESHOLD,
        TRAIN_SAMPLE_CAP,
        WEIGHT_DECAY,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load CIFAR-10

    The `cifar10_loaders` from `model-utilities` creates train and validation loaders with the standard
    CIFAR-10 transforms. The sample caps keep the notebook useful for quick
    demos while making it easy to scale back up.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    DATA_ROOT,
    NUM_WORKERS,
    RANK_SAMPLE_CAP,
    TRAIN_SAMPLE_CAP,
    cifar10_loaders,
):
    train_loader, rank_loader = cifar10_loaders(
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        train_sample_cap=TRAIN_SAMPLE_CAP,
        eval_sample_cap=RANK_SAMPLE_CAP,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    return rank_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attach rank tracking

    `interject_by_match` rewrites the model so that every ReLU output is passed
    to the rank analyser. We keep the analyser disabled during optimisation and
    enable it only during snapshot passes, so the training steps stay ordinary
    PyTorch training steps.
    """)
    return


@app.cell
def _(
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MOMENTUM,
    RANK_FEATURE_DIM,
    RANK_THRESHOLD,
    RankAnalyser,
    WEIGHT_DECAY,
    functools,
    get_device,
    interject_by_match,
    nn,
    node_selector,
    resnet18_3x3,
    set_seed,
    torch,
    torchbearer,
):
    set_seed(0)
    device = get_device(DEVICE)

    model = resnet18_3x3(weights=None, num_classes=10).to(device)

    rank_collector = RankAnalyser(n=RANK_FEATURE_DIM, threshold=RANK_THRESHOLD)
    rank_collector.enabled = False

    watched_model = interject_by_match(model, node_selector.Activations.is_relu, rank_collector).to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(watched_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    scheduler = functools.partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=EPOCHS)
    lr_callback = torchbearer.callbacks.TorchScheduler(scheduler)
    return (
        criterion,
        device,
        lr_callback,
        optimiser,
        rank_collector,
        watched_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train and snapshot

    The training itself is a plain Torchbearer `Trial`. A torchwatcher
    `AnalyserEvaluation` handles the analyser housekeeping: it resets and
    enables the rank analyser, runs the fixed loader with the model in eval
    mode, stores the analyser's native result, and then restores training
    state. Behind the scenes, timing is delegated to the callback helpers from
    `model-utilities`. Marimo's persistent cache wraps the experiment so
    rerunning the notebook can restore the rank snapshots directly.
    """)
    return


@app.cell
def _(
    AnalyserEvaluation,
    EPOCHS,
    RANK_EVERY_N_BATCHES,
    criterion,
    device,
    lr_callback,
    mo,
    optimiser,
    rank_collector,
    rank_loader,
    torchbearer,
    train_loader,
    watched_model,
):
    def train_and_measure():
        rank_evaluation = AnalyserEvaluation(
            rank_collector,
            rank_loader,
        )

        trial = torchbearer.Trial(
            watched_model,
            optimiser,
            criterion,
            metrics=["loss", "acc", "lr"],
            callbacks=[
                *rank_evaluation.callbacks(
                    every_n_batches=RANK_EVERY_N_BATCHES,
                    include_start=True,
                    include_end=True,
                ),
                lr_callback,
            ],
        )
        trial.with_generators(train_generator=train_loader).to(device)
        history = trial.run(EPOCHS, verbose=2)

        results = {
            "history": history,
            "snapshots": rank_evaluation.records,
            "layer_names": list(rank_evaluation.records[0]["result"].keys()),
        }
        return results

    with mo.persistent_cache(name="rank-evolution-results"):
        results = train_and_measure()
    results
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot rank evolution

    The heatmap gives a compact overview of when each layer's representation
    rank changes. The line plot below it shows selected snapshots across depth,
    which is often easier to read for the first few training updates.
    """)
    return


@app.cell
def _(plt, results, torch):
    snapshots = results["snapshots"]
    layer_names = results["layer_names"]
    steps = [snapshot["global_step"] for snapshot in snapshots]
    rank_matrix = torch.tensor(
        [
            [
                snapshot["result"][name].get("features_rank", float("nan"))
                for name in layer_names
            ]
            for snapshot in snapshots
        ],
        dtype=torch.float32,
    )

    _fig, _ax = plt.subplots(figsize=(8, 4.8))
    image = _ax.imshow(
        rank_matrix.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    _ax.set_xlabel("Training step")
    _ax.set_ylabel("ReLU layer")
    _ax.set_xticks(range(len(steps)))
    _ax.set_xticklabels(steps, rotation=45, ha="right")
    _ax.set_yticks(range(len(layer_names)))
    _ax.set_yticklabels(range(1, len(layer_names) + 1))
    _fig.colorbar(image, ax=_ax, label="Rank")
    _fig.tight_layout()
    _fig
    return rank_matrix, snapshots, steps


@app.cell
def _(plt, rank_matrix, snapshots, steps):
    selected_indices = sorted(
        {
            0,
            min(1, len(snapshots) - 1),
            len(snapshots) // 2,
            len(snapshots) - 1,
        }
    )
    # selected_indices = range(len(snapshots))
    layers = list(range(1, rank_matrix.shape[1] + 1))

    _fig, _ax = plt.subplots(figsize=(7.5, 4.5))
    for index in selected_indices:
        _ax.plot(
            layers,
            rank_matrix[index].tolist(),
            marker="o",
            label=f"step {steps[index]}",
        )

    _ax.set_xlabel("ReLU layer")
    _ax.set_ylabel("Rank")
    # _ax.set_ylim(0, 1.05)
    _ax.set_xticks(layers)
    _ax.grid(alpha=0.2)
    _ax.legend()
    _fig.tight_layout()
    _fig
    return (layers,)


@app.cell(hide_code=True)
def _(mo, results):
    mo.md(f"""
    Recorded **{len(results["snapshots"])}** rank snapshots across training.
    """)
    return


@app.cell
def _(layers, mo, plt, rank_matrix):
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    line, = ax.plot(layers, rank_matrix[0])

    ax.set_ylim(0, rank_matrix.max().item() * 1.05)

    def animate(i):
        line.set_ydata(rank_matrix[i])  # update the data.
        return line,


    ani = animation.FuncAnimation(
        fig, animate, interval=1, frames=rank_matrix.shape[0])

    # mo.Html(ani.to_jshtml())
    mo.Html(ani.to_html5_video())
    ani.save("movie.mp4")
    return


if __name__ == "__main__":
    app.run()
