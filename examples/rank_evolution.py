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
    `RankAnalyzer` to every ReLU activation with `torchwatcher`, run images
    through the watched model, and plot the resulting per-layer ranks. The main
    difference is that the model is not frozen. We train a CIFAR ResNet-18
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

    from torchwatcher.analysis.rank import RankAnalyzer
    from torchwatcher.interjection import interject_by_match, node_selector
    from torchwatcher.training import PeriodicEvaluation

    return (
        Path,
        PeriodicEvaluation,
        RankAnalyzer,
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
    CACHE_PATH = Path(".cache/rank-evolution-results.pt")

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
    USE_CACHE = True

    DEVICE = "auto"
    return (
        BATCH_SIZE,
        CACHE_PATH,
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
        USE_CACHE,
        WEIGHT_DECAY,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load CIFAR-10

    `model-utilities` builds the train and validation loaders with the standard
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
    to the rank analyzer. We keep the analyzer disabled during optimisation and
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
    RankAnalyzer,
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
    rank_collector = RankAnalyzer(
        n=RANK_FEATURE_DIM,
        threshold=RANK_THRESHOLD,
    )
    watched_model = interject_by_match(
        model,
        node_selector.Activations.is_relu,
        rank_collector,
    ).to(device)
    rank_collector.enabled = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        watched_model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = functools.partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=EPOCHS)
    lr_callback = torchbearer.callbacks.TorchScheduler(scheduler)
    return (
        criterion,
        device,
        lr_callback,
        optimizer,
        rank_collector,
        watched_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Caching helpers

    The cache is keyed by the settings that affect training and rank
    collection. Disable `USE_CACHE` above to force a fresh run.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    CACHE_PATH,
    EPOCHS,
    LEARNING_RATE,
    MOMENTUM,
    RANK_EVERY_N_BATCHES,
    RANK_FEATURE_DIM,
    RANK_SAMPLE_CAP,
    RANK_THRESHOLD,
    TRAIN_SAMPLE_CAP,
    USE_CACHE,
    WEIGHT_DECAY,
    torch,
):
    def cache_key():
        return {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "train_sample_cap": TRAIN_SAMPLE_CAP,
            "rank_sample_cap": RANK_SAMPLE_CAP,
            "rank_every_n_batches": RANK_EVERY_N_BATCHES,
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "rank_feature_dim": RANK_FEATURE_DIM,
            "rank_threshold": RANK_THRESHOLD,
        }

    def load_cached_results():
        if not USE_CACHE or not CACHE_PATH.exists():
            return None

        cached = torch.load(CACHE_PATH, map_location="cpu")
        if cached.get("cache_key") != cache_key():
            return None
        return cached["results"]

    def save_cached_results(results):
        if not USE_CACHE:
            return
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cache_key": cache_key(),
                "results": results,
            },
            CACHE_PATH,
        )

    return load_cached_results, save_cached_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train and snapshot

    The training itself is a plain Torchbearer `Trial`. A torchwatcher
    `PeriodicEvaluation` callback handles the training-time measurement rhythm:
    it pauses training every configured number of batches, switches the model
    to eval mode, runs the rank evaluator without gradients, stores the
    snapshot, and then restores training mode.
    """)
    return


@app.cell
def _(
    EPOCHS,
    PeriodicEvaluation,
    RANK_EVERY_N_BATCHES,
    criterion,
    device,
    load_cached_results,
    lr_callback,
    optimizer,
    rank_collector,
    rank_loader,
    save_cached_results,
    torchbearer,
    train_loader,
    watched_model,
):
    def measure_rank(model, loader, _state, *, global_step, event):
        rank_collector.reset()
        rank_collector.enabled = True

        for inputs, _targets in loader:
            model(inputs.to(next(model.parameters()).device))

        rank_collector.enabled = False
        ranks = rank_collector.to_dict()
        layer_names = list(ranks.keys())
        rank_values = [
            ranks[name].get("features_rank", float("nan"))
            for name in layer_names
        ]
        normalized_rank_values = [
            ranks[name].get("normalized_features_rank", float("nan"))
            for name in layer_names
        ]
        return {
            "layer_names": layer_names,
            "ranks": rank_values,
            "normalized_ranks": normalized_rank_values,
        }

    def train_and_measure():
        cached = load_cached_results()
        if cached is not None:
            return cached

        rank_callback = PeriodicEvaluation(
            measure_rank,
            loader=rank_loader,
            every_n_batches=RANK_EVERY_N_BATCHES,
            include_start=True,
            include_end=True,
        )

        trial = torchbearer.Trial(
            watched_model,
            optimizer,
            criterion,
            metrics=["loss", "acc", "lr"],
            callbacks=[rank_callback, lr_callback],
        )
        trial.with_generators(train_generator=train_loader).to(device)
        history = trial.run(EPOCHS, verbose=2)

        results = {
            "history": history,
            "snapshots": rank_callback.records,
            "layer_names": rank_callback.records[0]["layer_names"],
        }
        save_cached_results(results)
        return results

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
        [snapshot["ranks"] for snapshot in snapshots],
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
