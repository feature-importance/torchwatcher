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

    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import CIFAR10

    from model_utilities.models.cifar_resnet import resnet18_3x3

    from torchwatcher.analysis.rank import RankAnalyzer
    from torchwatcher.interjection import interject_by_match, node_selector

    from tqdm import tqdm

    return (
        CIFAR10,
        DataLoader,
        Path,
        RankAnalyzer,
        Subset,
        interject_by_match,
        nn,
        node_selector,
        plt,
        resnet18_3x3,
        torch,
        tqdm,
    )


@app.cell
def _(Path, torch):
    DATA_ROOT = Path("~/data").expanduser()
    CACHE_PATH = Path(".cache/rank-evolution-results.pt")

    EPOCHS = 20
    BATCH_SIZE = 128
    TRAIN_SAMPLE_CAP = 10000
    RANK_SAMPLE_CAP = 2048
    RANK_EVERY_N_BATCHES = 10
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    RANK_FEATURE_DIM = 4096
    RANK_THRESHOLD = 1e-3
    NUM_WORKERS = 2
    USE_CACHE = True

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

    The training loader is shuffled and used for optimisation. Rank snapshots
    are measured on a fixed, unshuffled subset of the training split so that
    each snapshot is based on the same images.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    CIFAR10,
    DATA_ROOT,
    DataLoader,
    NUM_WORKERS,
    RANK_SAMPLE_CAP,
    Subset,
    TRAIN_SAMPLE_CAP,
):
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    rank_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_data = CIFAR10(
        DATA_ROOT,
        train=True,
        download=True,
        transform=train_transform,
    )
    rank_data = CIFAR10(
        DATA_ROOT,
        train=True,
        download=True,
        transform=rank_transform,
    )

    train_subset = Subset(
        train_data,
        range(min(TRAIN_SAMPLE_CAP, len(train_data))),
    )
    rank_subset = Subset(
        rank_data,
        range(min(RANK_SAMPLE_CAP, len(rank_data))),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    rank_loader = DataLoader(
        rank_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
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
    interject_by_match,
    nn,
    node_selector,
    resnet18_3x3,
    torch,
):
    torch.manual_seed(0)

    model = resnet18_3x3(weights=None).to(DEVICE)
    rank_collector = RankAnalyzer(
        n=RANK_FEATURE_DIM,
        threshold=RANK_THRESHOLD,
    )
    watched_model = interject_by_match(
        model,
        node_selector.Activations.is_relu,
        rank_collector,
    ).to(DEVICE)
    rank_collector.enabled = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        watched_model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )
    return criterion, optimizer, rank_collector, scheduler, watched_model


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

    `measure_rank` switches the model to evaluation mode, resets the analyzer,
    and runs the fixed rank loader without gradients. `train_and_measure`
    records an initial snapshot before training starts, then records another
    one after every configured number of batches and at the end of each epoch.
    """)
    return


@app.cell
def _(
    DEVICE,
    EPOCHS,
    RANK_EVERY_N_BATCHES,
    criterion,
    load_cached_results,
    optimizer,
    rank_collector,
    rank_loader,
    save_cached_results,
    scheduler,
    torch,
    tqdm,
    train_loader,
    watched_model,
):
    def measure_rank(epoch, batch, global_step, train_loss=None):
        rank_collector.reset()
        rank_collector.enabled = True
        watched_model.eval()

        with torch.no_grad():
            for inputs, _targets in rank_loader:
                watched_model(inputs.to(DEVICE))

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
            "epoch": epoch,
            "batch": batch,
            "global_step": global_step,
            "train_loss": train_loss,
            "layer_names": layer_names,
            "ranks": rank_values,
            "normalized_ranks": normalized_rank_values,
        }

    def train_and_measure():
        cached = load_cached_results()
        if cached is not None:
            return cached

        snapshots = [measure_rank(0, 0, 0)]
        global_step = 0

        for epoch in range(1, EPOCHS + 1):
            watched_model.train()
            rank_collector.enabled = False
            running_loss = 0.0
            progress = tqdm(train_loader, desc=f"epoch {epoch}")

            for batch, (inputs, targets) in enumerate(progress, start=1):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                logits = watched_model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                global_step += 1
                running_loss += loss.item()
                avg_loss = running_loss / batch
                progress.set_postfix(loss=f"{avg_loss:.3f}")

                if global_step % RANK_EVERY_N_BATCHES == 0:
                    snapshots.append(
                        measure_rank(epoch, batch, global_step, avg_loss)
                    )
                    watched_model.train()
                    rank_collector.enabled = False

            scheduler.step()
            if snapshots[-1]["global_step"] != global_step:
                snapshots.append(
                    measure_rank(epoch, batch, global_step, avg_loss)
                )

        results = {
            "snapshots": snapshots,
            "layer_names": snapshots[0]["layer_names"],
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
        vmin=0,
        vmax=1,
    )
    _ax.set_xlabel("Training step")
    _ax.set_ylabel("ReLU layer")
    _ax.set_xticks(range(len(steps)))
    _ax.set_xticklabels(steps, rotation=45, ha="right")
    _ax.set_yticks(range(len(layer_names)))
    _ax.set_yticklabels(range(1, len(layer_names) + 1))
    _fig.colorbar(image, ax=_ax, label="Normalized rank")
    _fig.tight_layout()
    _fig
    return rank_matrix, snapshots, steps


@app.cell
def _(plt, rank_matrix, snapshots, steps):
    # selected_indices = sorted(
    #     {
    #         0,
    #         min(1, len(snapshots) - 1),
    #         len(snapshots) // 2,
    #         len(snapshots) - 1,
    #     }
    # )
    selected_indices = range(len(snapshots))
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
    return


@app.cell(hide_code=True)
def _(mo, results):
    mo.md(f"""
    Recorded **{len(results["snapshots"])}** rank snapshots across training.
    """)
    return


if __name__ == "__main__":
    app.run()
