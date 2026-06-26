import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reproducing the tunnel effect

    This notebook reproduces the Figure 1 workflow from *The Tunnel Effect:
    Building Data Representations in Deep Neural Networks* for a CIFAR-10
    pretrained VGG19. It tracks the numerical rank and linear-probe accuracy
    of representations across the 18 ReLU layers used in the paper.
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

    from model_utilities.models.cifar_vgg import vgg19, VGG19_Weights

    from torchwatcher.analysis.analysis import AnalyzerList
    from torchwatcher.analysis.linear_probe import LinearProbe
    from torchwatcher.analysis.rank import RankAnalyzer
    from torchwatcher.interjection import interject_by_match, node_selector

    return (
        AnalyzerList,
        CIFAR10,
        DataLoader,
        LinearProbe,
        Path,
        RankAnalyzer,
        Subset,
        VGG19_Weights,
        interject_by_match,
        nn,
        node_selector,
        plt,
        torch,
        vgg19,
    )


@app.cell
def _(Path, torch):
    # Practical defaults. Increase the sample caps and probe epochs for a
    # closer, slower reproduction of the paper figure.
    DATA_ROOT = Path("~/data").expanduser()
    CACHE_PATH = Path(".cache/tunnel-effect-results.pt")

    TRAIN_SAMPLE_CAP = 2048
    TEST_SAMPLE_CAP = 10000
    BATCH_SIZE = 64
    PROBE_EPOCHS = 8
    PROBE_LR = 1e-3
    RANK_FEATURE_DIM = 8000
    RANK_THRESHOLD = 1e-3
    TUNNEL_ACC_FRACTION = 0.95
    NUM_WORKERS = 0
    USE_CACHE = True

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return (
        BATCH_SIZE,
        CACHE_PATH,
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
        USE_CACHE,
    )


@app.cell
def _(
    BATCH_SIZE,
    CIFAR10,
    DATA_ROOT,
    DEVICE,
    DataLoader,
    NUM_WORKERS,
    Subset,
    TEST_SAMPLE_CAP,
    TRAIN_SAMPLE_CAP,
    VGG19_Weights,
    torch,
    vgg19,
):
    torch.manual_seed(0)

    weights = VGG19_Weights.CIFAR10_s0
    model = vgg19(weights=weights).to(DEVICE)
    transform = weights.transforms()

    train_data = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
    test_data = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)

    train_subset = Subset(
        train_data, range(min(TRAIN_SAMPLE_CAP, len(train_data)))
    )
    test_subset = Subset(
        test_data, range(min(TEST_SAMPLE_CAP, len(test_data)))
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return model, test_loader, train_loader


@app.cell
def _(
    AnalyzerList,
    DEVICE,
    LinearProbe,
    PROBE_LR,
    RANK_FEATURE_DIM,
    RANK_THRESHOLD,
    RankAnalyzer,
    interject_by_match,
    model,
    nn,
    node_selector,
    torch,
):
    rank_collector = RankAnalyzer(
        n=RANK_FEATURE_DIM,
        threshold=RANK_THRESHOLD,
    )
    probe_trainer = LinearProbe(
        10,
        partial_optim=lambda params: torch.optim.Adam(params, lr=PROBE_LR),
        criterion=nn.CrossEntropyLoss(),
    )
    analyzer = AnalyzerList(rank_collector, probe_trainer)

    watched_model = interject_by_match(
        model,
        node_selector.Activations.is_relu,
        analyzer,
    ).to(DEVICE)
    return analyzer, probe_trainer, rank_collector, watched_model


@app.cell
def _(
    BATCH_SIZE,
    CACHE_PATH,
    PROBE_EPOCHS,
    PROBE_LR,
    RANK_FEATURE_DIM,
    RANK_THRESHOLD,
    TEST_SAMPLE_CAP,
    TRAIN_SAMPLE_CAP,
    USE_CACHE,
    torch,
):
    def cache_key():
        return {
            "train_sample_cap": TRAIN_SAMPLE_CAP,
            "test_sample_cap": TEST_SAMPLE_CAP,
            "batch_size": BATCH_SIZE,
            "probe_epochs": PROBE_EPOCHS,
            "probe_lr": PROBE_LR,
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


@app.cell
def _(
    DEVICE,
    PROBE_EPOCHS,
    analyzer,
    load_cached_results,
    probe_trainer,
    rank_collector,
    save_cached_results,
    test_loader,
    torch,
    train_loader,
    watched_model,
):
    def set_analyzers(*, rank_enabled, probe_enabled, probe_training):
        rank_collector.enabled = rank_enabled
        probe_trainer.enabled = probe_enabled
        probe_trainer.train(probe_training)


    def evaluate_final_model():
        set_analyzers(
            rank_enabled=False,
            probe_enabled=False,
            probe_training=False,
        )
        watched_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                logits = watched_model(inputs)
                correct += (logits.argmax(dim=1) == targets).sum().item()
                total += targets.numel()
        return correct / total


    def collect_ranks():
        rank_collector.reset()
        set_analyzers(
            rank_enabled=True,
            probe_enabled=False,
            probe_training=False,
        )

        watched_model.eval()
        with torch.no_grad():
            for inputs, _targets in test_loader:
                watched_model(inputs.to(DEVICE))

        return rank_collector.to_dict()


    def train_probes():
        set_analyzers(
            rank_enabled=False,
            probe_enabled=True,
            probe_training=True,
        )
        watched_model.eval()

        for _epoch in range(PROBE_EPOCHS):
            for inputs, targets in train_loader:
                probe_trainer.reset()
                analyzer.targets = targets.to(DEVICE)
                watched_model(inputs.to(DEVICE))
                probe_trainer.train_step()


    def evaluate_probes():
        probe_trainer.reset()
        set_analyzers(
            rank_enabled=False,
            probe_enabled=True,
            probe_training=False,
        )
        watched_model.eval()

        with torch.no_grad():
            for inputs, targets in test_loader:
                analyzer.targets = targets.to(DEVICE)
                watched_model(inputs.to(DEVICE))

        return probe_trainer.to_dict()


    def probe_accuracy(score):
        if "acc" in score:
            return score["acc"]

        for key, value in score.items():
            if "acc" in key.lower():
                return value

        raise KeyError(f"Could not find an accuracy metric in {score.keys()}")


    def run_experiment():
        cached = load_cached_results()
        if cached is not None:
            return cached

        final_model_accuracy = evaluate_final_model()
        ranks = collect_ranks()
        train_probes()
        probe_scores = evaluate_probes()

        layer_names = list(ranks.keys())[:18]
        results = {
            "layer_names": layer_names,
            "layers": list(range(1, len(layer_names) + 1)),
            "ranks": [ranks[name]["features_rank"] for name in layer_names],
            "probe_acc": [
                probe_accuracy(probe_scores[name]) for name in layer_names
            ],
            "final_model_accuracy": final_model_accuracy,
        }
        save_cached_results(results)
        return results


    results = run_experiment()
    results
    return (results,)


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


if __name__ == "__main__":
    app.run()
