import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import torch
    import torch.nn as nn

    from pathlib import Path

    from model_utilities.models.cifar_resnet import resnet18_3x3, ResNet18_3x3_Weights 
    from model_utilities.datasets import cifar10_loaders

    from torchwatcher.analysis.analysis import Analyser, AnalyserState, PerClassAnalyser
    from torchwatcher.analysis.running_stats import Variance
    from torchwatcher.interjection import interject_by_match
    from torchwatcher.interjection.node_selector import is_activation

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # class GradientAccumulationAnalyser(Analyser):
    #     def __init__(self):
    #         super().__init__(gradient=True)

    #     def process_batch_state(self,
    #                         name: str,
    #                         state: AnalyserState,
    #                         working_results: Variance | None) -> Variance:
    #         if working_results is None:
    #             working_results = Variance()

    #         gradients = state.output_gradients.mean(dim=(-2,-1))
    #         working_results.add(gradients)

    #         return working_results

    #     def finalise_result(self, name: str, result: Variance) -> dict:
    #         rec = dict()
    #         rec['var'] = result.variance()
    #         rec['mean'] = result.mean()
    #         return rec
    from torchwatcher.analysis.gradients import GradientAccumulationAnalyser

    return (
        GradientAccumulationAnalyser,
        Path,
        PerClassAnalyser,
        ResNet18_3x3_Weights,
        cifar10_loaders,
        device,
        interject_by_match,
        is_activation,
        resnet18_3x3,
        torch,
    )


@app.cell
def _(
    GradientAccumulationAnalyser,
    PerClassAnalyser,
    ResNet18_3x3_Weights,
    device,
    interject_by_match,
    is_activation,
    resnet18_3x3,
):
    grad_analyser = GradientAccumulationAnalyser()
    analyser = PerClassAnalyser(grad_analyser)
    model = resnet18_3x3(weights=ResNet18_3x3_Weights.CIFAR10_s0)
    imodel = interject_by_match(model, is_activation, analyser).to(device)
    return analyser, imodel


@app.cell
def _(Path, analyser, cifar10_loaders, device, imodel, torch):
    train_loader, val_loader = cifar10_loaders(Path('~/data/').expanduser(), batch_size=4)
    loss = torch.nn.CrossEntropyLoss()

    for x, y in val_loader:
        print("HERE")
        y = y.to(device)
        analyser.targets = y
        pred = imodel(x.to(device))
        loss(pred, y).backward() 
        break

    print(analyser.to_dict())
    return


@app.cell
def _(analyser):
    analyser.to_dict()#['layer4.1.relu_1']["var"].shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
