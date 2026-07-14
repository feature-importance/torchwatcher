import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from torchwatcher.analysis.analysis import Analyzer, AnalyzerState, PerClassAnalyzer
    from torchwatcher.analysis.running_stats import Variance
    from torchwatcher.interjection import interject_by_match
    from torchwatcher.interjection.node_selector import is_activation

    class GradientAccumulationAnalyzer(Analyzer):
        def __init__(self):
            super().__init__(gradient=True)

        def process_batch_state(self,
                            name: str,
                            state: AnalyzerState,
                            working_results: Variance | None) -> Variance:
            if working_results is None:
                working_results = Variance()

            gradients = state.output_gradients.mean(dim=(-2,-1))
            working_results.add(gradients)

            return working_results

        def finalise_result(self, name: str, result: Variance) -> dict:
            rec = dict()
            rec['var'] = result.variance()
            rec['mean'] = result.mean()
            return rec


    return (
        GradientAccumulationAnalyzer,
        PerClassAnalyzer,
        interject_by_match,
        is_activation,
    )


@app.cell
def _(
    GradientAccumulationAnalyzer,
    PerClassAnalyzer,
    interject_by_match,
    is_activation,
):
    from model_utilities.models.cifar_resnet import resnet18_3x3, ResNet18_3x3_Weights 
    grad_analyzer = GradientAccumulationAnalyzer()
    analyzer = PerClassAnalyzer(grad_analyzer)
    model = resnet18_3x3(weights=ResNet18_3x3_Weights.CIFAR10_s0)
    imodel = interject_by_match(model, is_activation, analyzer).to("cuda:0")

    return analyzer, imodel


@app.cell
def _(analyzer, imodel):
    from model_utilities.datasets import cifar10_loaders
    import torch

    train_loader, val_loader = cifar10_loaders('/home/am1g15/data/cifar10', batch_size=4)
    loss = torch.nn.CrossEntropyLoss()

    for x, y in val_loader:
        y = y.to("cuda:0")
        analyzer.targets = y
        pred = imodel(x.to("cuda:0"))
        loss(pred, y).backward()    

        print(analyzer.to_dict())
        break


    
    return


@app.cell
def _(analyzer):
    analyzer.analyzers[list(analyzer.analyzers.keys())[0]].working_results
    return


@app.cell
def _(analyzer):
    analyzer.to_dict()['layer4.1.relu_1']["var"].shape
    return


@app.cell
def _(imodel):
    from torchwatcher.drawing import draw_graph

    draw_graph(imodel)
    return


if __name__ == "__main__":
    app.run()
