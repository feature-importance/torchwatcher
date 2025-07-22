import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Recording input gradients

    This notebook shows how to capture input gradient maps with an interjection. Whilst there is slightly more
    code than if you were to do this manually (e.g. setting `requires_grad=True` on the input, etc - see below), the nice thing is that 
    you can trivially change this to interject the gradients at any point(s) of the network; this would be very difficult 
    to do manually!
    """
    )
    return


@app.cell
def _():
    # Load a pre-trained model and set up some data

    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    from model_utilities.models.cifar_resnet import resnet18_3x3, ResNet18_3x3_Weights
    from torchwatcher.analysis.analysis import Analyzer

    model = resnet18_3x3(weights=ResNet18_3x3_Weights.CIFAR10_s0)

    data = CIFAR10(root="/Users/jsh2/data", train=False,
                   transform=ResNet18_3x3_Weights.CIFAR10_s0.transforms())
    loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=0)
    return Analyzer, loader, model


@app.cell
def _(Analyzer, model):
    import torch
    from torch import nn
    from torchwatcher.interjection import interject_by_match, node_selector
    from torchwatcher.nn import GradientIdentity

    # Create a class to record gradients 
    class GradTracker(Analyzer):
        def __init__(self):
            super().__init__(gradient=True)

        def process_batch_state(self, name, state, working_results):
            if working_results is None:
                working_results = []
            grad = state.output_gradients.detach()
            working_results.append(grad)
            return working_results

        def finalise_result(self, name, result):
            return torch.cat(result, dim=0)

    gt = GradTracker()

    # interject the model. we use the GradientIdentity module to capture the input gradients
    # if you uncomment the `node_selector.Activations.is_relu |` you can record _all_ the 
    # intermediate feature map gradients following each relu too!
    model2 = interject_by_match(nn.Sequential(GradientIdentity(), model),
                                # node_selector.Activations.is_relu |
                                node_selector.matches_module_class(GradientIdentity),
                                gt)
    return gt, model2, torch


@app.cell
def _(gt, loader, model2, torch):
    gt.reset()

    # we then just need to perform forward-backward passes against the network outputs
    model2.eval()
    for _X, _y in loader:
        with torch.set_grad_enabled(True):
            _pred = model2(_X)
            # this is just backpropping against the max prediction for each batch item
            _pred.max(dim=1)[0].sum().backward()
            break # comment this to get more than just the first batch

    # gt.to_dict()['0'] contains the gradient maps
    print(gt.to_dict()['0'].shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If you wanted to simply capture input gradients manually without torchwatcher you could do:""")
    return


@app.cell
def _(gt, loader, model, torch):
    model.eval()
    _grads = []
    for _X, _y in loader:
        with torch.set_grad_enabled(True):
            _X.requires_grad = True
            _pred = model(_X)
            # this is just backpropping against the max prediction for each batch item
            _pred.max(dim=1)[0].sum().backward()
            _grads.append(_X.grad.detach())
            break

    _grads = torch.cat(_grads, dim=0)
    print(_grads.shape)
    print("Gradients match:", torch.allclose(_grads, gt.to_dict()['0']))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Or by using `torch.autograd.grad` directly:""")
    return


@app.cell
def _(gt, loader, model, torch):
    model.eval()
    _grads = []
    for _X, _y in loader:
        _X.requires_grad = True
        _pred = model(_X).max(dim=1)[0]
    
        _grads.append(torch.autograd.grad(_pred, _X, grad_outputs=torch.ones(_pred.shape[0]), create_graph=True)[0])
        break

    _grads = torch.cat(_grads, dim=0)
    print(_grads.shape)
    print("Gradients match:", torch.allclose(_grads, gt.to_dict()['0']))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
