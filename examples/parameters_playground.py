import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from torchwatcher.parameters import select_parameters
    import torchwatcher.parameters.selectors as sel
    import torch.nn as nn

    from torchvision.models import resnet18

    model = resnet18()
    return model, nn, sel, select_parameters


@app.cell
def _(model, sel):
    # select every parameter
    sel.all.select(model)
    return


@app.cell
def _(model, sel):
    # select every parameter requiring gradients
    sel.requires_grad.select(model)
    return


@app.cell
def _(model, sel):
    # select all weights (but not biases) of a model
    sel.is_weight.select(model)
    return


@app.cell
def _(model, sel):
    # Select all bias parameters in a model
    sel.is_bias.select(model)
    return


@app.cell
def _(model, nn, sel, select_parameters):
    # Select all weight parameters belonging to Conv2d:
    select_parameters(model, sel.matches_module_class(nn.Conv2d) & sel.is_weight)

    # or
    # (sel.matches_module_class(nn.Conv2d) & sel.is_weight).select(model)
    return


@app.cell
def _(model, sel):
    # .parameters on the selector returns an iterable over the parameters - useful for constructing optimisers
    list(sel.all.parameters(model))

    # ditto:
    # parameters(model, sel.all)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
