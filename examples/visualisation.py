import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Basic model drawing
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch

    from torchvision.models import convnext_tiny
    from torchwatcher.interjection import trace
    from torchwatcher.drawing import draw_graph_pretty

    model = convnext_tiny()
    traced = trace(model)
    mo.Html(draw_graph_pretty(traced, torch.empty(1,3,224,224)).create_svg().decode('utf-8'))
    return draw_graph_pretty, mo, model, torch, trace, traced


@app.cell
def _(mo, traced):
    from torchwatcher.drawing import draw_graph

    mo.Html(draw_graph(traced).create_svg().decode('utf-8'))
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Model rewriting

    Let's replace all activations with Identity
    """)
    return


@app.cell
def _(draw_graph_pretty, mo, model, torch):
    from torchwatcher.interjection import replace
    from torchwatcher.interjection.node_selector import is_activation

    import torch.nn as nn

    replaced = replace(model, is_activation, lambda x: nn.Identity())
    mo.Html(draw_graph_pretty(replaced, torch.empty(1,3,224,224)).create_svg().decode('utf-8'))
    return is_activation, nn


@app.cell
def _(mo):
    mo.md(r"""
    # Drawing models with multiple inputs and outputs
    """)
    return


@app.cell
def _(draw_graph_pretty, mo, nn, torch, trace):
    class MultiOutputModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x, x+1

    class MultiInputModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[0] + x[1]

    class MyModel(nn.Module):        
        def __init__(self):
            super().__init__()
            self.m1 = MultiOutputModel()
            self.m2 = MultiInputModel()

        def forward(self, x):
            return self.m2(self.m1(x))

    model2 = nn.Sequential(MultiOutputModel(), MultiInputModel())
    traced2 = trace(model2)
    mo.Html(draw_graph_pretty(traced2, torch.empty(1,3,224,224)).create_svg().decode('utf-8'))
    return


@app.cell
def _(draw_graph_pretty, is_activation, mo, nn, torch):
    from torchvision.models import resnet18
    from torchwatcher.interjection import interject_by_match, WrappedForwardBackwardInterjection, ForwardInterjection

    class MyWrappedFwdBwd(WrappedForwardBackwardInterjection):
        def process(self, name, module, inputs, outputs):
            pass

        def process_backward(self, name, module, grad_input, grad_output):
            pass

    class MyForwardInterjection(ForwardInterjection):
        def process(self, name: str, module: nn.Module | None, inputs):
            pass


    model_rn18 = resnet18()
    # interjected_rn18 = interject_by_match(model_rn18, is_activation, MyForwardInterjection())
    interjected_rn18 = interject_by_match(model_rn18, is_activation, MyWrappedFwdBwd())
    mo.Html(draw_graph_pretty(interjected_rn18, torch.empty(1,3,224,224)).create_svg().decode('utf-8'))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
