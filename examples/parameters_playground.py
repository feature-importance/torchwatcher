import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from torchwatcher.parameters import select_parameters
    import torchwatcher.parameters.selectors as sel
    import torch.nn as nn

    from torchvision.models import resnet18

    model = resnet18()
    return model, nn, resnet18, sel, select_parameters


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
def _(nn, resnet18, sel):
    import torch
    from torchwatcher.logging import InMemorySink
    from torchwatcher.training import EveryNSteps, ParameterTracker
    from torchbearer import Trial
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor, Compose
    import torchvision.transforms.v2 as v2

    model2 = resnet18()

    sink = InMemorySink()
    tracker = ParameterTracker(
        sel.is_weight,
        {
            "update_norm": lambda step, info, state: step.norm(),
            "gradient_norm": lambda step, info, state: info.parameter.grad.norm(),
        },
        sink,
        schedule=EveryNSteps(10),
        event="optimizer_step"
    )

    data = MNIST(root="/Users/jsh2/Data/", train=True, transform=Compose([ToTensor(), v2.RGB()]))
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=32)

    optim = torch.optim.Adam(sel.is_weight.parameters(model2), 0.001)
    trial = Trial(model2, optim, nn.CrossEntropyLoss(), metrics=["acc"], callbacks=[tracker]).to("mps")
    trial.with_train_generator(loader).run(4)
    return (
        Compose,
        EveryNSteps,
        InMemorySink,
        ParameterTracker,
        ToTensor,
        Trial,
        sink,
        torch,
        v2,
    )


@app.cell
def _(sink):
    from torchwatcher.logging import observations_to_dataframe, path_contains

    frame = observations_to_dataframe(sink.observations)
    frame
    return frame, observations_to_dataframe, path_contains


@app.cell
def _(frame, path_contains):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # plt.figure()
    # sns.lineplot(path_contains(frame, ["update_norm"]), x="global_step", y="value")
    # plt.show()

    # plt.figure()
    # sns.lineplot(path_contains(frame, ["gradient_norm"]), x="global_step", y="value")
    # plt.show()

    max_grad = path_contains(frame, ["gradient_norm"])['value'].max()
    max_update = path_contains(frame, ["update_norm"])['value'].max()

    layers = set([i[:-1] for i in frame['path'].unique()])
    for layer in layers:
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_title('.'.join(layer))
        ax1.set_xlabel('step')
        ax1.set_ylabel('gradient_norm', color=color)
        ax1.set_ylim(0, max_grad)
        sns.lineplot(path_contains(frame, [*layer, "gradient_norm"]), x="global_step", y="value", color=color, ax=ax1)
        ax1.tick_params(axis='y', labelcolor=color)
    
        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        ax2.set_ylim(0, max_update)
        color = 'tab:blue'
        ax2.set_ylabel('update_norm', color=color)  # we already handled the x-label with ax1
        sns.lineplot(path_contains(frame, [*layer, "update_norm"]), x="global_step", y="value", color=color, ax=ax2)
        ax2.tick_params(axis='y', labelcolor=color)
    
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
    return


@app.cell
def _(
    CIFAR10,
    Compose,
    EveryNSteps,
    InMemorySink,
    ParameterTracker,
    ToTensor,
    Trial,
    nn,
    observations_to_dataframe,
    resnet18,
    sel,
    torch,
    v2,
):
    from copy import deepcopy
    from tqdm import tqdm

    def compare(epochs=1, steps=1):
        base = resnet18()
        data = CIFAR10(root="/Users/jsh2/Data/", train=True, transform=Compose([ToTensor(), v2.RGB()]))
        loader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=32)

        sink = InMemorySink()
        tracker = ParameterTracker(
            sel.matches_module_name("conv1") & sel.is_weight,
            {"update_norm": lambda step, info, state: step.norm()},
            sink,
            schedule=EveryNSteps(steps),
            event="optimizer_step"
        )

        model = deepcopy(base)
        optim = torch.optim.Adam(sel.is_weight.parameters(model), 0.001)
        trial = Trial(model, optim, nn.CrossEntropyLoss(), metrics=["acc"], callbacks=[tracker]).to("mps")
        trial.with_train_generator(loader).run(epochs)


        model = deepcopy(base).to('mps')
        optim = torch.optim.Adam(sel.is_weight.parameters(model), 0.001)

        trace = []
        counter = 1
        for epoch in range(epochs):
            for (X, y) in tqdm(loader):
                optim.zero_grad()
                pred = model(X.to('mps'))
                loss = nn.functional.cross_entropy(pred, y.to('mps'))
                loss.backward()

                before = model.conv1.weight.clone()
                optim.step()
                after = model.conv1.weight.clone()

                if counter % steps == 0:
                    trace.append((after.detach() - before.detach()).norm().cpu().item())
        
                counter +=1

        return observations_to_dataframe(sink.observations)['value'], trace

    return (compare,)


@app.cell
def _(compare):
    tw, man = compare()
    return man, tw


@app.cell
def _(man, tw):
    for i in range(len(man)):
        print(tw[i], man[i])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
