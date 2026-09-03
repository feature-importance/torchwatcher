import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Intervening on hooked features

    Interjections can change a feature as well as observe it. This notebook
    inserts a `ForwardInterjection` after a model's hidden activation, runs a
    baseline forward pass, then replaces one hidden feature before it reaches
    the classifier.

    The model is deliberately small and deterministic: its hidden features copy
    the two inputs, and its classifier prefers whichever feature is larger. With
    the default input, setting feature 0 to zero changes the logits and flips the
    prediction from class 0 to class 1. This demonstrates a causal feature
    intervention without modifying the model's `forward` method.
    """)
    return


@app.cell
def _():
    import torch
    from torch import nn

    from torchwatcher.interjection import ForwardInterjection, interject_by_name

    return ForwardInterjection, interject_by_name, nn, torch


@app.cell
def _(ForwardInterjection, nn, torch):
    class ToyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(2, 2, bias=False),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(2, 2, bias=False)

            # Hidden features copy the inputs. The classifier prefers class 0
            # when feature 0 dominates, and class 1 when feature 1 dominates.
            with torch.no_grad():
                self.features[0].weight.copy_(torch.eye(2))
                self.classifier.weight.copy_(
                    torch.tensor([
                        [2.0, -1.0],
                        [-1.0, 2.0],
                    ])
                )

        def forward(self, inputs):
            return self.classifier(self.features(inputs))

    class SetFeature(ForwardInterjection):
        """Replace one feature without modifying the hooked tensor in place."""

        def __init__(self):
            super().__init__()
            self.feature = 0
            self.value = 0.0
            self.enabled = False
            self.features_before = None
            self.features_after = None

        def process(self, name, module, features):
            self.features_before = features.detach().clone()
            if not self.enabled:
                self.features_after = self.features_before
                return None

            intervened = features.clone()
            intervened[..., self.feature] = self.value
            self.features_after = intervened.detach().clone()
            return intervened

    return SetFeature, ToyClassifier


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `SetFeature.process` receives the output of the hooked ReLU. Returning
    `None` leaves that tensor unchanged; returning the cloned and edited tensor
    sends the intervention to every downstream consumer. Cloning is important
    because an interjection must not mutate its input in place.
    """)
    return


@app.cell
def _(SetFeature, ToyClassifier, interject_by_name, torch):
    intervention = SetFeature()
    model = interject_by_name(
        ToyClassifier(),
        "features.1",  # ReLU output: [batch, hidden_features].
        intervention,
    ).eval()

    inputs = torch.tensor([[2.0, 1.0]])

    # Establish the baseline using the same interjected model, but with the
    # intervention disabled.
    with torch.no_grad():
        baseline_logits = model(inputs)
        baseline_features = intervention.features_after.clone()

    return baseline_features, baseline_logits, inputs, intervention, model


@app.cell
def _(mo):
    feature = mo.ui.dropdown(
        options={"Feature 0": 0, "Feature 1": 1},
        value="Feature 0",
        label="Feature to replace",
    )
    replacement = mo.ui.number(value=0.0, step=0.25, label="Replacement value")
    enabled = mo.ui.switch(value=True, label="Apply intervention")

    mo.hstack([feature, replacement, enabled], justify="start", gap=2)
    return enabled, feature, replacement


@app.cell
def _(enabled, feature, intervention, model, replacement, torch, inputs):
    intervention.feature = feature.value
    intervention.value = replacement.value
    intervention.enabled = enabled.value

    with torch.no_grad():
        intervened_logits = model(inputs)
        intervened_features = intervention.features_after.clone()

    return intervened_features, intervened_logits


@app.cell(hide_code=True)
def _(
    baseline_features,
    baseline_logits,
    intervened_features,
    intervened_logits,
    mo,
):
    baseline_class = baseline_logits.argmax(dim=-1).item()
    intervened_class = intervened_logits.argmax(dim=-1).item()

    mo.md(f"""
    ## Result

    | | Hidden features | Logits | Predicted class |
    |---|---|---|---:|
    | Baseline | `{baseline_features.tolist()}` | `{baseline_logits.tolist()}` | {baseline_class} |
    | Intervention | `{intervened_features.tolist()}` | `{intervened_logits.tolist()}` | {intervened_class} |

    **Prediction: class {baseline_class} → class {intervened_class}**
    """)
    return


if __name__ == "__main__":
    app.run()
