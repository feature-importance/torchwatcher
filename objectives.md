# TorchWatcher: A Programming Framework for Dissecting Models

This project, as well as many others we are working on, requires
"opening up" the model, extracting various statistics, carrying out
interventions (e.g. making small changes to the weights or setting the
layer output to a fixed value for all inputs), etc. So far, we "hacked"
this bit in the sense that we wrote a quick solution that worked for the
particular problem we had. As we're starting to extend this to a wider
analysis, it's becoming increasingly obvious that we need a more elegant
solution that can generalise to different statistics and interventions.

## What we have:

A number of pretrained custom models, plus models available in torch hub, etc.

## What we need:

Given a pretrained model of arbitrary architecture

-   hooking relevant layers (which layers we want to hook will depend on
    the task: it could be the activations following all convolutions,
    the last convolution or just arbitrary layer(s) --- see below) to
    extract:

    -   gradients

    -   weights

    -   ~~activations~~ layer outputs

-   computing statistics or performing operations on

    -   ~~activations~~ layer outputs

    -   weights

    -   ~~activations~~ layer outputs x gradients (up to the user to
        ensure that they are using activations)

    -   mean and std of feature maps

    -   values spectrum of feature maps

    -   rank of weights matrices per output channel

-   identifying specific types of layers and hooking the nth of its type
    (e.g. conv layers, linear layers, relu layers, etc.)

-   recording and nicely storing analysis data

-   running interventions

    -   modifying weight values (with the option to freeze the modified
        weights only)

    -   fixing the value of specific feature maps (e.g. setting the
        value of a specific feature map to its mean value across
        training)
