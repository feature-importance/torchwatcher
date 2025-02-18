from .node_types import *
from .node_selector import *

# Constant node selectors:
is_activation: Activations.is_activation
"""
A node selector which is true for all nodes commonly used as activation
functions. This isn't exact, but contains both modular and functional variants
of the following:
 - ELU
 - Hardshrink
 - Hardsigmoid
 - Hardtanh
 - Hardswish
 - LeakyReLU
 - LogSigmoid
 - PReLU
 - ReLU
 - ReLU6
 - RReLU
 - SELU
 - CELU
 - GELU
 - Sigmoid
 - SiLU
 - Mish
 - Softplus
 - Softshrink
 - Softsign
 - Tanh
 - Tanhshrink
 - Threshold
 - GLU
 - Softmin
 - Softmax
 - Softmax2d
 - LogSoftmax
 - AdaptiveLogSoftmaxWithLoss

For more control over selection of activations, combine with other node
selectors and/or use ``is_of`` to specify certain activations.

Note: MultiheadAttention is not included in the list of activation functions.
"""
# Aliases:
is_act, is_activ = Activations.is_activation, Activations.is_activation

is_conv: Convolutions.is_convolution
"""
A node selector which is true for all nodes performing convolution. This
includes convolutions from modules (``torch.nn.Conv2d`` and similar) as well as
functional calls (such as ``torch.nn.functional.conv2d``).
"""
# Aliases:
is_convolution, is_convolutional = Convolutions.is_convolution, Convolutions.is_convolution

is_pooling: Pooling.is_pooling
"""
A node selector which is true for all nodes performing pooling. This includes
both regular kernel-based pooling and adaptive pooling, but excludes
``interpolate`` and similar.
"""
# Aliases:
is_pool = Pooling.is_pooling

is_padding: Padding.is_padding
"""
A node selector which is true for all nodes performing padding, excluding
convolution layers with padding internal.
"""
# Aliases:
is_pad = Padding.is_padding

is_norm: Normalization.is_normalization
"""
A node selector which is true for all nodes performing normalization.
"""
# Aliases:
is_normalization, is_normalisation = Normalization.is_normalization, Normalization.is_normalization

is_recurrent: Recurrent.is_recurrent
"""
A node selector which is true for all primitive recurrent nodes. This excludes
nodes where recurrence is manually added by repeatedly calling the node -
symbolic tracing cannot handle such scenarios.
"""

is_transformer: Transformer.is_transformer
"""
A node selector which is true for all Transformer (and Transformer-related)
nodes.
"""

is_linear: Linear.is_linear
"""
A node selector which is true for all linear nodes. This includes identity,
linear, and bilinear layers, but does not include matrix multiplication.
"""
# Aliases:
is_lin = Linear.is_linear

is_dropout: Dropout.is_dropout
"""
A node selector which is true for all dropout nodes.
"""
# Aliases:
is_drop = Dropout.is_dropout

is_sparse: Sparse.is_sparse
"""
A node selector which is true for all sparse computation nodes. At the moment
this only includes embedding, temporarily allowing the alias ``is_embedding``.
"""
# Aliases:
is_embedding, is_embed = Sparse.is_sparse, Sparse.is_sparse

is_distance: Distance.is_distance
"""
A node selector which is true for all distance calculation nodes.
"""
# Aliases:
is_dist = Distance.is_distance

is_loss: Loss.is_loss
"""
A node selector which is true for all loss nodes.
"""

is_vision: Vision.is_vision
"""
A node selector which is true for all vision-related nodes. Specifically this
includes pixel (un)shuffling and upsampling.
"""
# Aliases:
is_vis, is_visual = Vision.is_vision, Vision.is_vision

is_shuffle: Shuffle.is_shuffle
"""
A node selector which is true for all shuffling nodes.
"""

is_utility: Utility.is_utility
"""
A node selector which is true for all utility nodes.
"""
# Aliases:
is_util = Utility.is_utility
