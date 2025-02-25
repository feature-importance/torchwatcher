"""Contains collections of node types."""
from enum import Enum
from typing import List, Any

import torch

from .node_selector import NodeSelector, _is_node_of_module
from .node_selector import _is_node_of_function, NodeState


def _is_node_of(
        # Arguments:
        nodestate: NodeState,
        structure: List[Any]
) -> bool:
    return any(
        _is_node_of_module(nodestate, f) or
        _is_node_of_function(nodestate, f) for f in structure
    )


class MultiNodeSelector(NodeSelector):
    def __init__(self, selector_functions: List[Any]):
        super().__init__(lambda x: _is_node_of(x, selector_functions))
        self.values = selector_functions


class _All:
    def __get__(self, instance, cls):
        all_mbr = list(iter(cls))[0]
        for name, member in cls.__members__.items():
            all_mbr |= member
        return all_mbr


class Activations(MultiNodeSelector, Enum):
    is_elu = [
        getattr(torch.nn, "ELU", None),
        getattr(torch.nn.functional, "elu", None),
        getattr(torch.nn.functional, "elu_", None)
    ]
    is_hardshrink = [
        getattr(torch, "hardshrink", None),
        getattr(torch.nn, "Hardshrink", None),
        getattr(torch.nn.functional, "hardshrink", None)
    ]
    is_hardsigmoid = [
        getattr(torch.nn, "Hardsigmoid", None),
        getattr(torch.nn.functional, "hardsigmoid", None)
    ]
    is_hardtanh = [
        getattr(torch.nn, "Hardtanh", None),
        getattr(torch.nn.functional, "hardtanh", None),
        getattr(torch.nn.functional, "hardtanh_", None)
    ]
    is_hardswish = [
        getattr(torch.nn, "Hardswish", None),
        getattr(torch.nn.functional, "hardswish", None)
    ]
    is_leaky_relu = [
        getattr(torch.nn, "LeakyReLU", None),
        getattr(torch.nn.functional, "leaky_relu", None),
        getattr(torch.nn.functional, "leaky_relu_", None)
    ]
    is_logsigmoid = [
        getattr(torch.nn, "LogSigmoid", None),
        getattr(torch.nn.functional, "logsigmoid", None)
    ]
    is_prelu = [
        getattr(torch, "prelu", None),
        getattr(torch.nn, "PReLU", None),
        getattr(torch.nn.functional, "prelu", None)  # _prelu_kernel?
    ]
    is_relu = [
        getattr(torch, "relu", None),
        getattr(torch, "relu_", None),
        getattr(torch.nn, "ReLU", None),
        getattr(torch.nn.functional, "relu", None),
        getattr(torch.nn.functional, "relu_", None)
    ]
    is_relu6 = [
        getattr(torch.nn, "ReLU6", None),
        getattr(torch.nn.functional, "relu6", None)
    ]
    is_rrelu = [
        getattr(torch, "rrelu", None),
        getattr(torch, "rrelu_", None),
        getattr(torch.nn, "RReLU", None),
        getattr(torch.nn.functional, "rrelu", None),
        getattr(torch.nn.functional, "rrelu_", None)
    ]
    is_selu = [
        getattr(torch, "selu", None),
        getattr(torch, "selu_", None),
        getattr(torch.nn, "SELU", None),
        getattr(torch.nn.functional, "selu", None),
        getattr(torch.nn.functional, "selu_", None)
    ]
    is_celu = [
        getattr(torch, "celu", None),
        getattr(torch, "celu_", None),
        getattr(torch.nn, "CELU", None),
        getattr(torch.nn.functional, "celu", None),
        getattr(torch.nn.functional, "celu_", None)
    ]
    is_gelu = [
        getattr(torch.nn, "GELU", None),
        getattr(torch.nn.functional, "gelu", None)
    ]
    is_sigmoid = [
        getattr(torch, "sigmoid", None),
        getattr(torch, "sigmoid_", None),
        getattr(torch.nn, "Sigmoid", None),
        getattr(torch.nn.functional, "sigmoid", None)
    ]
    is_silu = [
        getattr(torch.nn, "SiLU", None),
        getattr(torch.nn.functional, "silu", None)
    ]
    is_mish = [
        getattr(torch.nn, "Mish", None),
        getattr(torch.nn.functional, "mish", None)
    ]
    is_softplus = [
        getattr(torch.nn, "Softplus", None),
        getattr(torch.nn.functional, "softplus", None)
    ]
    is_softshrink = [
        getattr(torch.nn, "Softshrink", None),
        getattr(torch.nn.functional, "softshrink", None)
    ]
    is_softsign = [
        getattr(torch.nn, "Softsign", None),
        getattr(torch.nn.functional, "softsign", None)
    ]
    is_tanh = [
        getattr(torch, "tanh", None),
        getattr(torch, "tanh_", None),
        getattr(torch.nn, "Tanh", None),
        getattr(torch.nn.functional, "tanh", None)
    ]
    is_tanhshrink = [
        getattr(torch.nn, "Tanhshrink", None),
        getattr(torch.nn.functional, "tanhshrink", None)
    ]
    is_threshold = [
        getattr(torch, "threshold", None),
        getattr(torch, "threshold_", None),
        getattr(torch.nn, "Threshold", None),
        getattr(torch.nn.functional, "threshold", None)
    ]
    is_glu = [
        getattr(torch.nn, "GLU", None),
        getattr(torch.nn.functional, "glu", None)
    ]
    is_softmin = [
        getattr(torch.nn, "Softmin", None),
        getattr(torch.nn.functional, "softmin", None)
    ]
    is_softmax = [
        getattr(torch, "softmax", None),
        getattr(torch.nn, "Softmax", None),
        getattr(torch.nn.functional, "softmax", None)
    ]
    is_softmax2d = [
        getattr(torch.nn, "Softmax2d", None)
    ]
    is_log_softmax = [
        getattr(torch, "log_softmax", None),
        getattr(torch.nn, "LogSoftmax", None),
        getattr(torch.nn.functional, "log_softmax", None)
    ]
    is_adaptive_log_softmax_with_loss = [
        getattr(torch.nn, "AdaptiveLogSoftmaxWithLoss", None)
    ]
    is_activation = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


class Convolutions(MultiNodeSelector, Enum):
    is_conv1d = [
        getattr(torch, "conv1d", None),
        getattr(torch.nn, "Conv1d", None),
        getattr(torch.nn, "LazyConv1d", None),
        getattr(torch.nn.functional, "conv1d", None)
    ]
    is_conv2d = [
        getattr(torch, "conv2d", None),
        getattr(torch.nn, "Conv2d", None),
        getattr(torch.nn, "LazyConv2d", None),
        getattr(torch.nn.functional, "conv2d", None)
    ]
    is_conv3d = [
        getattr(torch, "conv3d", None),
        getattr(torch.nn, "Conv3d", None),
        getattr(torch.nn, "LazyConv3d", None),
        getattr(torch.nn.functional, "conv3d", None)
    ]
    is_conv_transpose1d = [
        getattr(torch, "conv_transpose1d", None),
        getattr(torch.nn, "ConvTranspose1d", None),
        getattr(torch.nn, "LazyConvTranspose1d", None),
        getattr(torch.nn.functional, "conv_transpose1d", None)
    ]
    is_conv_transpose2d = [
        getattr(torch, "conv_transpose2d", None),
        getattr(torch.nn, "ConvTranspose2d", None),
        getattr(torch.nn, "LazyConvTranspose2d", None),
        getattr(torch.nn.functional, "conv_transpose2d", None)
    ]
    is_conv_transpose3d = [
        getattr(torch, "conv_transpose3d", None),
        getattr(torch.nn, "ConvTranspose3d", None),
        getattr(torch.nn, "LazyConvTranspose3d", None),
        getattr(torch.nn.functional, "conv_transpose3d", None)
    ]
    is_unfold = [
        getattr(torch.nn, "Unfold", None),
        getattr(torch.nn.functional, "unfold", None)
    ]
    is_fold = [
        getattr(torch.nn, "Fold", None),
        getattr(torch.nn.functional, "fold", None)
    ]
    is_convolution = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


class Pooling(MultiNodeSelector, Enum):
    is_max_pool1d = [
        getattr(torch, "max_pool1d", None),
        getattr(torch, "max_pool1d_with_indices", None),
        getattr(torch, "quantized_max_pool1d", None),
        getattr(torch.nn, "MaxPool1d", None),
        getattr(torch.nn.functional, "max_pool1d", None),
        getattr(torch.nn.functional, "max_pool1d_with_indices", None)
    ]
    is_max_pool2d = [
        getattr(torch, "max_pool2d", None),
        getattr(torch, "quantized_max_pool2d", None),
        getattr(torch.nn, "MaxPool2d", None),
        getattr(torch.nn.functional, "max_pool2d", None),
        getattr(torch.nn.functional, "max_pool2d_with_indices", None)
    ]
    is_max_pool3d = [
        getattr(torch, "max_pool3d", None),
        getattr(torch, "quantized_max_pool3d", None),
        getattr(torch.nn, "MaxPool3d", None),
        getattr(torch.nn.functional, "max_pool3d", None),
        getattr(torch.nn.functional, "max_pool3d_with_indices", None)
    ]
    is_max_unpool1d = [
        getattr(torch.nn, "MaxUnpool1d", None),
        getattr(torch.nn.functional, "max_unpool1d", None)
    ]
    is_max_unpool2d = [
        getattr(torch.nn, "MaxUnpool2d", None),
        getattr(torch.nn.functional, "max_unpool2d", None)
    ]
    is_max_unpool3d = [
        getattr(torch.nn, "MaxUnpool3d", None),
        getattr(torch.nn.functional, "max_unpool3d", None)
    ]
    is_avg_pool1d = [
        getattr(torch, "avg_pool1d", None),
        getattr(torch.nn, "AvgPool1d", None),
        getattr(torch.nn.functional, "avg_pool1d", None)
    ]
    is_avg_pool2d = [
        getattr(torch.nn, "AvgPool2d", None),
        getattr(torch.nn.functional, "avg_pool2d", None)
    ]
    is_avg_pool3d = [
        getattr(torch.nn, "AvgPool3d", None),
        getattr(torch.nn.functional, "avg_pool3d", None)
    ]
    is_fractional_max_pool2d = [
        getattr(torch.nn, "FractionalMaxPool2d", None),
        getattr(torch.nn.functional, "fractional_max_pool2d", None),
        getattr(torch.nn.functional, "fractional_max_pool2d_with_indices", None)
    ]
    is_fractional_max_pool3d = [
        getattr(torch.nn, "FractionalMaxPool3d", None),
        getattr(torch.nn.functional, "fractional_max_pool3d", None),
        getattr(torch.nn.functional, "fractional_max_pool3d_with_indices", None)
    ]
    is_lp_pool1d = [
        getattr(torch.nn, "LPPool1d", None),
        getattr(torch.nn.functional, "lp_pool1d", None)
    ]
    is_lp_pool2d = [
        getattr(torch.nn, "LPPool2d", None),
        getattr(torch.nn.functional, "lp_pool2d", None)
    ]
    is_lp_pool3d = [
        getattr(torch.nn, "LPPool3d", None),
        getattr(torch.nn.functional, "lp_pool3d", None)
    ]
    is_adaptive_max_pool1d = [
        getattr(torch, "adaptive_max_pool1d", None),
        getattr(torch.nn, "AdaptiveMaxPool1d", None),
        getattr(torch.nn.functional, "adaptive_max_pool1d", None),
        getattr(torch.nn.functional, "adaptive_max_pool1d_with_indices", None)
    ]
    is_adaptive_max_pool2d = [
        getattr(torch.nn, "AdaptiveMaxPool2d", None),
        getattr(torch.nn.functional, "adaptive_max_pool2d", None),
        getattr(torch.nn.functional, "adaptive_max_pool2d_with_indices", None)
    ]
    is_adaptive_max_pool3d = [
        getattr(torch.nn, "AdaptiveMaxPool3d", None),
        getattr(torch.nn.functional, "adaptive_max_pool3d", None),
        getattr(torch.nn.functional, "adaptive_max_pool3d_with_indices", None)
    ]
    is_adaptive_avg_pool1d = [
        getattr(torch, "adaptive_avg_pool1d", None),
        getattr(torch.nn, "AdaptiveAvgPool1d", None),
        getattr(torch.nn.functional, "adaptive_avg_pool1d", None)
    ]
    is_adaptive_avg_pool2d = [
        getattr(torch.nn, "AdaptiveAvgPool2d", None),
        getattr(torch.nn.functional, "adaptive_avg_pool2d", None)
    ]
    is_adaptive_avg_pool3d = [
        getattr(torch.nn, "AdaptiveAvgPool3d", None),
        getattr(torch.nn.functional, "adaptive_avg_pool3d", None)
    ]
    is_pooling = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


# Putting this one under one category as they all use the same function.
class Padding(MultiNodeSelector, Enum):
    is_pad = [
        getattr(torch.nn, "ReflectionPad1d", None),
        getattr(torch.nn, "ReflectionPad2d", None),
        getattr(torch.nn, "ReflectionPad3d", None),
        getattr(torch.nn, "ReplicationPad1d", None),
        getattr(torch.nn, "ReplicationPad2d", None),
        getattr(torch.nn, "ReplicationPad3d", None),
        getattr(torch.nn, "ZeroPad1d", None),
        getattr(torch.nn, "ZeroPad2d", None),
        getattr(torch.nn, "ZeroPad3d", None),
        getattr(torch.nn, "ConstantPad1d", None),
        getattr(torch.nn, "ConstantPad2d", None),
        getattr(torch.nn, "ConstantPad3d", None),
        getattr(torch.nn, "CircularPad1d", None),
        getattr(torch.nn, "CircularPad2d", None),
        getattr(torch.nn, "CircularPad3d", None),
        getattr(torch.nn.functional, "pad", None)
    ]
    is_padding = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


class Normalization(MultiNodeSelector, Enum):
    is_batch_norm = [
        getattr(torch, "batch_norm", None),
        getattr(torch, "native_batch_norm", None),
        getattr(torch.nn, "BatchNorm1d", None),
        getattr(torch.nn, "BatchNorm2d", None),
        getattr(torch.nn, "BatchNorm3d", None),
        getattr(torch.nn, "SyncBatchNorm", None),
        getattr(torch.nn, "LazyBatchNorm1d", None),
        getattr(torch.nn, "LazyBatchNorm2d", None),
        getattr(torch.nn, "LazyBatchNorm3d", None),
        getattr(torch.nn.functional, "batch_norm", None)
    ]
    is_group_norm = [
        getattr(torch, "group_norm", None),
        getattr(torch, "native_group_norm", None),
        getattr(torch.nn, "GroupNorm", None),
        getattr(torch.nn.functional, "group_norm", None)
    ]
    is_instance_norm = [
        getattr(torch, "instance_norm", None),
        getattr(torch.nn, "InstanceNorm1d", None),
        getattr(torch.nn, "InstanceNorm2d", None),
        getattr(torch.nn, "InstanceNorm3d", None),
        getattr(torch.nn, "LazyInstanceNorm1d", None),
        getattr(torch.nn, "LazyInstanceNorm2d", None),
        getattr(torch.nn, "LazyInstanceNorm3d", None),
        getattr(torch.nn.functional, "instance_norm", None)
    ]
    is_layer_norm = [
        getattr(torch, "layer_norm", None),
        getattr(torch, "native_layer_norm", None),
        getattr(torch.nn, "LayerNorm", None),
        getattr(torch.nn.functional, "layer_norm", None)
    ]
    is_local_response_norm = [
        getattr(torch.nn, "LocalResponseNorm", None),
        getattr(torch.nn.functional, "local_response_norm", None)
    ]
    is_normalization = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


class Recurrent(MultiNodeSelector, Enum):
    is_rnn = [
        getattr(torch, "rnn_relu", None),
        getattr(torch, "rnn_tanh", None),
        getattr(torch.nn, "RNN", None),
        getattr(torch.nn, "RNNBase", None)
    ]
    is_lstm = [
        getattr(torch, "lstm", None),
        getattr(torch, "quantized_lstm", None),
        getattr(torch.nn, "LSTM", None)
    ]
    is_gru = [
        getattr(torch, "gru", None),
        getattr(torch, "quantized_gru", None),
        getattr(torch.nn, "GRU", None)
    ]
    is_rnn_cell = [
        getattr(torch, "rnn_relu_cell", None),
        getattr(torch, "rnn_tanh_cell", None),
        getattr(torch, "quantized_rnn_relu_cell", None),
        getattr(torch, "quantized_rnn_tanh_cell", None),
        getattr(torch.nn, "RNNCell", None),
        getattr(torch.nn, "RNNCellBase", None)
    ]
    is_lstm_cell = [
        getattr(torch, "lstm_cell", None),
        getattr(torch, "quantized_lstm_cell", None),
        getattr(torch.nn, "LSTMCell", None)
    ]
    is_gru_cell = [
        getattr(torch, "gru_cell", None),
        getattr(torch, "quantized_gru_cell", None),
        getattr(torch.nn, "GRUCell", None)
    ]
    is_recurrent = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


# This might need rejigging depending on use cases
class Transformer(MultiNodeSelector, Enum):
    is__transformer_ = [
        getattr(torch.nn, "Transformer", None)
    ]
    is_transformer_encoder = [
        getattr(torch.nn, "TransformerEncoder", None),
        getattr(torch.nn, "TransformerEncoderLayer", None)
    ]
    is_transformer_decoder = [
        getattr(torch.nn, "TransformerDecoder", None),
        getattr(torch.nn, "TransformerDecoderLayer", None)
    ]
    is_transformer = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


class Linear(MultiNodeSelector, Enum):
    is_identity = [
        getattr(torch.nn, "Identity", None)
    ]
    is__linear_ = [
        getattr(torch.nn, "Linear", None),
        getattr(torch.nn, "LazyLinear", None),
        getattr(torch.nn.functional, "linear", None)
    ],
    is_bilinear = [
        getattr(torch, "bilinear", None),
        getattr(torch.nn, "Bilinear", None),
        getattr(torch.nn.functional, "bilinear", None)
    ]
    is_linear = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


class Dropout(MultiNodeSelector, Enum):
    is__dropout_ = [
        getattr(torch, "dropout", None),
        getattr(torch, "dropout_", None),
        getattr(torch, "native_dropout", None),
        getattr(torch.nn, "Dropout", None),
        getattr(torch.nn.functional, "dropout", None)
    ]
    is_dropout1d = [
        getattr(torch.nn, "Dropout1d", None),
        getattr(torch.nn.functional, "dropout1d", None)
    ]
    is_dropout2d = [
        getattr(torch.nn, "Dropout2d", None),
        getattr(torch.nn.functional, "dropout2d", None)
    ]
    is_dropout3d = [
        getattr(torch.nn, "Dropout3d", None),
        getattr(torch.nn.functional, "dropout3d", None)
    ]
    is_alphadropout = [
        getattr(torch, "alpha_dropout", None),
        getattr(torch, "alpha_dropout_", None),
        getattr(torch.nn, "AlphaDropout", None),
        getattr(torch.nn.functional, "alpha_dropout", None)
    ]
    is_featurealphadropout = [
        getattr(torch, "feature_dropout", None),
        getattr(torch, "feature_dropout_", None),
        getattr(torch, "feature_alpha_dropout", None),
        getattr(torch, "feature_alpha_dropout_", None),
        getattr(torch.nn, "FeatureAlphaDropout", None),
        getattr(torch.nn.functional, "feature_alpha_dropout", None)
    ]
    is_dropout = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


# rename?
class Sparse(MultiNodeSelector, Enum):
    is_embedding = [
        getattr(torch, "embedding", None),
        getattr(torch, "embedding_renorm_", None),
        getattr(torch.nn, "Embedding", None),
        getattr(torch.nn.functional, "embedding", None)
    ]
    is_embedding_bag = [
        getattr(torch, "embedding_bag", None),
        getattr(torch.nn, "EmbeddingBag", None),
        getattr(torch.nn.functional, "embedding_bag", None)
    ]
    is_sparse = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


class Distance(MultiNodeSelector, Enum):
    is_cosine_similarity = [
        getattr(torch, "cosine_similarity", None),
        getattr(torch.nn, "CosineSimilarity", None),
        getattr(torch.nn.functional, "cosine_similarity", None)
    ]
    is_pairwise_distance = [
        getattr(torch, "pairwise_distance", None),
        getattr(torch.nn, "PairwiseDistance", None),
        getattr(torch.nn.functional, "pairwise_distance", None)
    ]
    is_distance = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


class Loss(MultiNodeSelector, Enum):
    is_l1_loss = [
        getattr(torch.nn, "L1Loss", None),
        getattr(torch.nn.functional, "l1_loss", None)
    ]
    is_mse_loss = [
        getattr(torch.nn, "MSELoss", None),
        getattr(torch.nn.functional, "mse_loss", None)
    ]
    is_cross_entropy = [
        getattr(torch.nn, "CrossEntropyLoss", None),
        getattr(torch.nn.functional, "cross_entropy", None)
    ]
    is_ctc_loss = [
        getattr(torch, "ctc_loss", None),
        getattr(torch.nn, "CTCLoss", None),
        getattr(torch.nn.functional, "ctc_loss", None)
    ]
    is_nll_loss = [
        getattr(torch.nn, "NLLLoss", None),
        getattr(torch.nn.functional, "nll_loss", None)
    ]
    is_poisson_nll_loss = [
        getattr(torch, "poisson_nll_loss", None),
        getattr(torch.nn, "PoissonNLLLoss", None),
        getattr(torch.nn.functional, "poisson_nll_loss", None)
    ]
    is_gaussian_nll_loss = [
        getattr(torch.nn, "GaussianNLLLoss", None),
        getattr(torch.nn.functional, "gaussian_nll_loss", None)
    ]
    is_kl_div = [
        getattr(torch, "kl_div", None),
        getattr(torch.nn, "KLDivLoss", None),
        getattr(torch.nn.functional, "kl_div", None)
    ]
    is_binary_cross_entropy = [
        getattr(torch.nn, "BCELoss", None),
        getattr(torch.nn.functional, "binary_cross_entropy", None)
    ]
    is_binary_cross_entropy_with_logits = [
        getattr(torch, "binary_cross_entropy_with_logits", None),
        getattr(torch.nn, "BCEWithLogitsLoss", None),
        getattr(torch.nn.functional, "binary_cross_entropy_with_logits", None)
    ]
    is_margin_ranking_loss = [
        getattr(torch, "margin_ranking_loss", None),
        getattr(torch.nn, "MarginRankingLoss", None),
        getattr(torch.nn.functional, "margin_ranking_loss", None)
    ]
    is_hinge_embedding_loss = [
        getattr(torch, "hinge_embedding_loss", None),
        getattr(torch.nn, "HingeEmbeddingLoss", None),
        getattr(torch.nn.functional, "hinge_embedding_loss", None)
    ]
    is_multilabel_margin_loss = [
        getattr(torch.nn, "MultiLabelMarginLoss", None),
        getattr(torch.nn.functional, "multilabel_margin_loss", None)
    ]
    is_huber_loss = [
        getattr(torch.nn, "HuberLoss", None),
        getattr(torch.nn.functional, "huber_loss", None)
    ]
    is_smooth_l1_loss = [
        getattr(torch.nn, "SmoothL1Loss", None),
        getattr(torch.nn.functional, "smooth_l1_loss", None)
    ]
    is_soft_margin_loss = [
        getattr(torch.nn, "SoftMarginLoss", None),
        getattr(torch.nn.functional, "soft_margin_loss", None)
    ]
    is_multilabel_soft_margin_loss = [
        getattr(torch.nn, "MultiLabelSoftMarginLoss", None),
        getattr(torch.nn.functional, "multilabel_soft_margin_loss", None)
    ]
    is_cosine_embedding_loss = [
        getattr(torch, "cosine_embedding_loss", None),
        getattr(torch.nn, "CosineEmbeddingLoss", None),
        getattr(torch.nn.functional, "cosine_embedding_loss", None)
    ]
    is_multi_margin_loss = [
        getattr(torch.nn, "MultiMarginLoss", None),
        getattr(torch.nn.functional, "multi_margin_loss", None)
    ]
    is_triplet_margin_loss = [
        getattr(torch, "triplet_margin_loss", None),
        getattr(torch.nn, "TripletMarginLoss", None),
        getattr(torch.nn.functional, "triplet_margin_loss", None)
    ]
    is_triplet_margin_with_distance_loss = [
        getattr(torch.nn, "TripletMarginWithDistanceLoss", None),
        getattr(torch.nn.functional, "triplet_margin_with_distance_loss", None)
    ]
    is_loss = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


class Vision(MultiNodeSelector, Enum):
    is_pixel_shuffle = [
        getattr(torch, "pixel_shuffle", None),
        getattr(torch.nn, "PixelShuffle", None),
        getattr(torch.nn.functional, "pixel_shuffle", None)
    ]
    is_pixel_unshuffle = [
        getattr(torch, "pixel_unshuffle", None),
        getattr(torch.nn, "PixelUnshuffle", None),
        getattr(torch.nn.functional, "pixel_unshuffle", None)
    ]
    is_upsample = [
        getattr(torch.nn, "Upsample", None),
        getattr(torch.nn.functional, "upsample", None)
    ]
    is_upsampling_nearest = [
        getattr(torch.nn, "UpsamplingNearest2d", None),
        getattr(torch.nn.functional, "upsample_nearest", None)
    ]
    is_upsampling_bilinear = [
        getattr(torch.nn, "UpsamplingBilinear2d", None),
        getattr(torch.nn.functional, "upsample_bilinear", None)
    ]
    is_vision = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


class Shuffle(MultiNodeSelector, Enum):
    is_channel_shuffle = [
        getattr(torch, "channel_shuffle", None),
        getattr(torch.nn, "ChannelShuffle", None),
        getattr(torch.nn.functional, "channel_shuffle", None),
        getattr(torch.nn.functional, "native_channel_shuffle", None)
    ],
    is_pixel_shuffle = Vision.is_pixel_shuffle
    is_pixel_unshuffle = Vision.is_pixel_unshuffle
    is_shuffle = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)


class Utility(MultiNodeSelector, Enum):
    is_flatten = [
        getattr(torch, "flatten", None),
        getattr(torch.nn, "Flatten", None)
    ]
    is_unflatten = [
        getattr(torch, "unflatten", None),
        getattr(torch.nn, "Unflatten", None),
    ]
    is_utility = _All()

    def __init__(self, value):
        Enum.__init__(self)
        MultiNodeSelector.__init__(self, value)
