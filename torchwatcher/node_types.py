################################################################################
# firewatcher/src/firewatcher/graph/node_types.py
#
# Jay Bear
# Vision, Learning, and Control,
# University of Southampton
# 2024
#
# Contains collections of node types.

import torch

ACTIVATIONS = {
  "ELU": [
    getattr(torch.nn,            "ELU",  None),
    getattr(torch.nn.functional, "elu",  None),
    getattr(torch.nn.functional, "elu_", None)
  ],
  "Hardshrink": [
    getattr(torch,               "hardshrink", None),
    getattr(torch.nn,            "Hardshrink", None),
    getattr(torch.nn.functional, "hardshrink", None)
  ],
  "Hardsigmoid": [
    getattr(torch.nn,            "Hardsigmoid", None),
    getattr(torch.nn.functional, "hardsigmoid", None)
  ],
  "Hardtanh": [
    getattr(torch.nn,            "Hardtanh",  None),
    getattr(torch.nn.functional, "hardtanh",  None),
    getattr(torch.nn.functional, "hardtanh_", None)
  ],
  "Hardswish": [
    getattr(torch.nn,            "Hardswish", None),
    getattr(torch.nn.functional, "hardswish", None)
  ],
  "LeakyReLU": [
    getattr(torch.nn,            "LeakyReLU",   None),
    getattr(torch.nn.functional, "leaky_relu",  None),
    getattr(torch.nn.functional, "leaky_relu_", None)
  ],
  "LogSigmoid": [
    getattr(torch.nn,            "LogSigmoid", None),
    getattr(torch.nn.functional, "logsigmoid", None)
  ],
  "PReLU": [
    getattr(torch,               "prelu", None),
    getattr(torch.nn,            "PReLU", None),
    getattr(torch.nn.functional, "prelu", None) #_prelu_kernel?
  ],
  "ReLU": [
    getattr(torch,               "relu",  None),
    getattr(torch,               "relu_", None),
    getattr(torch.nn,            "ReLU",  None),
    getattr(torch.nn.functional, "relu",  None),
    getattr(torch.nn.functional, "relu_", None)
  ],
  "ReLU6": [
    getattr(torch.nn,            "ReLU6", None),
    getattr(torch.nn.functional, "relu6", None)
  ],
  "RReLU": [
    getattr(torch,               "rrelu",  None),
    getattr(torch,               "rrelu_", None),
    getattr(torch.nn,            "RReLU",  None),
    getattr(torch.nn.functional, "rrelu",  None),
    getattr(torch.nn.functional, "rrelu_", None)
  ],
  "SELU": [
    getattr(torch,               "selu",  None),
    getattr(torch,               "selu_", None),
    getattr(torch.nn,            "SELU",  None),
    getattr(torch.nn.functional, "selu",  None),
    getattr(torch.nn.functional, "selu_", None)
  ],
  "CELU": [
    getattr(torch,               "celu",  None),
    getattr(torch,               "celu_", None),
    getattr(torch.nn,            "CELU",  None),
    getattr(torch.nn.functional, "celu",  None),
    getattr(torch.nn.functional, "celu_", None)
  ],
  "GELU": [
    getattr(torch.nn,            "GELU", None),
    getattr(torch.nn.functional, "gelu", None)
  ],
  "Sigmoid": [
    getattr(torch,               "sigmoid",  None),
    getattr(torch,               "sigmoid_", None),
    getattr(torch.nn,            "Sigmoid",  None),
    getattr(torch.nn.functional, "sigmoid",  None)
  ],
  "SiLU": [
    getattr(torch.nn,            "SiLU", None),
    getattr(torch.nn.functional, "silu", None)
  ],
  "Mish": [
    getattr(torch.nn,            "Mish", None),
    getattr(torch.nn.functional, "mish", None)
  ],
  "Softplus": [
    getattr(torch.nn,            "Softplus", None),
    getattr(torch.nn.functional, "softplus", None)
  ],
  "Softshrink": [
    getattr(torch.nn,            "Softshrink", None),
    getattr(torch.nn.functional, "softshrink", None)
  ],
  "Softsign": [
    getattr(torch.nn,            "Softsign", None),
    getattr(torch.nn.functional, "softsign", None)
  ],
  "Tanh": [
    getattr(torch,               "tanh",  None),
    getattr(torch,               "tanh_", None),
    getattr(torch.nn,            "Tanh",  None),
    getattr(torch.nn.functional, "tanh",  None)
  ],
  "Tanhshrink": [
    getattr(torch.nn,            "Tanhshrink", None),
    getattr(torch.nn.functional, "tanhshrink", None)
  ],
  "Threshold": [
    getattr(torch,               "threshold",  None),
    getattr(torch,               "threshold_", None),
    getattr(torch.nn,            "Threshold",  None),
    getattr(torch.nn.functional, "threshold",  None)
  ],
  "GLU": [
    getattr(torch.nn,            "GLU", None),
    getattr(torch.nn.functional, "glu", None)
  ],
  "Softmin": [
    getattr(torch.nn,            "Softmin", None),
    getattr(torch.nn.functional, "softmin", None)
  ],
  "Softmax": [
    getattr(torch,               "softmax", None),
    getattr(torch.nn,            "Softmax", None),
    getattr(torch.nn.functional, "softmax", None)
  ],
  "Softmax2d": [
    getattr(torch.nn, "Softmax2d", None)
  ],
  "LogSoftmax": [
    getattr(torch,               "log_softmax", None),
    getattr(torch.nn,            "LogSoftmax",  None),
    getattr(torch.nn.functional, "log_softmax", None)
  ],
  "AdaptiveLogSoftmaxWithLoss": [
    getattr(torch.nn, "AdaptiveLogSoftmaxWithLoss", None)
  ]
}

CONVOLUTIONS = {
  "Conv1d": [
    getattr(torch,               "conv1d",     None),
    getattr(torch.nn,            "Conv1d",     None),
    getattr(torch.nn,            "LazyConv1d", None),
    getattr(torch.nn.functional, "conv1d",     None)
  ],
  "Conv2d": [
    getattr(torch,               "conv2d",     None),
    getattr(torch.nn,            "Conv2d",     None),
    getattr(torch.nn,            "LazyConv2d", None),
    getattr(torch.nn.functional, "conv2d",     None)
  ],
  "Conv3d": [
    getattr(torch,               "conv3d",     None),
    getattr(torch.nn,            "Conv3d",     None),
    getattr(torch.nn,            "LazyConv3d", None),
    getattr(torch.nn.functional, "conv3d",     None)
  ],
  "ConvTranspose1d": [
    getattr(torch,               "conv_transpose1d",    None),
    getattr(torch.nn,            "ConvTranspose1d",     None),
    getattr(torch.nn,            "LazyConvTranspose1d", None),
    getattr(torch.nn.functional, "conv_transpose1d",    None)
  ],
  "ConvTranspose2d": [
    getattr(torch,               "conv_transpose2d",    None),
    getattr(torch.nn,            "ConvTranspose2d",     None),
    getattr(torch.nn,            "LazyConvTranspose2d", None),
    getattr(torch.nn.functional, "conv_transpose2d",    None)
  ],
  "ConvTranspose3d": [
    getattr(torch,               "conv_transpose3d",    None),
    getattr(torch.nn,            "ConvTranspose3d",     None),
    getattr(torch.nn,            "LazyConvTranspose3d", None),
    getattr(torch.nn.functional, "conv_transpose3d",    None)
  ],
  "Unfold": [
    getattr(torch.nn,            "Unfold", None),
    getattr(torch.nn.functional, "unfold", None)
  ],
  "Fold": [
    getattr(torch.nn,            "Fold", None),
    getattr(torch.nn.functional, "fold", None)
  ]
}

POOLINGS = {
  "MaxPool1d": [
    getattr(torch,               "max_pool1d",              None),
    getattr(torch,               "max_pool1d_with_indices", None),
    getattr(torch,               "quantized_max_pool1d",    None),
    getattr(torch.nn,            "MaxPool1d",               None),
    getattr(torch.nn.functional, "max_pool1d",              None),
    getattr(torch.nn.functional, "max_pool1d_with_indices", None)
  ],
  "MaxPool2d": [
    getattr(torch,               "max_pool2d",              None),
    getattr(torch,               "quantized_max_pool2d",    None),
    getattr(torch.nn,            "MaxPool2d",               None),
    getattr(torch.nn.functional, "max_pool2d",              None),
    getattr(torch.nn.functional, "max_pool2d_with_indices", None)
  ],
  "MaxPool3d": [
    getattr(torch,               "max_pool3d",              None),
    getattr(torch,               "quantized_max_pool3d",    None),
    getattr(torch.nn,            "MaxPool3d",               None),
    getattr(torch.nn.functional, "max_pool3d",              None),
    getattr(torch.nn.functional, "max_pool3d_with_indices", None)
  ],
  "MaxUnpool1d": [
    getattr(torch.nn,            "MaxUnpool1d",  None),
    getattr(torch.nn.functional, "max_unpool1d", None)
  ],
  "MaxUnpool2d": [
    getattr(torch.nn,            "MaxUnpool2d",  None),
    getattr(torch.nn.functional, "max_unpool2d", None)
  ],
  "MaxUnpool3d": [
    getattr(torch.nn,            "MaxUnpool3d",  None),
    getattr(torch.nn.functional, "max_unpool3d", None)
  ],
  "AvgPool1d": [
    getattr(torch,               "avg_pool1d", None),
    getattr(torch.nn,            "AvgPool1d",  None),
    getattr(torch.nn.functional, "avg_pool1d", None)
  ],
  "AvgPool2d": [
    getattr(torch.nn,            "AvgPool2d",  None),
    getattr(torch.nn.functional, "avg_pool2d", None)
  ],
  "AvgPool3d": [
    getattr(torch.nn,            "AvgPool3d",  None),
    getattr(torch.nn.functional, "avg_pool3d", None)
  ],
  "FractionalMaxPool2d": [
    getattr(torch.nn,            "FractionalMaxPool2d",                None),
    getattr(torch.nn.functional, "fractional_max_pool2d",              None),
    getattr(torch.nn.functional, "fractional_max_pool2d_with_indices", None)
  ],
  "FractionalMaxPool3d": [
    getattr(torch.nn,            "FractionalMaxPool3d",                None),
    getattr(torch.nn.functional, "fractional_max_pool3d",              None),
    getattr(torch.nn.functional, "fractional_max_pool3d_with_indices", None)
  ],
  "LPPool1d": [
    getattr(torch.nn,            "LPPool1d",  None),
    getattr(torch.nn.functional, "lp_pool1d", None)
  ],
  "LPPool2d": [
    getattr(torch.nn,            "LPPool2d",  None),
    getattr(torch.nn.functional, "lp_pool2d", None)
  ],
  "LPPool3d": [
    getattr(torch.nn,            "LPPool3d",  None),
    getattr(torch.nn.functional, "lp_pool3d", None)
  ],
  "AdaptiveMaxPool1d": [
    getattr(torch,               "adaptive_max_pool1d",              None),
    getattr(torch.nn,            "AdaptiveMaxPool1d",                None),
    getattr(torch.nn.functional, "adaptive_max_pool1d",              None),
    getattr(torch.nn.functional, "adaptive_max_pool1d_with_indices", None)
  ],
  "AdaptiveMaxPool2d": [
    getattr(torch.nn,            "AdaptiveMaxPool2d",                None),
    getattr(torch.nn.functional, "adaptive_max_pool2d",              None),
    getattr(torch.nn.functional, "adaptive_max_pool2d_with_indices", None)
  ],
  "AdaptiveMaxPool3d": [
    getattr(torch.nn,            "AdaptiveMaxPool3d",                None),
    getattr(torch.nn.functional, "adaptive_max_pool3d",              None),
    getattr(torch.nn.functional, "adaptive_max_pool3d_with_indices", None)
  ],
  "AdaptiveAvgPool1d": [
    getattr(torch,               "adaptive_avg_pool1d", None),
    getattr(torch.nn,            "AdaptiveAvgPool1d",   None),
    getattr(torch.nn.functional, "adaptive_avg_pool1d", None)
  ],
  "AdaptiveAvgPool2d": [
    getattr(torch.nn,            "AdaptiveAvgPool2d",   None),
    getattr(torch.nn.functional, "adaptive_avg_pool2d", None)
  ],
  "AdaptiveAvgPool3d": [
    getattr(torch.nn,            "AdaptiveAvgPool3d",   None),
    getattr(torch.nn.functional, "adaptive_avg_pool3d", None)
  ]
}

# Putting this one under one category as they all use the same function.
PADDINGS = {
  "Pad": [
    getattr(torch.nn,            "ReflectionPad1d",  None),
    getattr(torch.nn,            "ReflectionPad2d",  None),
    getattr(torch.nn,            "ReflectionPad3d",  None),
    getattr(torch.nn,            "ReplicationPad1d", None),
    getattr(torch.nn,            "ReplicationPad2d", None),
    getattr(torch.nn,            "ReplicationPad3d", None),
    getattr(torch.nn,            "ZeroPad1d",        None),
    getattr(torch.nn,            "ZeroPad2d",        None),
    getattr(torch.nn,            "ZeroPad3d",        None),
    getattr(torch.nn,            "ConstantPad1d",    None),
    getattr(torch.nn,            "ConstantPad2d",    None),
    getattr(torch.nn,            "ConstantPad3d",    None),
    getattr(torch.nn,            "CircularPad1d",    None),
    getattr(torch.nn,            "CircularPad2d",    None),
    getattr(torch.nn,            "CircularPad3d",    None),
    getattr(torch.nn.functional, "pad",              None)
  ]
}

NORMALIZATIONS = {
  "BatchNorm": [
    getattr(torch,               "batch_norm",        None),
    getattr(torch,               "native_batch_norm", None),
    getattr(torch.nn,            "BatchNorm1d",       None),
    getattr(torch.nn,            "BatchNorm2d",       None),
    getattr(torch.nn,            "BatchNorm3d",       None),
    getattr(torch.nn,            "SyncBatchNorm",     None),
    getattr(torch.nn,            "LazyBatchNorm1d",   None),
    getattr(torch.nn,            "LazyBatchNorm2d",   None),
    getattr(torch.nn,            "LazyBatchNorm3d",   None),
    getattr(torch.nn.functional, "batch_norm",        None)
  ],
  "GroupNorm": [
    getattr(torch,               "group_norm",        None),
    getattr(torch,               "native_group_norm", None),
    getattr(torch.nn,            "GroupNorm",         None),
    getattr(torch.nn.functional, "group_norm",        None)
  ],
  "InstanceNorm": [
    getattr(torch,               "instance_norm",      None),
    getattr(torch.nn,            "InstanceNorm1d",     None),
    getattr(torch.nn,            "InstanceNorm2d",     None),
    getattr(torch.nn,            "InstanceNorm3d",     None),
    getattr(torch.nn,            "LazyInstanceNorm1d", None),
    getattr(torch.nn,            "LazyInstanceNorm2d", None),
    getattr(torch.nn,            "LazyInstanceNorm3d", None),
    getattr(torch.nn.functional, "instance_norm",      None)
  ],
  "LayerNorm": [
    getattr(torch,               "layer_norm",        None),
    getattr(torch,               "native_layer_norm", None),
    getattr(torch.nn,            "LayerNorm",         None),
    getattr(torch.nn.functional, "layer_norm",        None)
  ],
  "LocalResponseNorm": [
    getattr(torch.nn,            "LocalResponseNorm",   None),
    getattr(torch.nn.functional, "local_response_norm", None)
  ]
}

RECURRENTS = {
  "RNN": [
    getattr(torch,    "rnn_relu", None),
    getattr(torch,    "rnn_tanh", None),
    getattr(torch.nn, "RNN",      None),
    getattr(torch.nn, "RNNBase",  None)
  ],
  "LSTM": [
    getattr(torch,    "lstm",           None),
    getattr(torch,    "quantized_lstm", None),
    getattr(torch.nn, "LSTM",           None)
  ],
  "GRU": [
    getattr(torch,    "gru",           None),
    getattr(torch,    "quantized_gru", None),
    getattr(torch.nn, "GRU",           None)
  ],
  "RNNCell": [
    getattr(torch,    "rnn_relu_cell",           None),
    getattr(torch,    "rnn_tanh_cell",           None),
    getattr(torch,    "quantized_rnn_relu_cell", None),
    getattr(torch,    "quantized_rnn_tanh_cell", None),
    getattr(torch.nn, "RNNCell",                 None),
    getattr(torch.nn, "RNNCellBase",             None)
  ],
  "LSTMCell": [
    getattr(torch,    "lstm_cell",           None),
    getattr(torch,    "quantized_lstm_cell", None),
    getattr(torch.nn, "LSTMCell",            None)
  ],
  "GRUCell": [
    getattr(torch,    "gru_cell",           None),
    getattr(torch,    "quantized_gru_cell", None),
    getattr(torch.nn, "GRUCell",            None)
  ]
}

# I don't know enough about Transformers to know if these should be
# separated.
TRANSFORMERS = {
  "Transformer": [
    getattr(torch.nn, "Transformer",             None),
    getattr(torch.nn, "TransformerEncoder",      None),
    getattr(torch.nn, "TransformerDecoder",      None),
    getattr(torch.nn, "TransformerEncoderLayer", None),
    getattr(torch.nn, "TransformerDecoderLayer", None)
  ]
}

LINEARS = {
  "Identity": [
    getattr(torch.nn, "Identity", None)
  ],
  "Linear": [
    getattr(torch.nn,            "Linear",     None),
    getattr(torch.nn,            "LazyLinear", None),
    getattr(torch.nn.functional, "linear",     None)
  ],
  "Bilinear": [
    getattr(torch,               "bilinear", None),
    getattr(torch.nn,            "Bilinear", None),
    getattr(torch.nn.functional, "bilinear", None)
  ]
}

DROPOUTS = {
  "Dropout": [
    getattr(torch,               "dropout",        None),
    getattr(torch,               "dropout_",       None),
    getattr(torch,               "native_dropout", None),
    getattr(torch.nn,            "Dropout",        None),
    getattr(torch.nn.functional, "dropout",        None)
  ],
  "Dropout1d": [
    getattr(torch.nn,            "Dropout1d", None),
    getattr(torch.nn.functional, "dropout1d", None)
  ],
  "Dropout2d": [
    getattr(torch.nn,            "Dropout2d", None),
    getattr(torch.nn.functional, "dropout2d", None)
  ],
  "Dropout3d": [
    getattr(torch.nn,            "Dropout3d", None),
    getattr(torch.nn.functional, "dropout3d", None)
  ],
  "AlphaDropout": [
    getattr(torch,               "alpha_dropout",  None),
    getattr(torch,               "alpha_dropout_", None),
    getattr(torch.nn,            "AlphaDropout",   None),
    getattr(torch.nn.functional, "alpha_dropout",  None)
  ],
  "FeatureAlphaDropout": [
    getattr(torch,               "feature_dropout",        None),
    getattr(torch,               "feature_dropout_",       None),
    getattr(torch,               "feature_alpha_dropout",  None),
    getattr(torch,               "feature_alpha_dropout_", None),
    getattr(torch.nn,            "FeatureAlphaDropout",    None),
    getattr(torch.nn.functional, "feature_alpha_dropout",  None)
  ]
}

SPARSES = {
  "Embedding": [
    getattr(torch,               "embedding",         None),
    getattr(torch,               "embedding_renorm_", None),
    getattr(torch.nn,            "Embedding",         None),
    getattr(torch.nn.functional, "embedding",         None)
  ],
  "EmbeddingBag": [
    getattr(torch,               "embedding_bag", None),
    getattr(torch.nn,            "EmbeddingBag",  None),
    getattr(torch.nn.functional, "embedding_bag", None)
  ]
}

DISTANCES = {
  "CosineSimilarity": [
    getattr(torch,               "cosine_similarity", None),
    getattr(torch.nn,            "CosineSimilarity",  None),
    getattr(torch.nn.functional, "cosine_similarity", None)
  ],
  "PairwiseDistance": [
    getattr(torch,               "pairwise_distance", None),
    getattr(torch.nn,            "PairwiseDistance",  None),
    getattr(torch.nn.functional, "pairwise_distance", None)
  ]
}

LOSSES = {
  "L1Loss": [
    getattr(torch.nn,            "L1Loss",  None),
    getattr(torch.nn.functional, "l1_loss", None)
  ],
  "MSELoss": [
    getattr(torch.nn,            "MSELoss",  None),
    getattr(torch.nn.functional, "mse_loss", None)
  ],
  "CrossEntropyLoss": [
    getattr(torch.nn,            "CrossEntropyLoss", None),
    getattr(torch.nn.functional, "cross_entropy",    None)
  ],
  "CTCLoss": [
    getattr(torch,               "ctc_loss", None),
    getattr(torch.nn,            "CTCLoss",  None),
    getattr(torch.nn.functional, "ctc_loss", None)
  ],
  "NLLLoss": [
    getattr(torch.nn,            "NLLLoss",  None),
    getattr(torch.nn.functional, "nll_loss", None)
  ],
  "PoissonNLLLoss": [
    getattr(torch,               "poisson_nll_loss", None),
    getattr(torch.nn,            "PoissonNLLLoss",   None),
    getattr(torch.nn.functional, "poisson_nll_loss", None)
  ],
  "GaussianNLLLoss": [
    getattr(torch.nn,            "GaussianNLLLoss",   None),
    getattr(torch.nn.functional, "gaussian_nll_loss", None)
  ],
  "KLDivLoss": [
    getattr(torch,               "kl_div",    None),
    getattr(torch.nn,            "KLDivLoss", None),
    getattr(torch.nn.functional, "kl_div",    None)
  ],
  "BCELoss": [
    getattr(torch.nn,            "BCELoss",              None),
    getattr(torch.nn.functional, "binary_cross_entropy", None)
  ],
  "BCEWithLogitsLoss": [
    getattr(torch,               "binary_cross_entropy_with_logits", None),
    getattr(torch.nn,            "BCEWithLogitsLoss",                None),
    getattr(torch.nn.functional, "binary_cross_entropy_with_logits", None)
  ],
  "MarginRankingLoss": [
    getattr(torch,               "margin_ranking_loss", None),
    getattr(torch.nn,            "MarginRankingLoss",   None),
    getattr(torch.nn.functional, "margin_ranking_loss", None)
  ],
  "HingeEmbeddingLoss": [
    getattr(torch,               "hinge_embedding_loss", None),
    getattr(torch.nn,            "HingeEmbeddingLoss",   None),
    getattr(torch.nn.functional, "hinge_embedding_loss", None)
  ],
  "MultiLabelMarginLoss": [
    getattr(torch.nn,            "MultiLabelMarginLoss",  None),
    getattr(torch.nn.functional, "multilabel_margin_loss", None)
  ],
  "HuberLoss": [
    getattr(torch.nn,            "HuberLoss",  None),
    getattr(torch.nn.functional, "huber_loss", None)
  ],
  "SmoothL1Loss": [
    getattr(torch.nn,            "SmoothL1Loss",   None),
    getattr(torch.nn.functional, "smooth_l1_loss", None)
  ],
  "SoftMarginLoss": [
    getattr(torch.nn,            "SoftMarginLoss",   None),
    getattr(torch.nn.functional, "soft_margin_loss", None)
  ],
  "MultiLabelSoftMarginLoss": [
    getattr(torch.nn,            "MultiLabelSoftMarginLoss",    None),
    getattr(torch.nn.functional, "multilabel_soft_margin_loss", None)
  ],
  "CosineEmbeddingLoss": [
    getattr(torch,               "cosine_embedding_loss", None),
    getattr(torch.nn,            "CosineEmbeddingLoss",   None),
    getattr(torch.nn.functional, "cosine_embedding_loss", None)
  ],
  "MultiMarginLoss": [
    getattr(torch.nn,            "MultiMarginLoss",   None),
    getattr(torch.nn.functional, "multi_margin_loss", None)
  ],
  "TripletMarginLoss": [
    getattr(torch,               "triplet_margin_loss", None),
    getattr(torch.nn,            "TripletMarginLoss",   None),
    getattr(torch.nn.functional, "triplet_margin_loss", None)
  ],
  "TripletMarginWithDistanceLoss": [
    getattr(torch.nn,            "TripletMarginWithDistanceLoss",     None),
    getattr(torch.nn.functional, "triplet_margin_with_distance_loss", None)
  ]
}

VISIONS = {
  "PixelShuffle": [
    getattr(torch,               "pixel_shuffle", None),
    getattr(torch.nn,            "PixelShuffle",  None),
    getattr(torch.nn.functional, "pixel_shuffle", None)
  ],
  "PixelUnshuffle": [
    getattr(torch,               "pixel_unshuffle", None),
    getattr(torch.nn,            "PixelUnshuffle",  None),
    getattr(torch.nn.functional, "pixel_unshuffle", None)
  ],
  "Upsample": [
    getattr(torch.nn,            "Upsample", None),
    getattr(torch.nn.functional, "upsample", None)
  ],
  "UpsamplingNearest2d": [
    getattr(torch.nn,            "UpsamplingNearest2d", None),
    getattr(torch.nn.functional, "upsample_nearest",    None)
  ],
  "UpsamplingBilinear2d": [
    getattr(torch.nn,            "UpsamplingBilinear2d", None),
    getattr(torch.nn.functional, "upsample_bilinear",    None)
  ]
}

SHUFFLINGS = {
  "ChannelShuffle": [
    getattr(torch,               "channel_shuffle",        None),
    getattr(torch.nn,            "ChannelShuffle",         None),
    getattr(torch.nn.functional, "channel_shuffle",        None),
    getattr(torch.nn.functional, "native_channel_shuffle", None)
  ],
  "PixelShuffle":   VISIONS["PixelShuffle"],
  "PixelUnshuffle": VISIONS["PixelUnshuffle"]
}

UTILITIES = {
  "Flatten": [
    getattr(torch,    "flatten", None),
    getattr(torch.nn, "Flatten", None)
  ],
  "Unflatten": [
    getattr(torch,    "unflatten", None),
    getattr(torch.nn, "Unflatten", None),
  ]
}
