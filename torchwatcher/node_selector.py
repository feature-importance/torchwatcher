################################################################################
# firewatcher/src/firewatcher/graph/node_selector.py
#
# Jay Bear
# Vision, Learning, and Control,
# University of Southampton
# 2024
#
# Functions and classes for handling the selection of certain nodes in the
# compute graph. This approach may seem unorthodox - but I'm a functional
# programmer at heart and this was the best approach I could think of.

from __future__ import annotations
from os import linesep
from torch.nn import Module
from torch.fx import GraphModule, Node
from typing import Any, Callable, Dict, List, Optional, Self, Tuple, Type

from node_types import *

IDENTITY_TRUE = lambda _: True
IDENTITY_FALSE = lambda _: False

NodeState = Tuple[GraphModule, Node]
SelectorFunction = Callable[[NodeState], bool]


class NodeSelector():
    """
  ``NodeSelector`` is the base class for all node selection operations. This
  allows for operator overloading, such as with:

  ``selector = (is_conv | (~is_activation & name_contains("linear")))``

  ``NodeSelector`` is primarily implemented in a similar fashion to a monadic
  parser combinator (https://en.wikipedia.org/wiki/Parser_combinator).
  Functions are of the form:

  nodestate -> bool
  """

    def __init__(
            # Arguments:
            self: Self,
            # Keyword Arguments:
            selector_function: Optional[SelectorFunction] = None
    ) -> None:
        """
    Initializes ``NodeSelector``.

    Args:
      self (Self):
        Object.
      selector_function (SelectorFunction, optional):
        The selector function of form "nodestate -> bool".
        Defaults to ``None``, which uses identity "x -> true".
    """
        # Selector function.
        assert callable(selector_function) or selector_function is None, \
            "selector_function must be callable or None, but has been given " + \
            f"a value of type '{type(selector_function)}' instead."
        self._selector_function = IDENTITY_TRUE if selector_function is None \
            else selector_function

    @property
    def fn(
            # Arguments:
            self: Self
    ) -> SelectorFunction:
        """
    Gets the selector function.

    Args:
      self (Self):
        Object.

    Returns:
      SelectorFunction:
        The selector function.
    """
        return self._selector_function

    def __and__(
            # Arguments:
            self: Self,
            other: NodeSelector
    ) -> NodeSelector:
        """
    Logical and.

    Args:
      self (Self):
        Self object.
      other (NodeSelector):
        Other object.

    Returns:
      NodeSelector:
        New selector containing logical and.
    """
        temp = lambda x: self._selector_function(x) and other.fn(x)
        return NodeSelector(temp)

    def __or__(
            # Arguments:
            self: Self,
            other: NodeSelector
    ) -> NodeSelector:
        """
    Logical or.

    Args:
      self (Self):
        Self object.
      other (NodeSelector):
        Other object.

    Returns:
      NodeSelector:
        New selector containing logical or.
    """
        temp = lambda x: self._selector_function(x) or other._selector_function(x)
        return NodeSelector(temp)

    def __xor__(
            # Arguments:
            self: Self,
            other: NodeSelector
    ) -> NodeSelector:
        """
    Logical exclusive or.

    Args:
      self (Self):
        Self object.
      other (NodeSelector):
        Other object.

    Returns:
      NodeSelector:
        New selector containing logical exclusive or.
    """
        temp = lambda x: self._selector_function(x) ^ other._selector_function(x)
        return NodeSelector(temp)

    def __invert__(
            # Arguments:
            self: Self,
            other: NodeSelector
    ) -> NodeSelector:
        """
    Logical inversion.

    Args:
      self (Self):
        Self object.

    Returns:
      NodeSelector:
        New selector performing logical inversion.
    """
        temp = lambda x: not self._selector_function(x)
        return NodeSelector(temp)

    def __bool__(
            # Arguments:
            self: Self
    ) -> bool:
        """
    Produces an error when attempting to use, directing users to use ``&``
    or ``|`` instead.

    Args:
      self (Self):
        Object.

    Returns:
      bool:
        This function will raise an error instead of returning a value.
    """
        raise ValueError(
            "Attempt to convert a NodeSelector into a Boolean value. This is " + \
            "likely a result of of trying to use Python primitive logical " + \
            f"operators (and, or, not, etc.).{linesep}" + \
            "If this was desired, please use the respective bitwise symbols " + \
            "(&, |, ~) instead."
        )

    def __call__(
            # Arguments:
            self: Self
    ) -> Self:
        """
    Handle call on the object. By default this just returns the object, which
    allows for both variable-like and function-like usages (for example,
    ``is_conv`` and ``is_conv()`` would perform the same).

    Args:
      self (Self):
        Object.

    Returns:
      Self:
        Same object.
    """
        return self


def _is_node_of_module(
        # Arguments:
        nodestate: NodeState,
        module: Type[Module]
) -> bool:
    """
  Given a nodestate from a graph, determine if the node is produced by calling
  a method (most likely ``forward``) contained in the given module.

  Args:
    nodestate (NodeState):
      The nodestate to check.
    module (Type[Module]):
      The module class.

  Returns:
    bool:
      Whether the node in the nodestate is produced by the module.
  """
    named_modules = dict(nodestate[0].named_modules())
    if nodestate[1].op != "call_module" or not isinstance(module, type):
        return False
    return isinstance(named_modules.get(nodestate[1].target), module)


def _is_node_of_function(
        # Arguments:
        nodestate: NodeState,
        function: Callable[..., Any]
) -> bool:
    """
  Given a nodestate from a graph, determine if the node is produced by calling
  the given function. This isn't perfect -- it's purely based on if the
  function for the node is literally the function given, even down to the
  memory location it's stored in.

  Args:
    nodestate (NodeState):
      The nodestate to check.
    module (Type[Module]):
      The module class.

  Returns:
    bool:
      Whether the node in the nodestate is produced by the function.
  """
    if nodestate[1].op != "call_function" or not callable(function):
        return False
    return nodestate[1].target == function


def _is_node_of_struct(
        # Arguments:
        nodestate: NodeState,
        structure: Dict[str, List[Any]]
) -> bool:
    """
  Given a nodestate from a graph, determine if the node is produced by calling
  either a module method or a function from the given structure.

  The structure will be of the form
  ``{"Name": [torch.nn.Thing1, torch.nn.functional.Thing2]}``

  Args:
    nodestate (NodeState):
      The nodestate to check.
    structure (Dict[str, List[Any]]):
      The structure.

  Returns:
    bool:
      Whether the node in the nodestate is produced by some callable in the
      structure.
  """
    return any(
        any(
            _is_node_of_module(nodestate, f) or _is_node_of_function(nodestate, f) \
            for f in v
        ) for v in structure.values()
    )


HAS_FOLLOWING_START = 1
HAS_FOLLOWING_STOP = None


def has_following(
        # Arguments:
        node_selector: NodeSelector,
        # Keyword Arguments:
        start: Optional[int] = None,
        stop: Optional[int] = None
) -> NodeSelector:
    """
  Produces a node selector which is true if and only if the specified node
  selector is true for at least one node in the bounds following the current
  node. For example, if used as:

  ``is_conv & has_following(is_activation, start = 1, stop = 3)``

  Then only convolution layers with activations immediately after or with
  one operation in between are selected. This does not select the activation.
  Specifically, ``start`` and ``stop`` look at all nodes in the range
  [``start``, ``stop``), where the current node is ``start = 0``.

  Note: "Following" in this context means that the graph has a forward path to
        one or more following nodes. If a node is computed after the current
        node, but there exists no forward path to that computed node from the
        current node, then it is not following.

  Args:
    node_selector (NodeSelector):
      The node selector to check if following.
    start (int, optional):
      The start position to check following. ``1`` is the node immediately
      following.
      Defaults to ``1`` if ``None``.
    stop (int, optional):
      The end position to check following. ``2`` would only look at the node
      immediately following (as ``stop`` is exclusive).
      Defaults to ``None``, which will check every following node.

  Returns:
    NodeSelector:
      A node selector which checks following nodes.
  """
    # TODO: implement
    raise NotImplementedError()


HAS_PRECEDING_START = 1
HAS_PRECEDING_STOP = None


def has_preceding(
        node_selector: NodeSelector,
        # Keyword Arguments:
        start: Optional[int] = None,
        stop: Optional[int] = None
) -> NodeSelector:
    """
  Produces a node selector which is true if and only if the specified node
  selector is true for at least one node in the bounds preceding the current
  node. For example, if used as:

  ``is_activation & has_preceding(is_conv, start = 1, stop = 3)``

  Then only activation layers with convolutions immediately before or with
  one operation in between are selected. This does not select the convolution.
  Specifically, ``start`` and ``stop`` look at all nodes backwards in the range
  [``start``, ``stop``), where the current node is ``start = 0``.

  Note: "Preceding" in this context means that the graph has a forward path
        from one or more preceding nodes. If a node is computed before the
        current node, but there exists no forward path to that current node
        from the computed node, then it is not preceding.

  Args:
    node_selector (NodeSelector):
      The node selector to check if preceding.
    start (int, optional):
      The start position to check preceding. ``1`` is the node immediately
      preceding.
      Defaults to ``1`` if ``None``.
    stop (int, optional):
      The end position to check preceding. ``2`` would only look at the node
      immediately preceding (as ``stop`` is exclusive).
      Defaults to ``None``, which will check every preceding node.

  Returns:
    NodeSelector:
      A node selector which checks preceding nodes.
  """
    # TODO: implement
    raise NotImplementedError()


# Node selector combinators:

def select_all(
        # Arguments:
        node_selector: NodeSelector
) -> NodeSelector:
    """
  Selects all nodes that match the given node selector.

  Args:
    node_selector (NodeSelector):
      The node selector.

  Returns:
    NodeSelector:
      A new node selector that selects all nodes matching the given node
      selector.
  """
    return node_selector


def select_slice(
        # Arguments:
        node_selector: NodeSelector,
        # Keyword Arguments:
        start: Optional[int] = None,
        stop: Optional[int] = None
) -> NodeSelector:
    # TODO: implement
    raise NotImplementedError()


# Node selector constructors:

# TODO: implement a node selector that gets node by name

def node_lambda(
        # Arguments:
        fn: Callable[[Node], bool]
) -> NodeSelector:
    """
  Constructs a node selector from a function on only a node, ignoring the graph
  context.

  Args:
    fn (Callable[[Node], bool]):
      A function from a node to a Boolean value.

  Returns:
    NodeSelector:
      A node selector which calls the function on the node only.
  """
    temp = lambda x: fn(x[1])
    return NodeSelector(temp)


# Constant node selectors:

is_activation: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, ACTIVATIONS)
)
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
is_act, is_activ = is_activation, is_activation

is_conv: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, CONVOLUTIONS)
)
"""
A node selector which is true for all nodes performing convolution. This
includes convolutions from modules (``torch.nn.Conv2d`` and similar) as well as
functional calls (such as ``torch.nn.functional.conv2d``).
"""
# Aliases:
is_convolution, is_convolutional = is_conv, is_conv

is_pooling: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, POOLINGS)
)
"""
A node selector which is true for all nodes performing pooling. This includes
both regular kernel-based pooling and adaptive pooling, but excludes
``interpolate`` and similar.
"""
# Aliases:
is_pool = is_pooling

is_padding: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, PADDINGS)
)
"""
A node selector which is true for all nodes performing padding, excluding
convolution layers with padding internal.
"""
# Aliases:
is_pad = is_padding

is_norm: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, NORMALIZATIONS)
)
"""
A node selector which is true for all nodes performing normalization.
"""
# Aliases:
is_normalization, is_normalisation = is_norm, is_norm

is_recurrent: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, RECURRENTS)
)
"""
A node selector which is true for all primitive recurrent nodes. This excludes
nodes where recurrence is manually added by repeatedly calling the node -
symbolic tracing cannot handle such scenarios.
"""

is_transformer: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, TRANSFORMERS)
)
"""
A node selector which is true for all Transformer (and Transformer-related)
nodes.
"""

is_linear: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, LINEARS)
)
"""
A node selector which is true for all linear nodes. This includes identity,
linear, and bilinear layers, but does not include matrix multiplication.
"""
# Aliases:
is_lin = is_linear

is_dropout: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, DROPOUTS)
)
"""
A node selector which is true for all dropout nodes.
"""
# Aliases:
is_drop = is_dropout

is_sparse: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, SPARSES)
)
"""
A node selector which is true for all sparse computation nodes. At the moment
this only includes embedding, temporarily allowing the alias ``is_embedding``.
"""
# Aliases:
is_embedding, is_embed = is_sparse, is_sparse

is_distance: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, DISTANCES)
)
"""
A node selector which is true for all distance calculation nodes.
"""
# Aliases:
is_dist = is_distance

is_loss: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, LOSSES)
)
"""
A node selector which is true for all loss nodes.
"""

is_vision: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, VISIONS)
)
"""
A node selector which is true for all vision-related nodes. Specifically this
includes pixel (un)shuffling and upsampling.
"""
# Aliases:
is_vis, is_visual = is_vision, is_vision

is_shuffle: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, SHUFFLINGS)
)
"""
A node selector which is true for all shuffling nodes.
"""

is_utility: NodeSelector = NodeSelector(
    lambda x: _is_node_of_struct(x, UTILITIES)
)
"""
A node selector which is true for all utility nodes.
"""
# Aliases:
is_util = is_utility


def _is_node_parametrized(
        # Arguments:
        nodestate: NodeState
) -> bool:
    """
  Given a nodestate from a graph, determine if the node is from a parametrized
  module.

  Args:
    nodestate (NodeState):
      The nodestate to check.

  Returns:
    bool:
      Whether the node in the nodestate is from a parametrized node.
  """
    named_modules = dict(nodestate[0].named_modules())
    if nodestate[1].op != "call_module":
        return False
    module = named_modules.get(nodestate[1].target)
    try:
        return len(module.parametrizations) >= 1
    except:
        return False


is_parametrized: NodeSelector = NodeSelector(
    _is_node_parametrized
)
"""
A node selector which is true for all layers with parametrizations.
"""
# Aliases:
is_parametrised = is_parametrized

is_module: NodeSelector = NodeSelector(
    lambda x: (x[1].op == "call_module")
)
"""
A node selector which is true for all nodes which call a module.
"""
