import copy
import importlib
import inspect
from contextvars import ContextVar
from typing import Type, Optional, Dict, Any, Union, Callable

import math

import torch
import torchvision
from torch import fx
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.proxy import Scope, ScopeContextManager
from torchvision.models.feature_extraction import NodePathTracer

from torchwatcher.interjection.node_selector import node_selector
from .interjection import (
    Interjection,
    ForwardInterjection,
    WrappedForwardBackwardInterjection,
    WrappedForwardInterjection,
)
from .node_selector import NodeSelector, matches_module_class
from ..nn import GradientIdentity
from ..utils import preserve_training_modes


_NativeCallSpec = tuple[int, tuple[str, ...]]
_native_call_specs: ContextVar[
    dict[int, tuple[_NativeCallSpec, ...]]
] = ContextVar("_native_call_specs", default={})
_native_apply_modules: ContextVar[Optional[set[int]]] = ContextVar(
    "_native_apply_modules",
    default=None,
)


def _dual_graph_state_dict_hook(
        module,
        state_dict,
        prefix,
        _local_metadata,
):
    """Collapse mode-graph state into one logical checkpoint namespace."""
    for graph_name in ("train_module", "eval_module"):
        graph_prefix = f"{prefix}{graph_name}."
        for key in tuple(state_dict):
            if not key.startswith(graph_prefix):
                continue
            logical_key = f"{prefix}{key[len(graph_prefix):]}"
            if logical_key not in state_dict:
                state_dict[logical_key] = state_dict[key]
            del state_dict[key]

    metadata = getattr(state_dict, "_metadata", None)
    if metadata is None:
        return

    logical_root = prefix[:-1]
    for graph_name in ("train_module", "eval_module"):
        graph_root = f"{prefix}{graph_name}"
        for key in tuple(metadata):
            if key == graph_root:
                del metadata[key]
                continue
            if not key.startswith(f"{graph_root}."):
                continue
            suffix = key[len(graph_root) + 1:]
            logical_key = (
                f"{logical_root}.{suffix}" if logical_root else suffix
            )
            if logical_key not in metadata:
                metadata[logical_key] = metadata[key]
            del metadata[key]


def _dual_graph_load_state_dict_pre_hook(
        module,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_messages,
):
    """Expand logical checkpoint keys for each mode graph that owns them."""
    translated_keys = set()
    for graph_name in ("train_module", "eval_module"):
        graph_module = getattr(module, graph_name)
        for key in graph_module.state_dict():
            logical_key = f"{prefix}{key}"
            if logical_key not in state_dict:
                continue
            graph_key = f"{prefix}{graph_name}.{key}"
            if graph_key not in state_dict:
                state_dict[graph_key] = state_dict[logical_key]
            translated_keys.add(logical_key)

    for key in translated_keys:
        del state_dict[key]


class DualGraphModule(nn.Module):
    def __init__(self, train_module: nn.Module, eval_module: nn.Module,
                 class_name="DualGraphModule"):
        super().__init__()

        # This is a per-instance label used by drawing and diagnostics. Renaming
        # ``self.__class__`` would rename every DualGraphModule instance.
        self.graph_name = class_name
        self.train_module = train_module
        self.eval_module = eval_module
        self.register_state_dict_post_hook(_dual_graph_state_dict_hook)
        self.register_load_state_dict_pre_hook(
            _dual_graph_load_state_dict_pre_hook
        )

    @property
    def graphmodule(self):
        if self.training:
            return self.train_module
        else:
            return self.eval_module

    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_module(*args, **kwargs)
        else:
            return self.eval_module(*args, **kwargs)

    def named_parameters(
            self,
            prefix: str = "",
            recurse: bool = True,
            remove_duplicate: bool = True,
    ):
        """Yield parameters under their original logical names.

        The two graph modules are implementation details. Common parameters
        therefore appear once without a ``train_module`` or ``eval_module``
        prefix, while parameters used by only one mode remain visible.
        """
        if not recurse:
            yield from super().named_parameters(
                prefix=prefix,
                recurse=False,
                remove_duplicate=remove_duplicate,
            )
            return

        seen_parameters = set()
        seen_registrations = set()
        for graph_module in (self.train_module, self.eval_module):
            for name, parameter in graph_module.named_parameters(
                    prefix=prefix,
                    remove_duplicate=False,
            ):
                registration = (name, id(parameter))
                if registration in seen_registrations:
                    continue
                seen_registrations.add(registration)
                if remove_duplicate and parameter in seen_parameters:
                    continue
                seen_parameters.add(parameter)
                yield name, parameter

    def named_buffers(
            self,
            prefix: str = "",
            recurse: bool = True,
            remove_duplicate: bool = True,
    ):
        """Yield buffers once under their logical, mode-independent names."""
        if not recurse:
            yield from super().named_buffers(
                prefix=prefix,
                recurse=False,
                remove_duplicate=remove_duplicate,
            )
            return

        seen_buffers = set()
        seen_names = set()
        for graph_module in (self.train_module, self.eval_module):
            for name, buffer in graph_module.named_buffers(
                    prefix=prefix,
                    remove_duplicate=False,
            ):
                if name in seen_names:
                    continue
                seen_names.add(name)
                if remove_duplicate and buffer in seen_buffers:
                    continue
                seen_buffers.add(buffer)
                yield name, buffer

    def apply(self, fn: Callable[[nn.Module], None]):
        """Apply ``fn`` once to each logical module in both mode graphs.

        Train and eval graphs contain distinct module objects whose parameters
        and buffers can represent the same original module. Traversing both
        registered children normally would therefore apply initialisers and
        other mutations more than once.
        """
        modules = []
        seen_names = set()
        seen_modules = set()
        seen_registrations = set()
        for graph_module in (self.train_module, self.eval_module):
            for name, module in graph_module.named_modules():
                registration = (
                    id(module._parameters),
                    id(module._buffers),
                )
                if (name in seen_names
                        or id(module) in seen_modules
                        or registration in seen_registrations):
                    continue
                seen_names.add(name)
                seen_modules.add(id(module))
                seen_registrations.add(registration)
                modules.append(module)

        # Match nn.Module.apply's child-before-parent ordering.
        for module in reversed(modules):
            fn(module)
        fn(self)
        return self


def _get_name(model: Union[torch.nn.Module, Callable[..., Any]]) -> str:
    """Get the name of a model; used for naming the wrapped GraphModule.

    Args:
        model: the model

    Returns:
        the name
    """
    if isinstance(model, nn.Module):
        return model.__class__.__name__
    else:
        return model.__name__


class HierarchyTracer(NodePathTracer):
    def _get_node_qualname(
            self, module_qualname: str, node: fx.Node) -> str:
        """Build a torchvision-style name using literal path matching.

        ``NodePathTracer`` uses an unescaped qualified name as a regular
        expression when detecting duplicates. A path such as ``a.b`` can
        therefore falsely match ``a_b`` and make its suffix parsing fail.
        """
        node_qualname = module_qualname
        if node.op != "call_module":
            if node_qualname:
                node_qualname += "."
            node_qualname += str(node)
            # FX adds numeric suffixes to generated node names. A call_module
            # qualname, by contrast, is the literal user-defined module path;
            # names such as ``block_1`` must never be normalised to ``block``.
            base, separator, suffix = node_qualname.rpartition("_")
            if separator and base and suffix.isdigit():
                node_qualname = base

        duplicate_prefix = f"{node_qualname}_"
        for existing in reversed(self.node_to_qualname.values()):
            if existing == node_qualname:
                return f"{node_qualname}_1"
            if existing.startswith(duplicate_prefix):
                duplicate_suffix = existing[len(duplicate_prefix):]
                if duplicate_suffix.isdigit():
                    return (
                        f"{node_qualname}_{int(duplicate_suffix) + 1}"
                    )

        return node_qualname

    def call_module(self, m, forward, args, kwargs):
        module_qualified_name = self.path_of_module(m)
        with ScopeContextManager(
                self.scope, Scope(module_qualified_name, type(m))
        ) as _scope:
            # module_stack is an ordered dict so writing then deleting the
            # entry is equivalent to push/pop on a list
            num_calls = self.num_calls.get(module_qualified_name, 0)
            module_key = (
                f"{_scope.module_path}@{num_calls}"
                if num_calls > 0
                else _scope.module_path
            )
            self.module_stack[module_key] = (module_qualified_name, _scope.module_type)

        node = super().call_module(m, forward, args, kwargs)

        # node.node.meta['nn_module_stack'] = self.module_stack.copy()
        self.module_stack.pop(module_key)
        return node


def _copy_tracer_kwargs(
        tracer_kwargs: Optional[dict[str, Any]]) -> dict[str, Any]:
    copied = {} if tracer_kwargs is None else dict(tracer_kwargs)
    if "leaf_modules" in copied:
        copied["leaf_modules"] = list(copied["leaf_modules"])
    return copied


def _set_default_tracer_kwargs(original_tr_kwargs: Optional[dict[str, Any]]) -> dict[str, Any]:
    # This is copied from torchvision.models.feature_extraction, but modified so that it doesn't
    # block tracing inside torchvision.ops (you can manually add those modules to the tracer_kwargs
    # if you want, but in general usage you probably do want to trace inside them - otherwise things
    # like tracing ReLUs might miss some!)
    default_autowrap_modules = (math, torchvision.ops)
    default_leaf_modules = []
    result_tracer_kwargs = _copy_tracer_kwargs(original_tr_kwargs)
    result_tracer_kwargs["autowrap_modules"] = (
        tuple(set(result_tracer_kwargs["autowrap_modules"] + default_autowrap_modules))
        if "autowrap_modules" in result_tracer_kwargs
        else default_autowrap_modules
    )
    result_tracer_kwargs["leaf_modules"] = (
        list(set(result_tracer_kwargs["leaf_modules"] + default_leaf_modules))
        if "leaf_modules" in result_tracer_kwargs
        else default_leaf_modules
    )
    return result_tracer_kwargs


def _validate_dual_graph_training_modes(
        model: nn.Module,
        tracer_kwargs: Optional[dict[str, Any]]) -> None:
    """Reject mixed modes which would be specialized during FX tracing.

    A leaf module retains its training flag as runtime module state. In
    contrast, a module which FX traces through can use ``self.training`` in
    Python control flow, permanently specializing the resulting graph. Two
    root-level graphs cannot represent arbitrary per-module mode combinations.
    """
    tracer = HierarchyTracer(**_set_default_tracer_kwargs(tracer_kwargs))
    root_mode = model.training
    mismatched = []

    def visit(module: nn.Module, prefix: str, ancestors: set[int]) -> None:
        for child_name, child in module._modules.items():
            if child is None:
                continue
            qualified_name = (
                f"{prefix}.{child_name}" if prefix else child_name
            )
            is_leaf = tracer.is_leaf_module(child, qualified_name)
            if child.training is not root_mode and not is_leaf:
                mismatched.append(qualified_name)
            if not is_leaf and id(child) not in ancestors:
                visit(child, qualified_name, ancestors | {id(child)})

    visit(model, "", {id(model)})
    if mismatched:
        names = ", ".join(repr(name) for name in mismatched)
        raise ValueError(
            "dual-graph tracing does not support mixed training modes for "
            f"modules traced through by FX; mismatched modules: {names}. "
            "Make their modes uniform or add their classes to "
            "tracer_kwargs['leaf_modules']."
        )


def symbolic_trace(model: Union[torch.nn.Module, Callable[..., Any]],
                   tracer_kwargs: Optional[Dict[str, Any]] = None,
                   concrete_args: Optional[Dict[str, Any]] = None,
                   prefix: str = "",
                   ) -> fx.GraphModule:
    """Custom symbolic tracing using functionality in torchvision's feature
    extraction framework.

    Rather than flattening the model we keep the hierarchy intact by recursively tracing each child submodule.

    Args:
        model: the model to trace
        tracer_kwargs: extra arguments for the tracer
        concrete_args
    """
    # Instantiate our NodePathTracer and use that to trace the model
    tracer_kwargs = _set_default_tracer_kwargs(tracer_kwargs)
    tracer = HierarchyTracer(**tracer_kwargs)
    # tracer = NodePathTracer(**tracer_kwargs)
    graph = tracer.trace(model, concrete_args=concrete_args)

    graph_module = fx.GraphModule(tracer.root, graph, prefix + _get_name(model))

    # We store the qualified names as an extra attribute of each node,
    # allowing a NodeSelector to access
    # without extra external information
    for node in graph_module.graph.nodes:
        if node in tracer.node_to_qualname:
            node.qualified_name = tracer.node_to_qualname[node]

    return graph_module


def get_fresh_qualname(traced: fx.GraphModule, prefix: str) -> str:
    """Generate a unique qualified name for a module that will be added to the
    graph.

    Args:
        traced: the traced module
        prefix: prefix for qualified names

    Returns:
        the generated name
    """

    i = 0
    while True:
        qualname = f"{prefix}{i}"
        i += 1
        if not hasattr(traced, qualname):
            break

    return qualname


def extract_node(graph_module: fx.GraphModule,
                 target_node: fx.Node) -> fx.GraphModule:
    """Extract a node into its own module/graph, re-writing inputs as appropriate.

    This trims all extraneous bits from the graph, just leaving the node of
    interest.

    Args:
        graph_module: the graph module to extract from
        target_node: the node to extract

    Returns:
        the new GraphModule with just that single node in it
    """
    graph = fx.Graph()
    inputs = {
        node: graph.placeholder(node.name)
        for node in target_node.all_input_nodes
    }
    copied_node = graph.node_copy(target_node, inputs.__getitem__)
    graph.output(copied_node)

    # Keep the underlying module shared across the train and eval extractions;
    # only their traced operations need to differ by mode.
    return fx.GraphModule(graph_module, graph)


def add_interjection(graph_module: fx.GraphModule,
                     interjection: Interjection) -> str:
    """Add (or find if it already exists) the given interjection and return its
    name. A Unique name is generated for the cases it's not in the traced
    module already.

    Any given interjection is added as a module just once; when it is
    called, the name of the node is passed along so that if it's reused
    across multiple points the user can tell where it's coming from.

    Args:
        graph_module: the graph module to add the interjection to
        interjection: the interjection to add

    Returns:
        the name of the interjection module within the graph_module
    """
    for name, module in graph_module.named_modules():
        if module == interjection:
            return name

    interjection_name = get_fresh_qualname(graph_module,
                                           type(interjection).__name__)

    graph_module.add_module(interjection_name, interjection)

    return interjection_name


def handle_inplace(
        graph_module: fx.GraphModule,
        node: fx.Node,
) -> fx.Node:
    # handle modules with 'inplace' flags
    if (node.op == 'call_module' and
            hasattr(graph_module.get_submodule(node.target), 'inplace')):
        module = copy.copy(graph_module.get_submodule(node.target))
        module.inplace = False
        graph_module.set_submodule(node.target, module)
        return node

    if node.op == 'call_function':
        if node.kwargs.get('inplace') is True:
            node.kwargs = {**node.kwargs, 'inplace': False}

        target_name = getattr(node.target, '__name__', None)
        if target_name is not None and target_name.endswith('_'):
            module_name = getattr(node.target, '__module__', None)
            replacement = None
            if module_name is not None:
                target_module = importlib.import_module(module_name)
                replacement = getattr(target_module, target_name[:-1], None)

            if replacement is not None:
                node.target = replacement
            else:
                raise ValueError(
                    f"cannot wrap in-place function {node.target!r} with a "
                    "backward interjection: no out-of-place counterpart was "
                    "found"
                )

    if node.op == 'call_method' and node.target.endswith('_'):
        replacement = node.target[:-1]
        if hasattr(torch.Tensor, replacement):
            node.target = replacement
        else:
            raise ValueError(
                f"cannot wrap in-place method {node.target!r} with a "
                "backward interjection: no out-of-place counterpart was "
                "found"
            )

    return node


def preferred_name(node: fx.Node) -> str:
    """Get the preferred name for a given node. Preferred name
     is the qualified name if it exists. Otherwise, it's the regular
     node name.

    Args:
        node: the node

    Returns:
        the preferred name
    """
    if hasattr(node, 'qualified_name'):
        return node.qualified_name
    return node.name


def _copy_module_structure(
        module: nn.Module,
        memo: dict[int, nn.Module] | None = None,
) -> nn.Module:
    """Copy a module hierarchy while retaining its parameters and buffers.

    Wrapped train and eval operations need independent module objects so that
    every descendant can retain its own ``training`` flag. Parameter and
    buffer registration dictionaries deliberately remain shared, preserving
    optimizer state and mutable buffers such as BatchNorm running statistics.
    """
    if memo is None:
        memo = {}

    module_id = id(module)
    if module_id in memo:
        return memo[module_id]

    copied = copy.copy(module)
    memo[module_id] = copied
    copied._modules = {
        name: (
            None
            if child is None
            else _copy_module_structure(child, memo)
        )
        for name, child in module._modules.items()
    }
    return copied


def insert_interjection(graph_module: fx.GraphModule, node: fx.Node,
                        interjection: Interjection,
                        training: bool):
    """Insert an interjection into the traced module at a given node. If the
    interjection is of the wrapped type, then the node is replaced with
    interjection (which wraps the original node); otherwise the new node is
    inserted immediately after the node.

    Args:
        graph_module: the traced module
        node: the insertion point
        interjection: the interjection to insert
    """

    interjection_name = add_interjection(graph_module, interjection)

    original_module = (
        graph_module.get_submodule(node.target)
        if node.op == 'call_module'
        else None
    )
    requires_out_of_place = (
        isinstance(interjection, WrappedForwardBackwardInterjection)
        or getattr(interjection, 'gradient', False)
    )
    if hasattr(interjection, '_wrapped') and node.op == 'call_module':
        # Each traced mode needs its own module object so that the wrapped train
        # and eval operations can retain independent training flags. Parameters
        # and buffers remain shared with the original module.
        module = _copy_module_structure(
            graph_module.get_submodule(node.target)
        )
        graph_module.set_submodule(node.target, module)
    if requires_out_of_place:
        node = handle_inplace(graph_module, node)

    if hasattr(interjection, '_wrapped'):
        with graph_module.graph.inserting_after(node):
            inputs = tuple(node.all_input_nodes)
            extracted = extract_node(graph_module, node)
            # new node to represent the call to the interjection
            args = (preferred_name(node), *inputs)
            new_node = graph_module.graph.call_module(
                interjection_name,
                args,
                {"_torchwatcher_training": training},
            )
            new_node.meta['nn_module_stack'] = node.meta.get(
                'nn_module_stack',
                {},
            ).copy()
            # register the extracted node that we wrap
            interjection._register_for_mode(
                preferred_name(node), extracted, training
            )
            interjection._register_original_module_for_mode(
                preferred_name(node),
                original_module,
                training,
            )
            # clean everything up by replacing uses and inputs, then removing
            # the original node
            node.replace_all_uses_with(new_node)
        graph_module.graph.erase_node(node)
    else:
        with graph_module.graph.inserting_after(node):
            if node.op == 'call_module':
                modules = dict(graph_module.named_modules())
                module = modules[node.target]
            else:
                module = None
            interjection.register(preferred_name(node), module)

            # create the interjection node after the current one
            new_node = graph_module.graph.call_module(interjection_name,
                                                      (preferred_name(node), node,))
            new_node.meta['nn_module_stack'] = node.meta.get(
                'nn_module_stack',
                {},
            ).copy()
            # and hook it to the graph
            node.replace_all_uses_with(new_node)
            new_node.replace_input_with(new_node, node)


def interject_by_module_class(model: nn.Module,
                              target_module_class: Type[nn.Module],
                              interjection: Interjection,
                              tracer_kwargs: Optional[Dict[str, Any]] = None) -> DualGraphModule:
    """Adds an interjection to all nodes that represent a particular nn.Module
    within the provided model.

    To make this work, we make the target module a leaf module in the tracer. This might cause problems if you also
    wanted to interject things inside the target module. Please file a feature request if you need this functionality.
    (Note to self: could be handled by recursively tracing, but we wwould probably need to look at the order in which
    things happen)

    Args:
        model: the model to add the interjection(s) to
        target_module_class: the module to match for the insertion point
        interjection: the interjection to insert
        tracer_kwargs: extra keyword arguments to pass to the tracing module.
                       The 'leaf_modules' key is particularly useful when you
                       want to interject a custom module without tracing inside
                       it.

    Returns:
        the interjected model
    """

    tracer_kwargs = _copy_tracer_kwargs(tracer_kwargs)
    if 'leaf_modules' not in tracer_kwargs:
        tracer_kwargs['leaf_modules'] = []
    tracer_kwargs['leaf_modules'].append(target_module_class)

    selector = matches_module_class(target_module_class)
    return interject_by_match(model, selector, interjection, tracer_kwargs=tracer_kwargs)


def interject_by_match(model: nn.Module, selector: NodeSelector,
                       interjection: Interjection,
                       tracer_kwargs: Optional[Dict[str, Any]] = None) -> DualGraphModule:
    """Adds an interjection to all nodes that represent a particular nn.Module

    Args:
        model: the model
        selector: node selector
        interjection: the interjection
        tracer_kwargs: extra keyword arguments to pass to the tracing module.
                       The 'leaf_modules' key is particularly useful when you
                       want to interject a custom module without tracing inside
                       it.

    Returns:
        the interjected model
    """
    is_training = model.training

    tracer_kwargs = _copy_tracer_kwargs(tracer_kwargs)
    if 'leaf_modules' not in tracer_kwargs:
        tracer_kwargs['leaf_modules'] = []
    tracer_kwargs['leaf_modules'].append(GradientIdentity)

    _validate_dual_graph_training_modes(model, tracer_kwargs)

    with preserve_training_modes(model):
        graphmodules = {}
        for mode in ["train", "eval"]:
            if mode == "train":
                model.train()
            elif mode == "eval":
                model.eval()

            traced = symbolic_trace(model, tracer_kwargs=tracer_kwargs)
            for node in traced.graph.nodes:
                if selector is not None and selector.fn((traced, node)):
                    insert_interjection(
                        traced,
                        node,
                        interjection,
                        training=mode == "train",
                    )

            traced.graph.eliminate_dead_code()
            traced.delete_all_unused_submodules()
            traced.recompile()
            graphmodules[mode] = traced

        # Build the final graph module
        graph_module = DualGraphModule(
            graphmodules["train"],
            graphmodules["eval"],
            class_name="DualGraphModule_" + _get_name(model),
        )
        graph_module.train(is_training)

    return graph_module


def interject_by_name(model: nn.Module, name: str,
                      interjection: Interjection,
                      tracer_kwargs: Optional[Dict[str, Any]] = None) -> DualGraphModule:
    """Adds an interjection to the particular named node

    Args:
        model: the model
        name: the node name
        interjection: the interjection
        tracer_kwargs: extra keyword arguments to pass to the tracing module.
                       The 'leaf_modules' key is particularly useful when you
                       want to interject a custom module without tracing inside
                       it.

    Returns:
        the interjected model
    """
    return interject_by_match(model,
                              node_selector.matches_qualified_name(name),
                              interjection, tracer_kwargs=tracer_kwargs)


def trace(model: nn.Module, tracer_kwargs: Optional[Dict[str, Any]] = None) -> DualGraphModule:
    """Trace a model using the same machinary as when adding an interjection.

    Args:
        model: the model
        tracer_kwargs: extra keyword arguments to pass to the tracing module.
                       The 'leaf_modules' key is particularly useful when you
                       want to interject a custom module without tracing inside
                       it.

    Returns:
        the traced model
    """
    return interject_by_match(model, None, None, tracer_kwargs=tracer_kwargs)


def _bind_shape_inputs(graph_module: fx.GraphModule, args, kwargs):
    """Bind a model call and return its inputs in FX placeholder order."""
    bound = inspect.signature(graph_module.forward).bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.args


def trace_shapes(model: DualGraphModule, *args, **kwargs):
    """Trace input and output shapes using the supplied example model call."""
    import torch._functorch.config as functorch_config
    from torch._subclasses.fake_tensor import FakeTensorMode

    previous_device_type = functorch_config.fake_tensor_prefer_device_type
    functorch_config.fake_tensor_prefer_device_type = 'meta'
    try:
        with preserve_training_modes(model):
            with FakeTensorMode(allow_non_fake_inputs=True):
                model.train_module.train()
                train_args = _bind_shape_inputs(model.train_module, args, kwargs)
                ShapeProp(model.train_module).propagate(*train_args)
                model.eval_module.eval()
                eval_args = _bind_shape_inputs(model.eval_module, args, kwargs)
                ShapeProp(model.eval_module).propagate(*eval_args)
    finally:
        functorch_config.fake_tensor_prefer_device_type = previous_device_type


def trim(network: fx.GraphModule):
    # TODO: implement. This should trim the tail of the graph so that it
    #  stops after the last interjection - but only if needed (like when not
    #  training!)
    pass


class _NativeForwardInterjection(WrappedForwardInterjection):
    """Adapt an output-only interjection for native module wrapping."""

    def __init__(self, interjection: ForwardInterjection):
        super().__init__()
        self.interjection = interjection

    def process(self, name, module, inputs, outputs):
        # Forward interjections observe the output because they are placed
        # after the selected operation.
        return self.interjection.process(name, module, outputs)

    def register(self, name, module):
        super().register(name, module)
        self.interjection.register(name, module)


class _NativeCall(nn.Module):
    """Call a native module while making keyword inputs visible to hooks."""

    def __init__(self, module: nn.Module):
        super().__init__()
        # The model wrapper owns the module. Keep only a non-registering
        # reference here so hook dispatch does not change state-dict keys or
        # let the interjection control the module lifecycle.
        self._module = (module,)

    def __call__(self, *args, **kwargs):
        # Full backward hooks only receive gradients for positional inputs.
        # Present keyword values positionally to the private hooked wrapper,
        # then reconstruct the original call in forward. The context-local
        # stack keeps recursive and concurrent calls independent.
        module_id = id(self)
        call_specs = _native_call_specs.get()
        stack = call_specs.get(module_id, ())
        spec = (len(args), tuple(kwargs))
        token = _native_call_specs.set({
            **call_specs,
            module_id: (*stack, spec),
        })
        try:
            return super().__call__(*args, *kwargs.values())
        finally:
            _native_call_specs.reset(token)

    def forward(self, *flattened_args):
        positional_count, keyword_names = (
            _native_call_specs.get()[id(self)][-1]
        )
        args = flattened_args[:positional_count]
        keyword_values = flattened_args[positional_count:]
        kwargs = dict(zip(
            keyword_names,
            keyword_values,
            strict=True,
        ))
        return self._module[0](*args, **kwargs)


class _NativeInterjectionWrapper(nn.Module):
    """Transparent, serialisable owner for a natively interjected module."""

    def __init__(
            self,
            name: str,
            module: nn.Module,
            interjection: Interjection,
    ):
        super().__init__()
        self._torchwatcher_name = name
        self._torchwatcher_original_module = (module,)
        self._torchwatcher_interjection = (interjection,)
        # The wrapper occupies the original registration slot, so it must
        # expose the target module's registrations directly. If the target
        # were instead registered as a child, ``0.weight`` would become
        # ``0.0.weight`` and selectors/checkpoints would no longer describe
        # the original model.
        self._parameters = module._parameters
        self._buffers = module._buffers
        self._modules = module._modules
        self._non_persistent_buffers_set = module._non_persistent_buffers_set
        self._version = module._version
        self.training = module.training
        self.register_load_state_dict_post_hook(
            self._dispatch_original_load_state_dict_post_hooks
        )

    @property
    def _original_module(self):
        return object.__getattribute__(
            self,
            "_torchwatcher_original_module",
        )[0]

    @property
    def _interjection(self):
        return object.__getattribute__(
            self,
            "_torchwatcher_interjection",
        )[0]

    def __getattr__(self, attribute):
        try:
            return super().__getattr__(attribute)
        except AttributeError as error:
            try:
                original_module = self._original_module
            except AttributeError:
                raise error
            return getattr(original_module, attribute)

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        # Parameter selectors (and standard PyTorch traversal) should see the
        # original owner and its original qualified name.
        yield from self._original_module.named_modules(
            memo=memo,
            prefix=prefix,
            remove_duplicate=remove_duplicate,
        )

    def train(self, mode: bool = True):
        self._original_module.train(mode)
        self.training = self._original_module.training
        return self

    def apply(self, fn):
        seen_modules = _native_apply_modules.get()
        token = None
        if seen_modules is None:
            seen_modules = set()
            token = _native_apply_modules.set(seen_modules)

        try:
            module_id = id(self._original_module)
            if module_id not in seen_modules:
                seen_modules.add(module_id)
                self._original_module.apply(fn)
            return self
        finally:
            if token is not None:
                _native_apply_modules.reset(token)

    def _apply(self, fn, recurse=True):
        self._original_module._apply(fn, recurse=recurse)
        return self

    def state_dict(self, *args, **kwargs):
        return self._original_module.state_dict(*args, **kwargs)

    def _load_from_state_dict(self, *args, **kwargs):
        return self._original_module._load_from_state_dict(*args, **kwargs)

    def _dispatch_original_load_state_dict_post_hooks(
            self,
            _module,
            incompatible_keys,
    ):
        for hook in self._original_module._load_state_dict_post_hooks.values():
            hook(self._original_module, incompatible_keys)

    def forward(self, *args, **kwargs):
        return self._interjection.forward(
            self._torchwatcher_name,
            *args,
            **kwargs,
        )


class _NativeApply:
    """Coordinate one logical ``apply`` traversal across alias wrappers."""

    def __init__(self, module: nn.Module):
        self._module = (module,)

    def __call__(self, fn):
        module = self._module[0]
        seen_modules = _native_apply_modules.get()
        if seen_modules is not None:
            return type(module).apply(module, fn)

        token = _native_apply_modules.set(set())
        try:
            return type(module).apply(module, fn)
        finally:
            _native_apply_modules.reset(token)


def interject_by_module_class_native(model: nn.Module,
                                     target_module_class: Type[nn.Module],
                                     interjection: Interjection,
                                     clone=True,
                                     ) -> nn.Module:
    """Adds an interjection to all nodes that represent a particular nn.Module
    within the provided model. We do this natively, without tracing. Because we're not
    computing the graph we always wrap the target module (so even a ForwardInterjection
    will be wrapped).

    Args:
        model: the model to add the interjection(s) to
        target_module_class: the module to match for the insertion point
        interjection: the interjection to insert
        clone: whether to clone the model before interjecting (default True)

    A matched root module is reported as ``"root"``. If that name is already
    used by a module in the model, a numeric ``@`` suffix is added.

    Returns:
        the interjected model
    """
    if clone:
        model = copy.deepcopy(model)

    existing_names = {
        name
        for name, _ in model.named_modules(remove_duplicate=False)
        if name
    }
    root_name = "root"
    suffix = 1
    while root_name in existing_names:
        root_name = f"root@{suffix}"
        suffix += 1

    if isinstance(interjection, ForwardInterjection):
        interjection = _NativeForwardInterjection(interjection)

    has_interjection = False

    def add_interjection(name, module):
        nonlocal has_interjection
        has_interjection = True
        native_call = _NativeCall(module)

        # Each alias gets its own call module so names and backward hooks stay
        # distinct, while every alias wrapper exposes the same original module.
        interjection.register(name, native_call)
        interjection.register_original_module(name, module)
        return _NativeInterjectionWrapper(name, module, interjection)

    traversed: set[int] = set()

    def add_recursively(
            module: nn.Module,
            name: str = "",
            *,
            is_root: bool = False,
    ) -> nn.Module:
        module_id = id(module)
        is_target = isinstance(module, target_module_class)
        if module_id not in traversed:
            traversed.add(module_id)
            for child_name, child in list(module._modules.items()):
                if child is None:
                    continue
                qualified_name = (
                    f"{name}.{child_name}" if name else child_name
                )
                replacement = add_recursively(child, qualified_name)
                if replacement is not child:
                    setattr(module, child_name, replacement)

        if not is_target:
            return module
        observation_name = root_name if is_root else name
        return add_interjection(
            observation_name,
            module,
        )

    model = add_recursively(model, is_root=True)
    if has_interjection and not isinstance(
            model,
            _NativeInterjectionWrapper,
    ):
        model.apply = _NativeApply(model)
    return model
