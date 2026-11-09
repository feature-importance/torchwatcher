import copy
import importlib
import inspect
import warnings
from collections import OrderedDict
from contextvars import ContextVar
from types import MethodType
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
_TORCHWATCHER_STATE_PREFIX = "_torchwatcher."


def _registered_state(module: nn.Module, *, keep_vars: bool = True):
    """Return standard persistent state without invoking state-dict hooks."""
    state = OrderedDict()
    metadata = OrderedDict()

    def visit(current: nn.Module, prefix: str, ancestors: set[int]):
        metadata[prefix[:-1]] = {"version": current._version}
        current._save_to_state_dict(state, prefix, keep_vars)

        for name, child in current._modules.items():
            if child is None or id(child) in ancestors:
                continue
            visit(
                child,
                prefix + name + ".",
                ancestors | {id(child)},
            )

    visit(module, "", {id(module)})
    state._metadata = metadata
    return state


def _state_objects(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return persistent parameter and buffer registrations by state name."""
    objects = {}

    def visit(current: nn.Module, prefix: str, ancestors: set[int]):
        for name, parameter in current._parameters.items():
            if parameter is not None:
                objects[prefix + name] = parameter
        for name, buffer in current._buffers.items():
            if (
                    buffer is not None
                    and name not in current._non_persistent_buffers_set
            ):
                objects[prefix + name] = buffer
        for name, child in current._modules.items():
            if child is None or id(child) in ancestors:
                continue
            visit(
                child,
                prefix + name + ".",
                ancestors | {id(child)},
            )

    visit(module, "", {id(module)})
    return objects


def _dual_graph_checkpoint_names(module, graph_name, state_name):
    source_names = module._graph_state_to_source.get(
        (graph_name, state_name),
    )
    if source_names is not None:
        return source_names
    if state_name.split(".", 1)[0] in module._auxiliary_module_names:
        return (f"{_TORCHWATCHER_STATE_PREFIX}{state_name}",)
    return (state_name,)


def _dual_graph_metadata_names(module, graph_name, module_name):
    """Return public checkpoint paths for an internal module registration."""
    if not module_name:
        return ("",)
    source_names = module._graph_module_to_source.get(
        (graph_name, module_name),
    )
    if source_names is not None:
        return source_names
    if module_name.split(".", 1)[0] in module._auxiliary_module_names:
        return (f"{_TORCHWATCHER_STATE_PREFIX}{module_name}",)
    return (module_name,)


def _state_shapes_mismatch(expected, supplied):
    """Match PyTorch's standard parameter and buffer shape validation."""
    if (
            not torch.overrides.is_tensor_like(expected)
            or not torch.overrides.is_tensor_like(supplied)
            or torch.nn.parameter.is_lazy(expected)
    ):
        return False
    if (
            len(expected.shape) == 0
            and len(supplied.shape) == 1
            and supplied.shape[0] == 1
    ):
        # Preserve PyTorch's compatibility with scalar checkpoints from 0.3.
        return False
    return supplied.shape != expected.shape


def _load_from_state_dict_with_current_version(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_messages,
):
    """Guard an internal load and supply absent version metadata if needed."""
    patch_state = module.__dict__["_torchwatcher_version_load_patch"]
    if patch_state["version_users"]:
        local_metadata.setdefault("version", module._version)
    try:
        return patch_state["original"](
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_messages,
        )
    except BaseException:
        for cleanup in tuple(reversed(patch_state["cleanups"])):
            cleanup()
        raise


def _patch_current_module_version(
        module,
        cleanup,
        *,
        supply_version,
):
    """Guard an internal load, optionally supplying current version metadata."""
    patch_state = module.__dict__.get("_torchwatcher_version_load_patch")
    if patch_state is None:
        patch_state = {
            "original": module._load_from_state_dict,
            "had_instance_attribute": (
                "_load_from_state_dict" in module.__dict__
            ),
            "instance_attribute": module.__dict__.get(
                "_load_from_state_dict"
            ),
            "users": 0,
            "version_users": 0,
            "cleanups": [],
        }
        module.__dict__["_torchwatcher_version_load_patch"] = patch_state
        object.__setattr__(
            module,
            "_load_from_state_dict",
            MethodType(_load_from_state_dict_with_current_version, module),
        )
    patch_state["users"] += 1
    patch_state["version_users"] += int(supply_version)
    if cleanup not in patch_state["cleanups"]:
        patch_state["cleanups"].append(cleanup)


def _unpatch_current_module_version(
        module,
        cleanup,
        *,
        supply_version,
):
    patch_state = module.__dict__["_torchwatcher_version_load_patch"]
    patch_state["users"] -= 1
    patch_state["version_users"] -= int(supply_version)
    if cleanup in patch_state["cleanups"]:
        patch_state["cleanups"].remove(cleanup)
    if patch_state["users"]:
        return
    if patch_state["had_instance_attribute"]:
        object.__setattr__(
            module,
            "_load_from_state_dict",
            patch_state["instance_attribute"],
        )
    else:
        object.__delattr__(module, "_load_from_state_dict")
    del module.__dict__["_torchwatcher_version_load_patch"]


def _is_dual_graph_load_hook(hook):
    hook = getattr(hook, "hook", hook)
    return hook in {
        _dual_graph_load_state_dict_pre_hook,
        _dual_graph_load_state_dict_post_hook,
    }


def _begin_dual_graph_load_transaction(module, context):
    """Deduplicate public hooks during internal mode-graph traversal."""
    source_module = module.__dict__["_torchwatcher_source_module"][0]
    mapped_modules = {
        id(getattr(module, graph_name).get_submodule(module_name))
        for graph_name, module_name in module._graph_module_to_source
    }
    source_modules = (
        set()
        if source_module is None
        else {
            id(child)
            for _name, child in source_module.named_modules(
                remove_duplicate=False
            )
        }
    )
    preprocessed_modules = context.get("preprocessed_modules", set())
    deferred_post_modules = context.get("deferred_post_modules", set())

    graph_modules = []
    versioned_modules = set()
    seen_modules = set()
    for graph_name in ("train_module", "eval_module"):
        graph_module = getattr(module, graph_name)
        for module_name, child in graph_module.named_modules(
                remove_duplicate=False):
            child_id = id(child)
            if child_id not in seen_modules:
                seen_modules.add(child_id)
                graph_modules.append(child)
            if (graph_name, module_name) in module._graph_module_to_source:
                versioned_modules.add(child_id)

    hook_states = []
    patched_modules = []
    active = True

    def cleanup(remove_context=True):
        nonlocal active
        if not active:
            return
        active = False
        for child, original_pre, original_post, adapted_pre, adapted_post in (
                hook_states):
            if child._load_state_dict_pre_hooks is adapted_pre:
                child._load_state_dict_pre_hooks = original_pre
            if child._load_state_dict_post_hooks is adapted_post:
                child._load_state_dict_post_hooks = original_post
        for child, supply_version in reversed(patched_modules):
            if "_torchwatcher_version_load_patch" in child.__dict__:
                _unpatch_current_module_version(
                    child,
                    cleanup,
                    supply_version=supply_version,
                )
        if remove_context:
            contexts = module.__dict__.get(
                "_torchwatcher_load_state_dict_contexts",
                [],
            )
            if context in contexts:
                contexts.remove(context)
            if not contexts:
                module.__dict__.pop(
                    "_torchwatcher_load_state_dict_contexts",
                    None,
                )

    def guarded(hook, should_run):
        def run(*args, **kwargs):
            if not should_run():
                return None
            try:
                return hook(*args, **kwargs)
            except BaseException:
                cleanup()
                raise
        return run

    for child in graph_modules:
        original_pre = child._load_state_dict_pre_hooks
        original_post = child._load_state_dict_post_hooks
        remaining = 0 if (
            id(child) in source_modules or id(child) in mapped_modules
        ) else 1
        hook_state = {
            "remaining": remaining,
            "pre": False,
            "post": False,
            "decisions": [],
        }

        def begin_pre(*_args, hook_state=hook_state):
            run = hook_state["remaining"] > 0
            if run:
                hook_state["remaining"] -= 1
            hook_state["pre"] = run
            hook_state["decisions"].append(run)

        def begin_post(*_args, hook_state=hook_state):
            hook_state["post"] = (
                hook_state["decisions"].pop(0)
                if hook_state["decisions"]
                else False
            )

        adapted_pre = OrderedDict()
        adapted_pre[("torchwatcher", id(context))] = begin_pre
        for key, hook in original_pre.items():
            adapted_pre[key] = guarded(
                hook,
                lambda hook=hook, hook_state=hook_state, child=child: (
                    _is_dual_graph_load_hook(hook)
                    or (
                        hook_state["pre"]
                        and id(child) not in preprocessed_modules
                    )
                ),
            )

        adapted_post = OrderedDict()
        adapted_post[("torchwatcher", id(context))] = begin_post
        for key, hook in original_post.items():
            adapted_post[key] = guarded(
                hook,
                lambda hook=hook, hook_state=hook_state, child=child: (
                    _is_dual_graph_load_hook(hook)
                    or (
                        hook_state["post"]
                        and id(child) not in deferred_post_modules
                    )
                ),
            )

        child._load_state_dict_pre_hooks = adapted_pre
        child._load_state_dict_post_hooks = adapted_post
        hook_states.append((
            child,
            original_pre,
            original_post,
            adapted_pre,
            adapted_post,
        ))

        supply_version = id(child) in versioned_modules
        _patch_current_module_version(
            child,
            cleanup,
            supply_version=supply_version,
        )
        patched_modules.append((child, supply_version))

    context["cleanup"] = cleanup


def _merge_dual_graph_state(
        module,
        state_dict,
        prefix,
        keep_vars,
):
    """Merge graph-owned state into an already saved logical checkpoint."""
    graph_written = set()
    graph_states = {}
    for graph_name in ("train_module", "eval_module"):
        graph_state = _registered_state(
            getattr(module, graph_name),
            keep_vars=keep_vars,
        )
        graph_states[graph_name] = graph_state
        for state_name, value in graph_state.items():
            if (graph_name, state_name) in module._graph_state_to_source:
                continue
            for checkpoint_name in _dual_graph_checkpoint_names(
                    module,
                    graph_name,
                    state_name,
            ):
                logical_key = f"{prefix}{checkpoint_name}"
                if logical_key in graph_written:
                    continue
                state_dict[logical_key] = value
                graph_written.add(logical_key)

    metadata = getattr(state_dict, "_metadata", None)
    if metadata is None:
        return

    logical_root = prefix[:-1]
    for graph_name in ("train_module", "eval_module"):
        graph_metadata = getattr(
            graph_states[graph_name],
            "_metadata",
            {},
        )
        for module_name, local_metadata in graph_metadata.items():
            if (graph_name, module_name) in module._graph_module_to_source:
                continue
            for checkpoint_name in _dual_graph_metadata_names(
                    module, graph_name, module_name):
                logical_key = ".".join(filter(None, (
                    logical_root,
                    checkpoint_name,
                )))
                metadata.setdefault(logical_key, local_metadata)


def _graph_owned_module_occurrences(module):
    """Return graph-only modules once in their public checkpoint namespace."""
    occurrences = []
    seen_names = set()
    for graph_name in ("train_module", "eval_module"):
        graph_module = getattr(module, graph_name)
        for module_name, child in graph_module.named_modules(
                remove_duplicate=False):
            if (
                    not module_name
                    or (graph_name, module_name)
                    in module._graph_module_to_source
            ):
                continue
            for checkpoint_name in _dual_graph_metadata_names(
                    module, graph_name, module_name):
                if checkpoint_name in seen_names:
                    continue
                seen_names.add(checkpoint_name)
                occurrences.append((checkpoint_name, child))
    return occurrences


def _run_graph_state_dict_pre_hooks(
        module,
        destination,
        prefix,
        keep_vars,
):
    occurrences = _graph_owned_module_occurrences(module)
    metadata = getattr(destination, "_metadata", None)
    for checkpoint_name, child in occurrences:
        module_prefix = f"{prefix}{checkpoint_name}."
        if metadata is not None:
            metadata.setdefault(
                module_prefix[:-1],
                {"version": child._version},
            )
        for hook in child._state_dict_pre_hooks.values():
            hook(child, module_prefix, keep_vars)
    return occurrences


def _run_graph_state_dict_post_hooks(
        occurrences,
        destination,
        prefix,
):
    metadata = getattr(destination, "_metadata", {})
    for checkpoint_name, child in reversed(occurrences):
        module_prefix = f"{prefix}{checkpoint_name}."
        local_metadata = metadata.get(
            module_prefix[:-1],
            {"version": child._version},
        )
        for hook in child._state_dict_hooks.values():
            hook_result = hook(
                child,
                destination,
                module_prefix,
                local_metadata,
            )
            if not getattr(hook, "_from_public_api", False):
                if hook_result is not None:
                    destination = hook_result
            elif hook_result is not None:
                raise RuntimeError("state_dict post-hook must return None")
    return destination


def _preprocess_graph_owned_load_hooks(
        module,
        state_dict,
        prefix,
        metadata,
        strict,
        assign,
        missing_keys,
        unexpected_keys,
        error_messages,
):
    """Run graph-only load pre-hooks once in their public namespace."""
    preprocessed_modules = set()
    logical_root = prefix[:-1]
    for checkpoint_name, child in _graph_owned_module_occurrences(module):
        hooks = [
            hook for hook in child._load_state_dict_pre_hooks.values()
            if not _is_dual_graph_load_hook(hook)
        ]
        if not hooks:
            continue

        module_prefix = f"{prefix}{checkpoint_name}."
        local_state = OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if key.startswith(module_prefix)
        )
        metadata_name = ".".join(filter(None, (
            logical_root,
            checkpoint_name,
        )))
        local_metadata = dict(
            {} if metadata is None else metadata.get(metadata_name, {})
        )
        if assign:
            local_metadata["assign_to_params_buffers"] = True
        for hook in hooks:
            hook(
                local_state,
                module_prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_messages,
            )

        for key in tuple(state_dict):
            if key.startswith(module_prefix):
                del state_dict[key]
        state_dict.update(local_state)
        preprocessed_modules.add(id(child))

    return preprocessed_modules


def _source_module_occurrences(source_module):
    return list(source_module.named_modules(remove_duplicate=False))


def _preprocess_source_load_hooks(
        occurrences,
        source_state,
        prefix,
        metadata,
        strict,
        assign,
        missing_keys,
        unexpected_keys,
        error_messages,
):
    """Run source load pre-hooks with the watched model's public prefix."""
    public_state = OrderedDict(
        (f"{prefix}{key}", value)
        for key, value in source_state.items()
    )
    logical_root = prefix[:-1]
    for module_name, child in occurrences:
        hooks = [
            hook for hook in child._load_state_dict_pre_hooks.values()
            if not _is_dual_graph_load_hook(hook)
        ]
        if not hooks:
            continue

        module_prefix = f"{prefix}{module_name}." if module_name else prefix
        local_state = OrderedDict(
            (key, value)
            for key, value in public_state.items()
            if key.startswith(module_prefix)
        )
        metadata_name = ".".join(filter(None, (
            logical_root,
            module_name,
        )))
        local_metadata = dict(
            {} if metadata is None else metadata.get(metadata_name, {})
        )
        if assign:
            local_metadata["assign_to_params_buffers"] = True
        for hook in hooks:
            hook(
                local_state,
                module_prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_messages,
            )

        for key in tuple(public_state):
            if key.startswith(module_prefix):
                del public_state[key]
        public_state.update(local_state)

    source_state.clear()
    source_state.update(
        (key[len(prefix):], value)
        for key, value in public_state.items()
        if key.startswith(prefix)
    )


def _suppress_source_load_hooks(occurrences):
    """Temporarily retain only TorchWatcher-internal source load hooks."""
    states = []
    seen = set()
    for _module_name, child in occurrences:
        if id(child) in seen:
            continue
        seen.add(id(child))
        original_pre = child._load_state_dict_pre_hooks
        original_post = child._load_state_dict_post_hooks
        adapted_pre = OrderedDict(
            (key, hook) for key, hook in original_pre.items()
            if _is_dual_graph_load_hook(hook)
        )
        adapted_post = OrderedDict(
            (key, hook) for key, hook in original_post.items()
            if _is_dual_graph_load_hook(hook)
        )
        child._load_state_dict_pre_hooks = adapted_pre
        child._load_state_dict_post_hooks = adapted_post
        states.append((
            child,
            original_pre,
            original_post,
            adapted_pre,
            adapted_post,
        ))

    def restore():
        for child, original_pre, original_post, adapted_pre, adapted_post in (
                states):
            if child._load_state_dict_pre_hooks is adapted_pre:
                child._load_state_dict_pre_hooks = original_pre
            if child._load_state_dict_post_hooks is adapted_post:
                child._load_state_dict_post_hooks = original_post

    return restore


def _deferred_load_post_hooks(occurrences):
    deferred = []
    for _module_name, child in occurrences:
        hooks = tuple(
            hook for hook in child._load_state_dict_post_hooks.values()
            if not _is_dual_graph_load_hook(hook)
        )
        if hooks:
            deferred.append((child, hooks))
    return deferred


def _dual_graph_load_state_dict_pre_hook(
        module,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        missing_keys,
        _unexpected_keys,
        error_messages,
):
    """Expand logical checkpoint keys for each mode graph that owns them."""
    translated_keys = set()
    reported_shape_mismatches = set()
    relative_keys = {
        key[len(prefix):]
        for key in state_dict
        if key.startswith(prefix)
    }
    is_source_checkpoint = not any(
        key.startswith(_TORCHWATCHER_STATE_PREFIX)
        for key in relative_keys
    )
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        logical_root = prefix[:-1]
        for graph_name in ("train_module", "eval_module"):
            graph_module = getattr(module, graph_name)
            for module_name, _ in graph_module.named_modules(
                    remove_duplicate=False):
                checkpoint_names = _dual_graph_metadata_names(
                    module,
                    graph_name,
                    module_name,
                )
                supplied_names = [
                    ".".join(filter(None, (logical_root, checkpoint_name)))
                    for checkpoint_name in checkpoint_names
                ]
                supplied_names = [
                    name for name in supplied_names if name in metadata
                ]
                if supplied_names:
                    internal_name = ".".join(filter(None, (
                        logical_root,
                        graph_name,
                        module_name,
                    )))
                    metadata.setdefault(
                        internal_name,
                        metadata[supplied_names[-1]],
                    )

    preprocessed_modules = _preprocess_graph_owned_load_hooks(
        module,
        state_dict,
        prefix,
        metadata,
        _strict,
        bool(_local_metadata.get("assign_to_params_buffers", False)),
        missing_keys,
        _unexpected_keys,
        error_messages,
    )

    for graph_name in ("train_module", "eval_module"):
        graph_module = getattr(module, graph_name)
        graph_state_objects = _state_objects(graph_module)
        for key, value in _registered_state(
                graph_module, keep_vars=False).items():
            checkpoint_names = _dual_graph_checkpoint_names(
                module,
                graph_name,
                key,
            )
            supplied_keys = [
                f"{prefix}{checkpoint_name}"
                for checkpoint_name in checkpoint_names
                if f"{prefix}{checkpoint_name}" in state_dict
            ]
            graph_key = f"{prefix}{graph_name}.{key}"
            if supplied_keys:
                # Shared aliases are loaded in source traversal order, so the
                # final alias has the same precedence as nn.Module loading.
                supplied_key = supplied_keys[-1]
                supplied_value = state_dict[supplied_key]
                shape_mismatch = (
                    key in graph_state_objects
                    and _state_shapes_mismatch(
                        graph_state_objects[key],
                        supplied_value,
                    )
                )
                if (
                        shape_mismatch
                        and supplied_key not in reported_shape_mismatches
                ):
                    error_messages.append(
                        f"size mismatch for {supplied_key}: copying a param "
                        f"with shape {supplied_value.shape} from checkpoint, "
                        f"the shape in current model is {value.shape}."
                    )
                    reported_shape_mismatches.add(supplied_key)
                if graph_key not in state_dict:
                    # A mismatched public tensor has already produced one
                    # logical error. Use the current value internally so each
                    # mode graph does not report the same implementation-path
                    # mismatch again.
                    state_dict[graph_key] = (
                        graph_state_objects[key]
                        if shape_mismatch
                        else supplied_value
                    )
                translated_keys.update(supplied_keys)
                for checkpoint_name in checkpoint_names:
                    checkpoint_key = f"{prefix}{checkpoint_name}"
                    if (
                            checkpoint_key not in state_dict
                            and checkpoint_key not in missing_keys
                    ):
                        missing_keys.append(checkpoint_key)
            elif (
                    is_source_checkpoint
                    and checkpoint_names[0].startswith(
                        _TORCHWATCHER_STATE_PREFIX
                    )
                    and graph_key not in state_dict
            ):
                # An original checkpoint intentionally has no analyser state.
                # Preserve the analyser's current initialisation.
                state_dict[graph_key] = value

    deferred_source_post_hooks = []
    source_module = module.__dict__["_torchwatcher_source_module"][0]
    if source_module is not None:
        source_template = _registered_state(source_module, keep_vars=True)
        source_state = OrderedDict()
        source_state._metadata = copy.deepcopy(
            getattr(source_template, "_metadata", OrderedDict())
        )
        source_state_objects = _state_objects(source_module)
        for source_name in module._source_state_names:
            source_key = f"{prefix}{source_name}"
            if source_key not in state_dict:
                continue

            supplied_value = state_dict[source_key]
            shape_mismatch = (
                source_name in source_state_objects
                and _state_shapes_mismatch(
                    source_state_objects[source_name],
                    supplied_value,
                )
            )
            if shape_mismatch:
                if source_key not in reported_shape_mismatches:
                    error_messages.append(
                        f"size mismatch for {source_key}: copying a param "
                        f"with shape {supplied_value.shape} from checkpoint, "
                        "the shape in current model is "
                        f"{source_template[source_name].shape}."
                    )
                    reported_shape_mismatches.add(source_key)
                source_state[source_name] = source_template[source_name]
            else:
                source_state[source_name] = supplied_value
            translated_keys.add(source_key)

        # State-dict post-hooks may add public keys which are consumed by a
        # corresponding load pre-hook. They are deliberately absent from the
        # hook-free construction schema, so offer otherwise unknown logical
        # keys to the source load and retain only genuinely unexpected ones.
        hook_candidates = {}
        for full_key in state_dict:
            if not full_key.startswith(prefix):
                continue
            supplied_key = full_key[len(prefix):]
            if (
                    full_key in translated_keys
                    or supplied_key.startswith(_TORCHWATCHER_STATE_PREFIX)
                    or supplied_key in source_template
            ):
                continue
            source_state[supplied_key] = state_dict[full_key]
            hook_candidates[supplied_key] = full_key

        source_metadata = getattr(source_state, "_metadata", None)
        if source_metadata is not None and metadata is not None:
            logical_root = prefix[:-1]
            for source_name in tuple(source_metadata):
                logical_name = ".".join(filter(None, (
                    logical_root,
                    source_name,
                )))
                if logical_name in metadata:
                    source_metadata[source_name] = metadata[logical_name]
        assign = bool(_local_metadata.get("assign_to_params_buffers", False))
        source_occurrences = _source_module_occurrences(source_module)
        deferred_source_post_hooks = _deferred_load_post_hooks(
            source_occurrences
        )
        _preprocess_source_load_hooks(
            source_occurrences,
            source_state,
            prefix,
            metadata,
            _strict,
            assign,
            missing_keys,
            _unexpected_keys,
            error_messages,
        )
        restore_source_hooks = _suppress_source_load_hooks(
            source_occurrences
        )
        try:
            try:
                source_incompatible = source_module.load_state_dict(
                    source_state,
                    strict=False,
                    assign=assign,
                )
            finally:
                restore_source_hooks()
        except RuntimeError as error:
            error_messages.append(str(error))
        else:
            source_missing_names = source_incompatible.missing_keys
            unexpected = set(source_incompatible.unexpected_keys)
            translated_keys.update(
                full_key
                for source_name, full_key in hook_candidates.items()
                if source_name not in unexpected
            )
            for source_name in source_missing_names:
                source_key = f"{prefix}{source_name}"
                if (
                        source_key not in missing_keys
                ):
                    missing_keys.append(source_key)

            # Source hooks may rename expected keys before loading them. Seed
            # mapped graph registrations from the resulting source state so
            # internal traversal observes exactly what those hooks accepted.
            loaded_source_state = _registered_state(
                source_module,
                keep_vars=True,
            )
            for (graph_name, state_name), source_names in (
                    module._graph_state_to_source.items()):
                graph_key = f"{prefix}{graph_name}.{state_name}"
                loaded_names = [
                    source_name
                    for source_name in source_names
                    if source_name in loaded_source_state
                ]
                if loaded_names:
                    state_dict[graph_key] = loaded_source_state[
                        loaded_names[-1]
                    ]
                else:
                    state_dict.pop(graph_key, None)

    graph_owned_occurrences = _graph_owned_module_occurrences(module)
    deferred_graph_post_hooks = _deferred_load_post_hooks(
        graph_owned_occurrences
    )
    context = {
        "prefix": prefix,
        "cleanup": None,
        "preprocessed_modules": preprocessed_modules,
        "deferred_post_modules": {
            id(child)
            for child, _hooks in deferred_graph_post_hooks
        },
        "deferred_post_hooks": (
            deferred_source_post_hooks + deferred_graph_post_hooks
        ),
    }
    module.__dict__.setdefault(
        "_torchwatcher_load_state_dict_contexts",
        [],
    ).append(context)
    _begin_dual_graph_load_transaction(module, context)

    for key in translated_keys:
        del state_dict[key]


def _dual_graph_load_state_dict_post_hook(module, incompatible_keys):
    """Expose load diagnostics in the public checkpoint namespace."""
    contexts = module.__dict__["_torchwatcher_load_state_dict_contexts"]
    context = contexts.pop()
    context["cleanup"](remove_context=False)
    prefix = context["prefix"]
    if not contexts:
        del module.__dict__["_torchwatcher_load_state_dict_contexts"]

    translations = {}
    for graph_name in ("train_module", "eval_module"):
        graph_module = getattr(module, graph_name)
        for state_name in _registered_state(graph_module):
            internal_name = f"{graph_name}.{state_name}"
            checkpoint_names = _dual_graph_checkpoint_names(
                module,
                graph_name,
                state_name,
            )
            translations[internal_name] = checkpoint_names

    def translate(keys):
        translated = []
        for key in keys:
            if not key.startswith(prefix):
                replacements = (key,)
            else:
                checkpoint_names = translations.get(key[len(prefix):])
                replacements = (
                    (key,)
                    if checkpoint_names is None
                    else tuple(
                        f"{prefix}{name}" for name in checkpoint_names
                    )
                )
            for replacement in replacements:
                if replacement not in translated:
                    translated.append(replacement)
        return translated

    incompatible_keys.missing_keys[:] = translate(
        incompatible_keys.missing_keys
    )
    incompatible_keys.unexpected_keys[:] = translate(
        incompatible_keys.unexpected_keys
    )

    for child, hooks in reversed(context["deferred_post_hooks"]):
        for hook in hooks:
            result = hook(child, incompatible_keys)
            if result is not None:
                raise AssertionError(
                    "Hooks registered with "
                    "register_load_state_dict_post_hook are not expected "
                    "to return new values; modify incompatible_keys in place"
                )


def _build_dual_graph_state_mapping(
        source_module,
        train_module,
        eval_module,
        original_module_copies=(),
):
    if source_module is None:
        return (), (), {}, {}

    source_objects = _state_objects(source_module)
    source_state_names = tuple(_registered_state(source_module))
    source_state_name_set = set(source_state_names)
    source_names_by_id: dict[int, list[str]] = {}
    for name, value in source_objects.items():
        source_names_by_id.setdefault(id(value), []).append(name)

    source_module_names_by_id: dict[int, list[str]] = {}
    for name, source_child in source_module.named_modules(
            remove_duplicate=False):
        source_module_names_by_id.setdefault(id(source_child), []).append(name)

    copied_to_source_id = {}

    def pair_module_trees(copied, original, seen):
        pair = (id(copied), id(original))
        if pair in seen:
            return
        seen.add(pair)
        copied_to_source_id[id(copied)] = id(original)
        for name, copied_child in copied._modules.items():
            original_child = original._modules.get(name)
            if copied_child is not None and original_child is not None:
                pair_module_trees(copied_child, original_child, seen)

    seen_pairs = set()
    for copied, original in original_module_copies:
        pair_module_trees(copied, original, seen_pairs)

    graph_to_source = {}
    graph_module_to_source = {}
    graph_state_names_by_graph = {}
    for graph_name, graph_module in (
            ("train_module", train_module),
            ("eval_module", eval_module),
    ):
        for module_name, graph_child in graph_module.named_modules(
                remove_duplicate=False):
            source_id = copied_to_source_id.get(
                id(graph_child),
                id(graph_child),
            )
            source_names = source_module_names_by_id.get(source_id)
            if source_names:
                graph_module_to_source[(graph_name, module_name)] = tuple(
                    source_names
                )

        for state_name, value in _state_objects(graph_module).items():
            source_names = source_names_by_id.get(id(value))
            if source_names:
                graph_to_source[(graph_name, state_name)] = tuple(
                    source_names
                )

        graph_state_names = set(_registered_state(graph_module))
        graph_state_names_by_graph[graph_name] = graph_state_names
        for (mapped_graph_name, module_name), source_names in tuple(
                graph_module_to_source.items()):
            if mapped_graph_name != graph_name:
                continue
            state_name = ".".join(filter(None, (
                module_name,
                "_extra_state",
            )))
            if state_name not in graph_state_names:
                continue
            checkpoint_names = tuple(
                ".".join(filter(None, (source_name, "_extra_state")))
                for source_name in source_names
            )
            checkpoint_names = tuple(
                name for name in checkpoint_names
                if name in source_state_name_set
            )
            if checkpoint_names:
                graph_to_source[(graph_name, state_name)] = checkpoint_names

    mapped_source_names = {
        source_name
        for source_names in graph_to_source.values()
        for source_name in source_names
    }
    # A parameterised replacement can deliberately retain an original public
    # name while owning a different tensor. In that case the graph value is
    # authoritative and the obsolete source value is not source-only state.
    mapped_source_names.update(
        state_name
        for graph_name in ("train_module", "eval_module")
        for state_name in graph_state_names_by_graph[graph_name]
        if (
                (graph_name, state_name) not in graph_to_source
                and state_name in source_state_names
        )
    )
    return (
        source_state_names,
        tuple(
            name for name in source_state_names
            if name not in mapped_source_names
        ),
        graph_to_source,
        graph_module_to_source,
    )


class DualGraphModule(nn.Module):
    def __init__(self, train_module: nn.Module, eval_module: nn.Module,
                 class_name="DualGraphModule", source_module=None,
                 auxiliary_module_names=(), original_module_copies=()):
        super().__init__()

        # This is a per-instance label used by drawing and diagnostics. Renaming
        # ``self.__class__`` would rename every DualGraphModule instance.
        self.graph_name = class_name
        self.train_module = train_module
        self.eval_module = eval_module
        (
            self._source_state_names,
            self._source_only_state_names,
            self._graph_state_to_source,
            self._graph_module_to_source,
        ) = _build_dual_graph_state_mapping(
            source_module,
            train_module,
            eval_module,
            original_module_copies,
        )
        self.__dict__["_torchwatcher_source_module"] = (source_module,)
        self._auxiliary_module_names = frozenset(auxiliary_module_names)
        self.register_load_state_dict_pre_hook(
            _dual_graph_load_state_dict_pre_hook
        )
        self.register_load_state_dict_post_hook(
            _dual_graph_load_state_dict_post_hook
        )

    def state_dict(
            self,
            *args,
            destination=None,
            prefix="",
            keep_vars=False,
    ):
        """Return one logical checkpoint for both implementation graphs."""
        if args:
            warnings.warn(
                "Positional args are being deprecated, use kwargs instead. "
                "Refer to https://pytorch.org/docs/main/generated/torch.nn."
                "Module.html#torch.nn.Module.state_dict for details.",
                FutureWarning,
                stacklevel=2,
            )
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == "":
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = {"version": self._version}
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        for hook in self._state_dict_pre_hooks.values():
            hook(self, prefix, keep_vars)

        self._save_to_state_dict(destination, prefix, keep_vars)
        source_module = self.__dict__["_torchwatcher_source_module"][0]
        if source_module is not None:
            source_module.state_dict(
                destination=destination,
                prefix=prefix,
                keep_vars=keep_vars,
            )
        graph_owned_modules = _run_graph_state_dict_pre_hooks(
            self,
            destination,
            prefix,
            keep_vars,
        )
        _merge_dual_graph_state(
            self,
            destination,
            prefix,
            keep_vars,
        )
        destination = _run_graph_state_dict_post_hooks(
            graph_owned_modules,
            destination,
            prefix,
        )

        for hook in self._state_dict_hooks.values():
            hook_result = hook(
                self,
                destination,
                prefix,
                local_metadata,
            )
            if not getattr(hook, "_from_public_api", False):
                if hook_result is not None:
                    destination = hook_result
            elif hook_result is not None:
                raise RuntimeError("state_dict post-hook must return None")

        return destination

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
        for name, _module, _parameter_name, parameter in (
                self.logical_parameter_registrations()):
            if remove_duplicate and parameter in seen_parameters:
                continue
            seen_parameters.add(parameter)
            yield ".".join(filter(None, (prefix, name))), parameter

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
        for name, _module, _buffer_name, buffer in (
                self._logical_registrations("buffer")):
            if remove_duplicate and buffer in seen_buffers:
                continue
            seen_buffers.add(buffer)
            yield ".".join(filter(None, (prefix, name))), buffer

    def _logical_registrations(self, kind):
        """Yield logical names, owners and values for one registration type."""
        source_module = self.__dict__["_torchwatcher_source_module"][0]
        registration_name = f"_{kind}s"
        graph_registrations = []
        graph_names = set()
        graph_value_ids = set()
        for graph_name, graph_module in (
                ("train_module", self.train_module),
                ("eval_module", self.eval_module),
        ):
            for module_name, owner in graph_module.named_modules(
                    remove_duplicate=False):
                for local_name, value in getattr(
                        owner,
                        registration_name,
                ).items():
                    if value is None:
                        continue
                    state_name = ".".join(filter(None, (
                        module_name,
                        local_name,
                    )))
                    for full_name in _dual_graph_checkpoint_names(
                            self, graph_name, state_name):
                        graph_names.add(full_name)
                        graph_value_ids.add(id(value))
                        graph_registrations.append((
                            full_name,
                            owner,
                            local_name,
                            value,
                        ))

        emitted_names = set()
        source_value_ids = set()
        if source_module is not None:
            for module_name, owner in source_module.named_modules(
                    remove_duplicate=False):
                for local_name, value in getattr(owner, registration_name).items():
                    if value is None:
                        continue
                    full_name = ".".join(filter(None, (
                        module_name,
                        local_name,
                    )))
                    if (
                            full_name in graph_names
                            and id(value) not in graph_value_ids
                    ):
                        continue
                    emitted_names.add(full_name)
                    source_value_ids.add(id(value))
                    yield full_name, owner, local_name, value

        for full_name, owner, local_name, value in graph_registrations:
            if (
                    id(value) in source_value_ids
                    or full_name in emitted_names
            ):
                continue
            emitted_names.add(full_name)
            yield full_name, owner, local_name, value

    def logical_parameter_registrations(self):
        """Yield parameter registrations in the original model namespace."""
        yield from self._logical_registrations("parameter")

    def _apply(self, fn, recurse=True):
        """Apply a tensor conversion to graph and retained source state.

        FX omits registrations which are not used by either forward graph.
        Those registrations remain part of the public module lifecycle, so
        convert them separately and keep every original/graph alias tied to
        the same converted object.
        """
        if not recurse:
            return super()._apply(fn, recurse=False)

        source_module = self.__dict__["_torchwatcher_source_module"][0]

        def registration_groups(kind):
            registration_name = f"_{kind}s"
            groups = {}

            def add(root, location):
                if root is None:
                    return
                seen_slots = set()
                for _module_name, owner in root.named_modules(
                        remove_duplicate=False):
                    registrations = getattr(owner, registration_name)
                    for local_name, value in registrations.items():
                        if value is None:
                            continue
                        slot = (id(owner), local_name)
                        if slot in seen_slots:
                            continue
                        seen_slots.add(slot)
                        group = groups.setdefault(id(value), {
                            "value": value,
                            "source": [],
                            "graph": [],
                        })
                        group[location].append((owner, local_name))

            add(source_module, "source")
            add(self.train_module, "graph")
            add(self.eval_module, "graph")
            return groups

        parameter_groups = registration_groups("parameter")
        buffer_groups = registration_groups("buffer")

        super()._apply(fn, recurse=True)

        # Graph conversion can replace buffers (and, depending on PyTorch's
        # future flags, parameters). Reconnect source and graph aliases to one
        # canonical converted registration.
        for kind, groups in (
                ("parameter", parameter_groups),
                ("buffer", buffer_groups),
        ):
            registrations_name = f"_{kind}s"
            for group in groups.values():
                if not group["graph"]:
                    continue
                owner, local_name = group["graph"][0]
                converted = getattr(owner, registrations_name)[local_name]
                for alias_owner, alias_name in (
                        group["source"] + group["graph"]):
                    getattr(alias_owner, registrations_name)[alias_name] = (
                        converted
                    )

        # Let nn.Module implement the exact parameter, gradient and buffer
        # conversion semantics for values which FX omitted entirely.
        source_only = nn.Module()
        parameter_keys = {}
        buffer_keys = {}
        for index, group in enumerate(parameter_groups.values()):
            if group["graph"]:
                continue
            key = f"parameter_{index}"
            source_only._parameters[key] = group["value"]
            parameter_keys[key] = group
        for index, group in enumerate(buffer_groups.values()):
            if group["graph"]:
                continue
            key = f"buffer_{index}"
            source_only._buffers[key] = group["value"]
            buffer_keys[key] = group

        nn.Module._apply(source_only, fn, recurse=False)

        for key, group in parameter_keys.items():
            converted = source_only._parameters[key]
            for owner, local_name in group["source"]:
                owner._parameters[local_name] = converted
        for key, group in buffer_keys.items():
            converted = source_only._buffers[key]
            for owner, local_name in group["source"]:
                owner._buffers[local_name] = converted

        return self

    def apply(self, fn: Callable[[nn.Module], None]):
        """Apply ``fn`` once to each logical or source-only module.

        Train and eval graphs contain distinct module objects whose parameters
        and buffers can represent the same original module. Traversing both
        registered children normally would therefore apply initialisers and
        other mutations more than once.
        """
        modules = []
        seen_names = set()
        seen_modules = set()
        seen_registrations = set()
        source_module = self.__dict__["_torchwatcher_source_module"][0]
        roots = tuple(
            root
            for root in (
                self.train_module,
                self.eval_module,
                source_module,
            )
            if root is not None
        )
        for graph_module in roots:
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
    """Reject mixed modes which would be specialised during FX tracing.

    A leaf module retains its training flag as runtime module state. In
    contrast, a module which FX traces through can use ``self.training`` in
    Python control flow, permanently specialising the resulting graph. Two
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
    optimiser state and mutable buffers such as BatchNorm running statistics.
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
                        training: bool,
                        original_module_copies=None) -> str:
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

    if (
            original_module_copies is not None
            and original_module is not None
            and hasattr(interjection, '_wrapped')
    ):
        original_module_copies.append((
            graph_module.get_submodule(node.target),
            original_module,
        ))

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

    return interjection_name


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
        auxiliary_module_names = set()
        original_module_copies = []
        for mode in ["train", "eval"]:
            if mode == "train":
                model.train()
            elif mode == "eval":
                model.eval()

            traced = symbolic_trace(model, tracer_kwargs=tracer_kwargs)
            for node in traced.graph.nodes:
                if selector is not None and selector.fn((traced, node)):
                    interjection_name = insert_interjection(
                        traced,
                        node,
                        interjection,
                        training=mode == "train",
                        original_module_copies=original_module_copies,
                    )
                    auxiliary_module_names.add(interjection_name)

            traced.graph.eliminate_dead_code()
            traced.delete_all_unused_submodules()
            traced.recompile()
            graphmodules[mode] = traced

        # Build the final graph module
        graph_module = DualGraphModule(
            graphmodules["train"],
            graphmodules["eval"],
            class_name="DualGraphModule_" + _get_name(model),
            source_module=model,
            auxiliary_module_names=auxiliary_module_names,
            original_module_copies=original_module_copies,
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

    def add_interjection(name, module):
        native_call = _NativeCall(module)

        # Each alias gets its own call module so names and backward hooks stay
        # distinct, while every alias wrapper exposes the same original module.
        interjection.register(name, native_call)
        interjection.register_original_module(name, module)
        return _NativeInterjectionWrapper(name, module, interjection)

    traversed: set[int] = set()
    contains_interjection: dict[int, bool] = {}

    def add_recursively(
            module: nn.Module,
            name: str = "",
            *,
            is_root: bool = False,
    ) -> tuple[nn.Module, bool]:
        module_id = id(module)
        is_target = isinstance(module, target_module_class)
        subtree_contains_interjection = is_target
        if module_id not in traversed:
            traversed.add(module_id)
            for child_name, child in list(module._modules.items()):
                if child is None:
                    continue
                qualified_name = (
                    f"{name}.{child_name}" if name else child_name
                )
                replacement, child_contains_interjection = add_recursively(
                    child,
                    qualified_name,
                )
                subtree_contains_interjection |= child_contains_interjection
                if replacement is not child:
                    setattr(module, child_name, replacement)
            contains_interjection[module_id] = subtree_contains_interjection
        else:
            subtree_contains_interjection = contains_interjection.get(
                module_id,
                is_target,
            )

        if not is_target:
            if (
                    subtree_contains_interjection
                    and not isinstance(
                        module.__dict__.get("apply"),
                        _NativeApply,
                    )
            ):
                module.apply = _NativeApply(module)
            return module, subtree_contains_interjection
        observation_name = root_name if is_root else name
        return (
            add_interjection(observation_name, module),
            True,
        )

    model, _ = add_recursively(model, is_root=True)
    return model
