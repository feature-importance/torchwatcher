import hashlib
from itertools import chain
from typing import Any, Optional
from collections import Counter

import pydot
import torch
from torch.fx.passes.graph_drawer import FxGraphDrawer, _WEIGHT_TEMPLATE, _COLOR_MAP, _HASH_COLOR_MAP
from torch.fx.passes.shape_prop import TensorMetadata
from torchwatcher.interjection.tracing import DualGraphModule, trace_shapes


def compact_list_repr(x: list[Any]) -> str:
    '''returns more compact representation of list with
    repeated elements. This is useful for e.g. output of transformer/rnn
    models where hidden state outputs shapes is repetation of one hidden unit
    output'''

    list_counter = Counter(x)
    x_repr = ''

    for elem, cnt in list_counter.items():
        if cnt == 1:
            x_repr += f'{elem}, '
        else:
            x_repr += f'{cnt} x {elem}, '

    # get rid of last comma
    return x_repr[:-2]


class SubmoduleFxGraphDrawer(FxGraphDrawer):
    """
    FxGraphDrawer extended to show the submodules of the original code as traced by the torchwatcher.
    """

    def _get_node_style(self, node: torch.fx.Node) -> dict[str, str]:
        template = {
            "shape": self.dot_graph_shape,
            "fillcolor": "#CAFFE3",
            "style": '"filled,rounded"',
            "fontcolor": "#000000",
            "fontsize": "10"
        }
        if node.op in _COLOR_MAP:
            template["fillcolor"] = _COLOR_MAP[node.op]
        else:
            # Use a random color for each node; based on its name so it's stable.
            target_name = node._pretty_print_target(node.target)
            target_hash = int(
                hashlib.md5(
                    target_name.encode(), usedforsecurity=False
                ).hexdigest()[:8],
                16,
            )
            template["fillcolor"] = _HASH_COLOR_MAP[
                target_hash % len(_HASH_COLOR_MAP)
                ]
        return template

    def _to_dot(
            self,
            graph_module: torch.fx.GraphModule,
            name: str,
            ignore_getattr: bool,
            ignore_parameters_and_buffers: bool,
            skip_node_names_in_args: bool,
            parse_stack_trace: bool,
    ) -> pydot.Dot:
        dot_graph = pydot.Dot(name, rankdir="TB")

        sub_name_to_subgraph = {}

        for node in graph_module.graph.nodes:
            if ignore_getattr and node.op == "get_attr":
                continue

            style = self._get_node_style(node)
            dot_node = pydot.Node(
                node.name,
                label=self._get_node_label(
                    graph_module, node, skip_node_names_in_args, parse_stack_trace
                ),
                **style,  # type: ignore[arg-type]
            )

            current_graph = dot_graph

            sub_meta = node.meta.get("nn_module_stack", None)
            if sub_meta is not None:
                if node.op == 'call_module':
                    sub_meta = list(sub_meta.values())[:-1]
                else:
                    sub_meta = list(sub_meta.values())

                for sub_name, sub_clz in sub_meta:
                    if sub_name not in sub_name_to_subgraph:
                        sub_name_to_subgraph[sub_name] = pydot.Cluster(
                            sub_name, label=sub_clz.__name__
                        )
                        current_graph.add_subgraph(sub_name_to_subgraph[sub_name])
                    current_graph = sub_name_to_subgraph[sub_name]

            current_graph.add_node(dot_node)

            def get_module_params_or_buffers():
                for pname, ptensor in chain(
                        leaf_module.named_parameters(), leaf_module.named_buffers()
                ):
                    pname1 = node.name + "." + pname
                    label1 = (
                        pname1 + "|op_code=get_" + "parameter"
                        if isinstance(ptensor, torch.nn.Parameter)
                        else "buffer" + r"\l"
                    )
                    dot_w_node = pydot.Node(
                        pname1,
                        label="{" + label1 + self._get_tensor_label(ptensor) + "}",
                        **_WEIGHT_TEMPLATE,  # type: ignore[arg-type]
                    )
                    current_graph.add_node(dot_w_node)
                    current_graph.add_edge(pydot.Edge(pname1, node.name))

            if node.op == "call_module":
                leaf_module = self._get_leaf_node(graph_module, node)

                if not ignore_parameters_and_buffers and not isinstance(
                        leaf_module, torch.fx.GraphModule
                ):
                    get_module_params_or_buffers()

        for subgraph in sub_name_to_subgraph.values():
            subgraph.set("penwidth", "1")
            subgraph.set("style", "dashed")
            subgraph.set("labeljust", "l")
            subgraph.set("fontsize", "10")

        for node in graph_module.graph.nodes:
            if ignore_getattr and node.op == "get_attr":
                continue

            for user in node.users:
                dot_graph.add_edge(pydot.Edge(node.name, user.name))

        return dot_graph


def _sanitize_shapes(meta) -> str:
    if isinstance(meta['tensor_meta'], dict):
        return [tuple(v.shape) for v in meta['tensor_meta'].values()]
    elif isinstance(meta['tensor_meta'], (list, tuple)) and not isinstance(meta['tensor_meta'], TensorMetadata):
        return [tuple(v.shape) for v in meta['tensor_meta']]
    else:
        return [tuple(meta['tensor_meta'].shape)]


class ShapeFxGraphDrawer(SubmoduleFxGraphDrawer):
    def __init__(
            self,
            graph_module: torch.fx.GraphModule,
            name: str,
            show_shapes: bool = True,
            show_node_names: bool = False,
            show_extra_info: bool = False,
    ):
        self.html_config = {
            'border': 0,
            'cell_border': 1,
            'cell_spacing': 0,
            'cell_padding': 4,
            'col_span': 2,
            'row_span': 2,
        }
        self.show_shapes = show_shapes
        self.show_node_names = show_node_names
        self.show_extra_info = show_extra_info

        super().__init__(graph_module,
                         name,
                         ignore_getattr=True,
                         ignore_parameters_and_buffers=True,
                         skip_node_names_in_args=True
                         )

    def _get_node_label(
            self,
            module: torch.fx.GraphModule,
            node: torch.fx.Node,
            *args,
            **kwargs
    ) -> str:
        '''Returns html-like format for the label of node. This html-like
        label is based on Graphviz API for html-like format. For setting of node label
        it uses graph config and html_config.'''
        input_str = 'input'
        output_str = 'output'
        border = self.html_config['border']
        cell_sp = self.html_config['cell_spacing']
        cell_pad = self.html_config['cell_padding']
        cell_bor = self.html_config['cell_border']

        def name_row(cols):
            return f'''<TR>
            <TD COLSPAN="{cols}">{node.qualified_name if hasattr(node, 'qualified_name') else node.name}
            </TD></TR> ''' if self.show_node_names else ''

        def extra_info_row(cols):
            if not self.show_extra_info:
                return ''
            if node.op == 'call_module':
                leaf_module = self._get_leaf_node(module, node)
            else:
                return ''

            extra = ""
            if hasattr(leaf_module, "__constants__"):
                extra = r"\n".join(
                    [
                        f"<TR><TD COLSPAN='{cols}'>{c}: {getattr(leaf_module, c)}</TD></TR>"
                        for c in leaf_module.__constants__
                    ]
                )
            return extra

        if node.op == 'call_module':
            leaf_module = self._get_leaf_node(module, node)
            name = leaf_module.__class__.__name__
        elif node.op == 'call_function':
            name = node.target.__name__
        else:
            name = node.name

        if self.show_shapes:
            if node.op != 'call_module' and node.op != 'call_function':
                tensor_shape = compact_list_repr(_sanitize_shapes(node.meta))
                label = f'''<
                            <TABLE BORDER="{border}" CELLBORDER="{cell_bor}"
                            CELLSPACING="{cell_sp}" CELLPADDING="{cell_pad}">
                                {name_row(2)}
                                <TR><TD><BR/>{name}<BR/>&nbsp;</TD><TD>{tensor_shape}</TD></TR>
                                {extra_info_row(2)}
                            </TABLE>>'''
            else:
                inputs = []
                for n in node.all_input_nodes:
                    inputs.extend(_sanitize_shapes(n.meta))
                input_repr = compact_list_repr(inputs)
                output_repr = compact_list_repr(_sanitize_shapes(node.meta))
                label = f'''<
                            <TABLE BORDER="{border}" CELLBORDER="{cell_bor}"
                            CELLSPACING="{cell_sp}" CELLPADDING="{cell_pad}">
                            {name_row(5)}
                            <TR>
                                <TD ROWSPAN="2">{name}</TD>
                                <TD COLSPAN="2">{input_str}:</TD>
                                <TD COLSPAN="2">{input_repr} </TD>
                            </TR>
                            <TR>
                                <TD COLSPAN="2">{output_str}: </TD>
                                <TD COLSPAN="2">{output_repr} </TD>
                            </TR>
                            {extra_info_row(5)}
                            </TABLE>>'''
        else:
            label = f'''<
                            <TABLE BORDER="{border}" CELLBORDER="{cell_bor}"
                            CELLSPACING="{cell_sp}" CELLPADDING="{cell_pad}">
                                {name_row(1)}
                                <TR><TD>{name}</TD></TR>
                                {extra_info_row(1)}
                            </TABLE>>'''
        return label

    def _get_node_style(self, node: torch.fx.Node) -> dict[str, str]:
        template = {
            "shape": 'plaintext',
            "fillcolor": "#CAFFE3",
            "style": '"filled"',
            'margin': '0',
            "fontcolor": "#000000",
            "fontsize": "10"
        }
        if node.op in _COLOR_MAP:
            template["fillcolor"] = _COLOR_MAP[node.op]
        else:
            # Use a random color for each node; based on its name so it's stable.
            target_name = node._pretty_print_target(node.target)
            target_hash = int(
                hashlib.md5(
                    target_name.encode(), usedforsecurity=False
                ).hexdigest()[:8],
                16,
            )
            template["fillcolor"] = _HASH_COLOR_MAP[
                target_hash % len(_HASH_COLOR_MAP)
                ]
        return template


def draw_graph(
        module: DualGraphModule,
        ignore_getattr: bool = True,
        ignore_parameters_and_buffers: bool = True,
        skip_node_names_in_args: bool = True,
        parse_stack_trace: bool = False,
        dot_graph_shape: Optional[str] = None,
        normalize_args: bool = False):

    return SubmoduleFxGraphDrawer(module.graphmodule, module.__class__.__name__, ignore_getattr,
                                  ignore_parameters_and_buffers, skip_node_names_in_args,
                                  parse_stack_trace, dot_graph_shape, normalize_args).get_dot_graph()


def draw_graph_pretty(
        module: DualGraphModule,
        *inputs,
        show_shapes: bool = True,
        show_node_names: bool = False,
        show_extra_info: bool = False,
        ):
    if show_shapes:
        trace_shapes(module, *inputs)

    return ShapeFxGraphDrawer(module.graphmodule,
                              module.__class__.__name__,
                              show_shapes,
                              show_node_names,
                              show_extra_info).get_dot_graph()
