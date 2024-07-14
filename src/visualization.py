from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from networkx import DiGraph
    from numpy.typing import NDArray
    from .tensor import Tensor


def visualize(tensor: Tensor, save_path: str | None = None) -> tuple[Figure, Axes]:
    """
    Visualizes the computation graph for a given tensor.

    Args:
        tensor (Tensor): The starting point for building the
            computation graph.
        save_path (str | None): The path to save the figure.
            If None, the figure is not saved.

    Returns:
        tuple[Figure, Axes]: A tuple containing the matplotlib Figure
            and Axes objects of the visualization.
    """
    G = build_computation_graph(tensor)
    pos = calculate_layout(G, tensor)
    fig, ax = create_figure(pos)
    draw_graph(G, pos, ax, shape_scale=tensor.data.shape[0])

    if save_path is not None:
        fig.savefig(save_path)

    return fig, ax


def build_computation_graph(tensor: Tensor) -> DiGraph:
    """
    Builds the computation graph starting from the given tensor.

    Args:
        tensor (Tensor): The starting tensor for building the graph.

    Returns:
        DiGraph: A networkx.Digraph object representing the computation
            graph.
    """
    G = nx.DiGraph(flow_direction="forward")
    visited = set()

    def add_nodes_and_edges(t: Tensor) -> None:
        """
        Recursively adds nodes and edges to the graph.

        Args:
            t (Tensor): The current tensor being processed.
        """
        if t.uid in visited:
            return

        visited.add(t.uid)
        G.add_node(
            t.uid, label=str(np.array2string(t.data, prefix="  ", separator=", "))
        )

        if not t._creator_operation:  # leaf nodes of computation graph
            return

        op = t._creator_operation
        G.add_node(op.uid, label=op.__class__.__name__)

        for input_tensor in op.inputs:
            add_nodes_and_edges(input_tensor)  # build graph recursively
            G.add_edge(input_tensor.uid, op.uid)
        G.add_edge(op.uid, t.uid)

        if t.grad is not None:  # add grads flow (if available) as edge labels
            G.graph["flow_direction"] = "backward"

            for input_tensor, grad in zip(op.inputs, op.backward(t.grad)):
                edge_label = (
                    np.array2string(grad, prefix="  ", separator=", ")
                    if input_tensor.require_grad
                    else "not require grad"
                )
                G.edges[input_tensor.uid, op.uid]["label"] = edge_label

            G.edges[op.uid, t.uid]["label"] = np.array2string(
                t.grad, prefix="  ", separator=", "
            )

    add_nodes_and_edges(tensor)

    return G


def calculate_layout(G: DiGraph, tensor: Tensor) -> dict[str, NDArray]:
    """
    Calculates the layout for the graph.

    Args:
        G (DiGraph): The computation graph.
        tensor (Tensor): The tensor used to determine the scale,
            and the starting node for `nx.bfs_layout`.

    Returns:
        dict[str, NDArray]: A dict mapping tensor's uid to their positions.
    """
    pos_scale = 2 * tensor.data.shape[0]
    return nx.bfs_layout(
        G.reverse(), tensor.uid, align="horizontal", scale=(pos_scale, 2 * pos_scale)
    )


def create_figure(pos: dict[str, NDArray]) -> tuple[Figure, Axes]:
    """
    Creates the figure and axes for the graph based on the layout
    positions for the graph nodes.

    Args:
        G (DiGraph): The computation graph.
        pos (dict): The layout positions for the graph nodes.

    Returns:
        tuple[Figure, Axes]: A tuple containing the matplotlib Figure
            and Axes objects.
    """
    pos_array = np.array(list(pos.values()))
    x_min, y_min = pos_array.min(axis=0)
    x_max, y_max = pos_array.max(axis=0)
    width, height = x_max - x_min, y_max - y_min
    margin = 0.05
    fig_width, fig_height = width + 2 * margin, height + 2 * margin

    fig, ax = plt.subplots(figsize=(fig_width * 5, fig_height * 5), dpi=300)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)

    return fig, ax


def draw_graph(G: DiGraph, pos: dict, ax: Axes, shape_scale: int) -> None:
    """
    Adjusts the visual elements based on the graph size and tensor
    shapes. Draws the computational graph on the given matplotlib axes.

    Args:
        G (DiGraph): The computation graph to be visualized.
        pos (dict): The layout positions for the graph nodes.
        ax (Axes): The matplotlib Axes object to draw on.
        shape_scale (int): A scaling factor for node sizes based on
            tensor shape.
    """
    # 1500 < node_size/shape_scale < 5000
    node_size = shape_scale * min(5000, max(1500, 35000 / (len(G) ** 0.9)))
    line_width = node_size / 1500
    arrowsize = node_size / 250
    font_size = node_size / 250

    # draw nodes and edges
    nx.draw(
        G.reverse() if G.graph.get("flow_direction") == "backward" else G,
        pos,
        with_labels=False,
        node_color="lightblue",
        node_shape="s",
        node_size=node_size,
        width=line_width,
        style="--" if G.graph.get("flow_direction") == "backward" else "-",
        arrows=True,
        arrowsize=arrowsize,
        ax=ax,
    )

    # draw node labels (tensors' data)
    node_labels = {n: attr["label"] for n, attr in G.nodes(data=True)}
    nx.draw_networkx_labels(
        G,
        pos,
        node_labels,
        font_size=font_size,
        font_weight="bold",
        ax=ax,
    )

    # draw edge labels (grads flow, if available)
    edge_labels = {(v, u): attr.get("label", "") for u, v, attr in G.edges(data=True)}
    grad_font_scale = 1.1
    nx.draw_networkx_edge_labels(
        G.reverse(),
        pos,
        edge_labels,
        font_size=font_size / grad_font_scale,
        font_weight="bold",
        ax=ax,
        rotate=False,
    )

    # set title and margins
    ax.set_title("Computation Graph", fontsize=2 * font_size, fontweight="bold")
    ax.axis("off")
    ax.margins(0.1, 0.05)
