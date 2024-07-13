from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeAlias
from uuid import uuid4

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from .operations import Add, Divide, MatrixMultiply, Multiply, Operation, Subtract

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray


Scalar: TypeAlias = int | float


class Tensor:
    """
    A simple implementation of a Tensor for automatic differentiation.

    Attributes:
        data: The data of the tensor.
        grad: The gradient of the tensor.
        require_grad: Indicates whether the tensor requires
            gradient computation.
        uid: A unique id that represents the tensor.
        _creator_operation: The operation that creates this tensor.
        _backward_fn: The function to compute the gradient backward pass.
            This depends on how the tensor was created.
    """

    data: NDArray
    grad: ArrayLike | None
    uid: str
    require_grad: bool
    _creator_operation: Operation | None
    _backward_fn: Callable[[ArrayLike], None] | None

    def __init__(self, data: ArrayLike, require_grad=False) -> None:
        """
        Initializes the Tensor with data and optionally sets require_grad.

        Args:
            data: The data to initialize the tensor.
            require_grad: Whether to track gradients for this tensor.
                Default is False.
        """
        self.data = np.array(data)
        self.grad = None
        self.uid = str(uuid4())
        self.require_grad = require_grad
        self._creator_operation = None
        self._backward_fn = None

    def backward(self, grad: ArrayLike | None = None) -> None:
        """
        Computes the gradient of the tensor.

        Args:
            grad: The gradient to be backpropagated. If None, a tensor of ones with the same shape
                  as data is used. Default is None.
        """
        if not self.require_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self._backward_fn:
            self._backward_fn(self.grad)

    def __str__(self) -> str:
        """Returns a string representation of the tensor."""
        data_str = np.array2string(self.data, prefix=" " * 7, separator=", ")
        grad_str = (
            "None"
            if self.grad is None
            else np.array2string(self.grad, prefix=" " * 7, separator=", ")
        )

        return (
            f"Tensor(\n"
            f"  data={data_str},\n"
            f"  grad={grad_str},\n"
            f"  shape={self.data.shape},\n"
            f"  dtype={self.data.dtype},\n"
            f"  require_grad={self.require_grad}\n"
            ")"
        )

    def __add__(self, other: Tensor | Scalar) -> Tensor:
        """Add a Tensor or Scalar to this Tensor."""
        if isinstance(other, Scalar):
            other = Tensor(other)
        return Add()(self, other)

    def __radd__(self, other) -> Tensor:
        """
        Add a Tensor or Scalar to this Tensor.

        __radd__ simply calls __add__ to ensure commutative property.
        """
        return self.__add__(other)

    def __sub__(self, other: Tensor | Scalar) -> Tensor:
        """Subtract a Tensor or Scalar to this Tensor."""
        if isinstance(other, Scalar):
            other = Tensor(other)
        return Subtract()(self, other)

    def __mul__(self, other: Tensor | Scalar) -> Tensor:
        """Multiply a Tensor or Scalar to this Tensor."""
        if isinstance(other, Scalar):
            other = Tensor(other)
        return Multiply()(self, other)

    def __rmul__(self, other: Tensor | Scalar) -> Tensor:
        """
        Multiply a Tensor or Scalar to this Tensor.

        __rmul__ simply calls __mul__ to ensure commutative property.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Tensor | Scalar) -> Tensor:
        """Divide this Tensor by a Tensor or Scalar."""
        if isinstance(other, Scalar):
            other = Tensor(other)
        return Divide()(self, other)

    def __matmul__(self, other: Tensor) -> Tensor:
        """
        Perform matrix multiplication between two Tensors.

        A.__matmul__(B) --> A @ B

        Args:
            other (Tensor): another Tensor as right operand
        """
        return MatrixMultiply()(self, other)


def visualize(tensor: Tensor) -> tuple[Figure, Axes]:
    """
    Visualizes the computation graph for a given tensor.

    Args:
        tensor (Tensor): The starting point for building
            the computation graph.

    Returns:
        tuple[Figure, Axes]: A tuple containing the matplotlib
            Figure and Axes objects of the visualization.
    """
    G = nx.DiGraph()
    visited = set()
    node_labels = {}

    def build_computation_graph(tensor: Tensor) -> None:
        """
        Recursively builds a computation graph starting from the
        given tensor.

        Args:
            tensor (Tensor): The starting tensor for building the graph
        """
        if tensor.uid in visited:
            return

        visited.add(tensor.uid)
        G.add_node(tensor.uid)
        node_labels[tensor.uid] = str(
            np.array2string(tensor.data, prefix="  ", separator=", ")
        )

        if tensor._creator_operation:
            inputs = tensor._creator_operation.inputs
            for input_tensor in inputs:
                build_computation_graph(input_tensor)

                op_uid = tensor._creator_operation.uid
                G.add_node(op_uid)
                node_labels[op_uid] = str(tensor._creator_operation.__class__.__name__)
                G.add_edge(op_uid, tensor.uid)
                G.add_edge(input_tensor.uid, op_uid)

    build_computation_graph(tensor)

    # calculate the size of the figure
    pos: dict = nx.bfs_layout(G.reverse(), tensor.uid, align="horizontal", scale=(2, 4))
    pos_array = np.array(list(pos.values()))
    x_min, y_min = pos_array.min(axis=0)
    x_max, y_max = pos_array.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    margin = 0.1
    fig_width = width + 2 * margin
    fig_height = height + 2 * margin

    # create fig and ax, adjust subplot position
    fig, ax = plt.subplots(figsize=(fig_width * 5, fig_height * 5), dpi=300)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02)

    # some adaptive parameters
    node_size = min(
        10_000, max(3000, 70_000 / (len(G) ** 0.9))
    )  # 3000 < node_size < 10_000
    arrowsize = node_size // 250
    font_size = node_size // 250

    # draw nodes and edges
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color="lightblue",
        node_shape="s",
        node_size=node_size,
        arrows=True,
        arrowsize=arrowsize,
        ax=ax,
    )

    # draw node labels
    nx.draw_networkx_labels(
        G,
        pos,
        node_labels,
        font_size=font_size,
        font_weight="bold",
        ax=ax,
    )

    # set title and margins
    ax.set_title("Computation Graph", fontsize=2 * font_size)
    ax.axis("off")
    ax.margins(0.1, 0.05)

    return fig, ax


if __name__ == "__main__":
    a_data = np.array([[1, 2, 3], [3, 1, 4]])
    b_data = np.array([[2, 3, 4], [1, 7, 5]])
    a = Tensor(a_data, require_grad=True)
    b = Tensor(b_data, require_grad=True)
    c = a + b

    constant1_data = np.array([2, 3, 4])  # broadcast: (3,) -> (2, 3)
    constant1 = Tensor(constant1_data, require_grad=False)
    d = c * constant1

    constant2_data = np.array([[2, 2], [3, 9], [4, 7]])
    constant2 = Tensor(constant2_data, require_grad=False)
    e = d @ constant2

    f = Tensor(np.array([[2, 3], [1, 7]]), require_grad=True)
    g = Tensor(np.array([[22, 1], [6, 3]]), require_grad=True)
    h = f * g + f
    j = e @ h

    fig, ax = visualize(j)

    fig.savefig(r"./tests/figs/test3.png")
    plt.show()
