from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeAlias
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from .operations import Add, Divide, MatrixMultiply, Multiply, Operation, Subtract

if TYPE_CHECKING:
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


if __name__ == "__main__":
    a_data = np.array([[1, 2, 3], [3, 1, 4]])
    b_data = np.array([[2, 3, 4], [1, 7, 5]])
    a = Tensor(a_data, require_grad=True)
    b = Tensor(b_data, require_grad=True)
    c = 3 * a + b

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
    

    j.backward()
    from .visualization import visualize

    fig, ax = visualize(j)

    fig.savefig(r"./tests/figs/test5.png")
    plt.show()
