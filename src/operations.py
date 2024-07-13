from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from .tensor import Tensor


class Operation(ABC):
    """
    Abstract base class for all operations on tensors.

    Attributes:
        inputs: A tuple of input tensors for the operation.
        uid: A unique id that represents the operation.
    """

    inputs: tuple[Tensor, ...]
    uid: str

    def __init__(self) -> None:
        self.uid = str(uuid4())

    def __call__(self, *inputs: Tensor) -> Tensor:
        """
        Calls the operation and computes the forward pass.

        Args:
            inputs: A variable number of Tensor objects to perform the operation on.

        Returns:
            A new Tensor resulting from the operation.
        """
        from .tensor import Tensor  # avoid circular import

        self.inputs = inputs
        outputs = self.forward(*[t.data for t in inputs])
        result = Tensor(outputs, require_grad=any(t.require_grad for t in inputs))

        if result.require_grad:

            def backward_fn(input_grad: NDArray) -> None:
                """
                Computes and propagates gradients through the
                computational graph.

                Args:
                    input_grad (NDArray): The gradient flowing back
                        from subsequent operations in the computational graph.
                """
                grads = self.backward(input_grad)
                for tensor, grad in zip(inputs, grads):
                    if tensor.require_grad:
                        tensor.backward(grad)

            result._backward_fn = backward_fn
            result._creator_operation = self

        return result

    @abstractmethod
    def forward(self, *args: NDArray) -> NDArray:
        """
        Computes the forward pass of the operation.

        Args:
            args (NDArray): A variable number of NDArray representing
                the inputs data.

        Returns:
            NDArray: result of the operation.
        """
        pass

    @abstractmethod
    def backward(self, grad: NDArray) -> tuple[NDArray, ...]:
        """
        Computes the backward pass of the operation.

        Args:
            grad (NdArray): An NDArray of the gradient of the output.

        Returns:
            tuple[NDArray, ...]: A tuple of NDArrays representing the
                gradients of the inputs.
        """
        pass


class Add(Operation):
    """The element-wise addition of two NDArrays."""

    def forward(self, a: NDArray, b: NDArray) -> NDArray:
        return a + b

    def backward(self, grad: NDArray) -> tuple[NDArray, NDArray]:
        return grad, grad


class Subtract(Operation):
    """The element-wise subtraction of two NDArrays."""

    def forward(self, a: NDArray, b: NDArray) -> NDArray:
        return a - b

    def backward(self, grad: NDArray) -> tuple[NDArray, NDArray]:
        return grad, -grad


class Multiply(Operation):
    """The element-wise multiplication of two NDArrays."""

    def forward(self, a: NDArray, b: NDArray) -> NDArray:
        return a * b

    def backward(self, grad: NDArray) -> tuple[NDArray, NDArray]:
        a, b = self.inputs
        return grad * b.data, grad * a.data


class Divide(Operation):
    """The element-wise division of two NDArrays, avoiding division by zero."""

    def forward(self, a: NDArray, b: NDArray) -> NDArray:
        # avoid dividing by 0
        epsilon = 1e-5
        safe_b = np.where(b != 0, b, epsilon)
        return a / safe_b

    def backward(self, grad: NDArray) -> tuple[NDArray, NDArray]:
        a, b = self.inputs
        return grad / b.data, -grad * a.data / (b.data**2)


class MatrixMultiply(Operation):
    """The matrix multiplication of two NDArrays."""

    def forward(self, a: NDArray, b: NDArray) -> NDArray:
        return np.matmul(a, b)

    def backward(self, grad: NDArray) -> tuple[NDArray, NDArray]:
        a, b = self.inputs
        return np.matmul(grad, b.data.T), np.matmul(a.data.T, grad)


class Transpose(Operation):
    """The transope operation of the given NDArray."""

    def forward(self, a: NDArray) -> NDArray:
        return a.T

    def backward(self, grad: NDArray) -> tuple[NDArray]:
        return (grad.T,)
