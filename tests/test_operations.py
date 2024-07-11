import unittest

import numpy as np
from src.operations import Add, Divide, MatrixMultiply, Multiply, Subtract
from src.tensor import Tensor


class TestOperations(unittest.TestCase):
    def test_add(self):
        t1 = Tensor([1, 2, 3], require_grad=True)
        t2 = Tensor([4, 5, 6], require_grad=True)
        res = Add()(t1, t2)
        # test forward
        self.assertTrue(np.array_equal(res.data, np.array([5, 7, 9])))

        # test backward
        res.backward()
        self.assertTrue(np.array_equal(t1.grad, np.array([1, 1, 1])))
        self.assertTrue(np.array_equal(t2.grad, np.array([1, 1, 1])))

    def test_subtract(self):
        t1 = Tensor([1, 2, 3], require_grad=True)
        t2 = Tensor([4, 5, 6], require_grad=True)
        res = Subtract()(t2, t1)
        # test forward
        self.assertTrue(np.array_equal(res.data, np.array([3, 3, 3])))

        # test backward
        res.backward()
        self.assertTrue(np.array_equal(t1.grad, np.array([-1, -1, -1])))
        self.assertTrue(np.array_equal(t2.grad, np.array([1, 1, 1])))

    def test_multiply(self):
        t1 = Tensor([1, 2, 3], require_grad=True)
        t2 = Tensor([4, 5, 6], require_grad=True)
        res = Multiply()(t1, t2)
        # test forward
        self.assertTrue(np.array_equal(res.data, np.array([4, 10, 18])))

        # test backward
        res.backward(np.array([1, 2, 1]))
        self.assertTrue(np.array_equal(t1.grad, np.array([4, 10, 6])))
        self.assertTrue(np.array_equal(t2.grad, np.array([1, 4, 3])))

    def test_divide(self):
        t1 = Tensor([1, 2, 3], require_grad=True)
        t2 = Tensor([4, 5, 6], require_grad=True)
        res = Divide()(t2, t1)
        # test forward
        self.assertTrue(np.array_equal(res.data, np.array([4.0, 2.5, 2.0])))

        # test backward
        res.backward()
        self.assertTrue(
            np.array_equal(
                t1.grad, np.array([-4.0 / (1**2), -5.0 / (2**2), -6.0 / (3**2)])
            )
        )
        self.assertTrue(np.array_equal(t2.grad, np.array([1.0 / 1, 1.0 / 2, 1.0 / 3])))

    def test_matmul(self):
        d1 = np.array([[1, 2, 3], [23, 6, 5], [2, 3, 1]])
        d2 = np.array([[2, 57, 4], [1, 11, 1], [7, 12, 1]])
        t1 = Tensor(d1, require_grad=True)
        t2 = Tensor(d2, require_grad=True)
        res = MatrixMultiply()(t1, t2)
        # test forward
        self.assertTrue(np.array_equal(res.data, np.matmul(d1, d2)))

        # test backward
        res.backward()
        self.assertTrue(np.array_equal(t1.grad, np.matmul(np.ones_like(d1), d2.T)))
        self.assertTrue(np.array_equal(t2.grad, np.matmul(d1.T, np.ones_like(d1))))
