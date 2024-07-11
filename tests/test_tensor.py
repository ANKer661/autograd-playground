import unittest

import numpy as np
from src.tensor import Tensor


class TestTensor(unittest.TestCase):
    def test_tensor_creation(self):
        t = Tensor([1, 2, 3])
        self.assertTrue(np.array_equal(t.data, np.array([1, 2, 3])))
        self.assertFalse(t.require_grad)
        self.assertIsNone(t.grad)
        self.assertIsNone(t._backward_fn)

        t = Tensor([1, 2, 3], require_grad=True)
        self.assertTrue(np.array_equal(t.data, np.array([1, 2, 3])))
        self.assertTrue(t.require_grad)
        self.assertIsNone(t.grad)
        self.assertIsNone(t._backward_fn)

    def test_tensor_backward(self):
        t = Tensor([1, 2, 3], require_grad=True)
        t.backward()
        self.assertTrue(np.array_equal(t.grad, np.array([1, 1, 1])))

    def test_add(self):
        t1 = Tensor([1, 2, 3], require_grad=True)
        t2 = Tensor([2, 3, 4], require_grad=False)
        res = t1 + t2
        self.assertTrue(np.array_equal(res.data, np.array([3, 5, 7])))
        self.assertTrue(res.require_grad)
        self.assertIsNone(res.grad)
        self.assertIsNotNone(res._backward_fn)

        # test tensor+scalar & commutative property
        t3 = Tensor([1, 2, 3], require_grad=True)
        self.assertTrue(np.array_equal((12 + t3).data, np.array([13, 14, 15])))
        self.assertTrue(np.array_equal((t3 + 12).data, np.array([13, 14, 15])))

    def test_substract(self):
        t1 = Tensor([1, 2, 3], require_grad=False)
        t2 = Tensor([2, 3, 4], require_grad=True)
        res = t1 - t2
        self.assertTrue(np.array_equal(res.data, np.array([-1, -1, -1])))
        self.assertTrue(res.require_grad)
        self.assertIsNone(res.grad)
        self.assertIsNotNone(res._backward_fn)

    def test_multiply(self):
        # test tenser * tensor
        t1 = Tensor([1, 2, 3], require_grad=True)
        t2 = Tensor([2, 3, 4], require_grad=True)
        res = t1 * t2
        self.assertTrue(np.array_equal(res.data, np.array([2, 6, 12])))
        self.assertTrue(res.require_grad)
        self.assertIsNone(res.grad)
        self.assertIsNotNone(res._backward_fn)

        # test scalar*tensor & commutative property
        t3 = Tensor([1, 2, 3], require_grad=True)
        self.assertTrue(np.array_equal((3 * t3).data, np.array([3, 6, 9])))
        self.assertTrue(np.array_equal((t3 * 3).data, np.array([3, 6, 9])))

    def test_divide(self):
        # test tensor / tensor
        t1 = Tensor([1, 3, 1], require_grad=True)
        t2 = Tensor([2, 3, 4], require_grad=True)
        res = t2 / t1
        self.assertTrue(np.array_equal(res.data, np.array([2.0, 1.0, 4.0])))
        self.assertTrue(res.require_grad)
        self.assertIsNone(res.grad)
        self.assertIsNotNone(res._backward_fn)

        # test divided by 0
        t3 = Tensor([1, 3, 1], require_grad=True)
        t4 = Tensor([2, 3, 0], require_grad=True)
        res = t3 / t4
        self.assertTrue(np.linalg.norm(res.data - np.array([0.5, 1.0, 1.0e5])) < 1e-7)
        self.assertTrue(res.require_grad)
        self.assertIsNone(res.grad)
        self.assertIsNotNone(res._backward_fn)

    def test_matmul(self):
        # test matrix multiplication
        d1 = np.array([[1, 2, 3], [4, 6, 5], [2, 3, 1]])
        d2 = np.array([[2, 3, 4], [1, 1, 1], [7, 1, 1]])
        t1 = Tensor(d1, require_grad=False)
        t2 = Tensor(d2, require_grad=True)

        res = t1 @ t2
        self.assertTrue(np.array_equal(res.data, np.matmul(d1, d2)))
        self.assertTrue(res.require_grad)
        self.assertIsNone(res.grad)
        self.assertIsNotNone(res._backward_fn)

        res = t2 @ t1
        self.assertTrue(np.array_equal(res.data, np.matmul(d2, d1)))
        self.assertTrue(res.require_grad)
        self.assertIsNone(res.grad)
        self.assertIsNotNone(res._backward_fn)

    def test_backward(self):
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
        self.assertTrue(
            np.array_equal(
                e.data,
                ((a_data * 3 + b_data) * np.array([2, 3, 4])).dot(
                    np.array([[2, 2], [3, 9], [4, 7]])
                ),
            )
        )

        # test backward
        e.backward()
        self.assertTrue(np.array_equal(e.grad, np.ones_like(e.data)))
        self.assertIsNone(constant2.grad)
        self.assertTrue(np.array_equal(d.grad, e.grad @ constant2_data.T))
        self.assertTrue(np.array_equal(c.grad, d.grad * constant1_data))
        self.assertIsNone(constant1.grad)
        self.assertTrue(np.array_equal(b.grad, c.grad))
        self.assertTrue(np.array_equal(a.grad, 3 * c.grad))
