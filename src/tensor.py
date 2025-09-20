from __future__ import annotations
from enum import Enum
import numpy as np


class GradFn(Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    MEAN = "mean"
    EXP = "exp"
    LOG = "log"
    RELU = "relu"
    SIGMOID = "sigmoid"


class Tensor:
    def __init__(
        self,
        data: int | float | list | np.ndarray,
        requires_grad: bool = False,
        grad_fn: GradFn | None = None,
        parents: list[Tensor] | None = None,
    ):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.parents = parents or []

    # ---------------- Utility ---------------- #
    def __match_shape(self, grad, shape):
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        for axis, (g, s) in enumerate(zip(grad.shape, shape)):
            if s == 1 and g > 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad.reshape(shape)

    def __ensure_tensor(self, other: int | float | list | np.ndarray | Tensor) -> Tensor:
        if isinstance(other, Tensor):
            return other
        if type(other) in [int, float, list, np.ndarray]:
            return Tensor(other)
        raise TypeError(
            f"Expected Tensor, int, float, list or np.ndarray, got {type(other)}"
        )

    # ---------------- Operators ---------------- #
    def __add__(self, other: Tensor | int | float):
        other = self.__ensure_tensor(other)
        return Tensor(
            self.data + other.data, requires_grad=True,
            grad_fn=GradFn.ADD, parents=[self, other]
        )

    def __sub__(self, other: Tensor | int | float):
        other = self.__ensure_tensor(other)
        return Tensor(
            self.data - other.data, requires_grad=True,
            grad_fn=GradFn.SUB, parents=[self, other]
        )

    def __mul__(self, other: Tensor | int | float):
        other = self.__ensure_tensor(other)
        return Tensor(
            self.data * other.data, requires_grad=True,
            grad_fn=GradFn.MUL, parents=[self, other]
        )

    def __truediv__(self, other: Tensor | int | float):
        other = self.__ensure_tensor(other)
        return Tensor(
            self.data / other.data, requires_grad=True,
            grad_fn=GradFn.DIV, parents=[self, other]
        )

    def __pow__(self, other: Tensor | int | float):
        other = self.__ensure_tensor(other)
        return Tensor(
            self.data ** other.data, requires_grad=True,
            grad_fn=GradFn.POW, parents=[self, other]
        )

    # ---------------- Functions ---------------- #
    def mean(self):
        return Tensor(
            np.mean(self.data), requires_grad=True,
            grad_fn=GradFn.MEAN, parents=[self]
        )

    def exp(self):
        return Tensor(
            np.exp(self.data), requires_grad=True,
            grad_fn=GradFn.EXP, parents=[self]
        )

    def log(self):
        return Tensor(
            np.log(self.data), requires_grad=True,
            grad_fn=GradFn.LOG, parents=[self]
        )

    def relu(self):
        return Tensor(
            np.maximum(0, self.data), requires_grad=True,
            grad_fn=GradFn.RELU, parents=[self]
        )

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        return Tensor(
            sig, requires_grad=True,
            grad_fn=GradFn.SIGMOID, parents=[self]
        )

    # ---------------- Autograd ---------------- #
    def zero_grad(self):
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)

    def backward(self, grad: np.ndarray | float = 1.0):
        grad = np.array(grad, dtype=float)

        if self.requires_grad:
            grad = self.__match_shape(grad, self.data.shape)
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

        if self.grad_fn == GradFn.ADD:
            self.parents[0].backward(grad)
            self.parents[1].backward(grad)

        elif self.grad_fn == GradFn.SUB:
            self.parents[0].backward(grad)
            self.parents[1].backward(-grad)

        elif self.grad_fn == GradFn.MUL:
            grad_x = grad * self.parents[1].data
            grad_y = grad * self.parents[0].data
            self.parents[0].backward(self.__match_shape(
                grad_x, self.parents[0].data.shape))
            self.parents[1].backward(self.__match_shape(
                grad_y, self.parents[1].data.shape))

        elif self.grad_fn == GradFn.DIV:
            grad_x = grad / self.parents[1].data
            grad_y = -grad * self.parents[0].data / (self.parents[1].data ** 2)
            self.parents[0].backward(self.__match_shape(
                grad_x, self.parents[0].data.shape))
            self.parents[1].backward(self.__match_shape(
                grad_y, self.parents[1].data.shape))

        elif self.grad_fn == GradFn.POW:
            base, exp = self.parents
            safe_log = np.where(base.data > 0, np.log(base.data), 0.0)
            grad_base = grad * exp.data * (base.data ** (exp.data - 1))
            grad_exp = grad * (base.data ** exp.data) * safe_log
            self.parents[0].backward(
                self.__match_shape(grad_base, base.data.shape))
            self.parents[1].backward(
                self.__match_shape(grad_exp, exp.data.shape))

        elif self.grad_fn == GradFn.MEAN:
            grad_input = grad * \
                np.ones_like(self.parents[0].data) / self.parents[0].data.size
            grad_input = self.__match_shape(
                grad_input, self.parents[0].data.shape)
            self.parents[0].backward(grad_input)

        elif self.grad_fn == GradFn.EXP:
            grad_input = grad * self.data
            self.parents[0].backward(self.__match_shape(
                grad_input, self.parents[0].data.shape))

        elif self.grad_fn == GradFn.LOG:
            grad_input = grad / self.parents[0].data
            self.parents[0].backward(self.__match_shape(
                grad_input, self.parents[0].data.shape))

        elif self.grad_fn == GradFn.RELU:
            grad_input = grad * (self.parents[0].data > 0).astype(float)
            self.parents[0].backward(self.__match_shape(
                grad_input, self.parents[0].data.shape))

        elif self.grad_fn == GradFn.SIGMOID:
            sig = self.data
            grad_input = grad * sig * (1 - sig)
            self.parents[0].backward(self.__match_shape(
                grad_input, self.parents[0].data.shape))

    # ---------------- Debug ---------------- #
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"


class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
