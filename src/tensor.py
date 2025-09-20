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
        self.grad = np.zeros_like(self.data)
        self.parents = parents or []

    def __ensure_tensor(self, other: int | float | Tensor) -> Tensor:
        if isinstance(other, Tensor):
            return other
        return Tensor(other)

    def __add__(self, other: Tensor | int | float):
        other = self.__ensure_tensor(other)
        return Tensor(self.data + other.data, requires_grad=True,
                      grad_fn=GradFn.ADD, parents=[self, other])

    def __sub__(self, other: Tensor | int | float):
        other = self.__ensure_tensor(other)
        return Tensor(self.data - other.data, requires_grad=True,
                      grad_fn=GradFn.SUB, parents=[self, other])

    def __mul__(self, other: Tensor | int | float):
        other = self.__ensure_tensor(other)
        return Tensor(self.data * other.data, requires_grad=True,
                      grad_fn=GradFn.MUL, parents=[self, other])

    def __truediv__(self, other: Tensor | int | float):
        other = self.__ensure_tensor(other)
        return Tensor(self.data / other.data, requires_grad=True,
                      grad_fn=GradFn.DIV, parents=[self, other])

    def __pow__(self, other: Tensor | int | float):
        other = self.__ensure_tensor(other)
        return Tensor(self.data ** other.data, requires_grad=True,
                      grad_fn=GradFn.POW, parents=[self, other])

    def mean(self):
        return Tensor(np.mean(self.data), requires_grad=True,
                      grad_fn=GradFn.MEAN, parents=[self])

    def backward(self, grad: np.ndarray | float = 1.0):
        grad = np.array(grad, dtype=float)
        self.grad += grad

        if self.grad_fn == GradFn.ADD:
            self.parents[0].backward(grad)
            self.parents[1].backward(grad)
        elif self.grad_fn == GradFn.MEAN:
            grad_input = grad * \
                np.ones_like(self.parents[0].data) / self.parents[0].data.size
            self.parents[0].backward(grad_input)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
