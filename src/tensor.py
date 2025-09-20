from __future__ import annotations
from enum import Enum


class GradFn(Enum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    POW = 4


class Tensor:
    def __init__(self, data: int | float, grad_fn: GradFn = None):
        self.data = data
        self.grad_fn = grad_fn

    def __validate_operand(self, other: Tensor | int | float) -> int | float:
        if type(other) == int or type(other) == float:
            return other
        if type(other) == Tensor:
            return other.data
        return TypeError(f"Unsupported type: {type(other)}")

    def __add__(self, other: Tensor | int | float):
        data = self.__validate_operand(other)
        return Tensor(self.data + data, GradFn.ADD)

    def __sub__(self, other: Tensor | int | float):
        data = self.__validate_operand(other)
        return Tensor(self.data - data, GradFn.SUB)

    def __mul__(self, other: Tensor | int | float):
        data = self.__validate_operand(other)
        return Tensor(self.data * data, GradFn.MUL)

    def __truediv__(self, other: Tensor | int | float):
        data = self.__validate_operand(other)
        return Tensor(self.data / data, GradFn.DIV)

    def __pow__(self, other: Tensor | int | float):
        data = self.__validate_operand(other)
        return Tensor(self.data ** data, GradFn.POW)
