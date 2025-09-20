from abc import abstractmethod
from tensor import Tensor
from typing import Callable


class Module:
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass
