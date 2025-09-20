from abc import abstractmethod
from src.module import Model
from tensor import Tensor


class Loss(Model):
    def __init__(self):
        pass

    @abstractmethod
    def calc(self, y_true: Tensor, y_pred: Tensor):
        pass


class MSELoss(Loss):
    def calc(self, y_true: Tensor, y_pred: Tensor):
        return ((y_true - y_pred) ** 2).mean()

    def forward(self, x):
        return super().forward(x)

    def backward(self, x):
        return super().backward(x)
