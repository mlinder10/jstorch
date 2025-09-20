from __future__ import annotations


class Tensor:
    def __init__(self, data: int | float):
        self.data = data

    def __validate_operand(self, other: Tensor | int | float) -> int | float:
        if type(other) == int or type(other) == float:
            return other
        if type(other) == Tensor:
            return other.data
        return TypeError(f"Unsupported type: {type(other)}")

    def __add__(self, other: Tensor | int | float):
        data = self.__validate_operand(other)
        return Tensor(self.data + data)

    def __sub__(self, other: Tensor | int | float):
        data = self.__validate_operand(other)
        return Tensor(self.data - data)

    def __mul__(self, other: Tensor | int | float):
        data = self.__validate_operand(other)
        return Tensor(self.data * data)

    def __truediv__(self, other: Tensor | int | float):
        data = self.__validate_operand(other)
        return Tensor(self.data / data)


def main():
    t1 = Tensor(1)
    t2 = Tensor(2)
    t3 = Tensor(3)
    t4 = Tensor(4)

    t5 = t1 + t2 * t3 - t4 / 4
    print(t5.data)


main()
