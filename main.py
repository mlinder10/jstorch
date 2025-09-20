from src.tensor import Tensor


def main():
    t1 = Tensor(1)
    t2 = Tensor(2)
    t3 = Tensor(3)
    t4 = Tensor(4)

    t5 = t1 + t2 * t3 ** 7 - t4 / 4
    print(t5.grad_fn)


main()
