from src.tensor import Tensor


x = Tensor(2, requires_grad=True)
y = Tensor(7, requires_grad=True)
b = Tensor(5, requires_grad=True)

z = x * y + b ** 3

z.backward()

print("x.grad:", x.grad)
print("y.grad:", y.grad)
print("b.grad:", b.grad)
