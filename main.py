from src.tensor import Tensor


x = Tensor([1, 2, 3], requires_grad=True)
y = Tensor([4, 5, 6], requires_grad=True)
b = Tensor([7, 8, 9], requires_grad=True)

z = x * y + b

z.backward()

print("x.grad:", x.grad)
print("y.grad:", y.grad)
