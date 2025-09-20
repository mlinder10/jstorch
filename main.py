import numpy as np
from src.tensor import Tensor, SGD

xs = np.linspace(-5, 5, 20)
ys = 2 * xs + 5

w = Tensor(np.random.randn(), requires_grad=True)
b = Tensor(np.random.randn(), requires_grad=True)

optimizer = SGD([w, b], lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()

    preds = w * xs + b
    loss = ((preds - ys) ** 2).mean()

    loss.backward()

    optimizer.step()

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch}: loss={loss.data:.4f}, w={w.data:.4f}, b={b.data:.4f}")
