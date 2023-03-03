import torch
from torch.optim.optimizer import Optimizer


class ClampedSGD(Optimizer):
    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data.clamp(-1.0, 1.0)
                p.data.add_(-group["lr"], d_p)

        return loss


class AutoScaleSGD(Optimizer):
    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            scale = torch.stack(
                [
                    torch.tensor(0.0) if p.grad is None else p.grad.data.abs().max()
                    for p in group["params"]
                ]
            ).max()
            if scale > 0.0:
                lr = group["lr"] / scale
            else:
                lr = 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-lr, d_p)

        return loss


def example():
    import torch

    x = torch.tensor([1.0], requires_grad=True)
    optimizer = AutoScaleSGD([x], lr=0.5)
    for _ in range(100):
        optimizer.zero_grad()
        loss = x**2
        loss.backward()
        optimizer.step()
    print(x)
    print(x.allclose(torch.zeros(1)))


if __name__ == "__main__":
    example()
