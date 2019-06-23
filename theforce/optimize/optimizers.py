
# coding: utf-8

# In[ ]:


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
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data.clamp(-1., 1.)
                p.data.add_(-group['lr'], d_p)

        return loss


def example():
    import torch
    x = torch.tensor([1.0], requires_grad=True)
    optimizer = ClampedSGD([x], lr=0.2)
    for _ in range(100):
        optimizer.zero_grad()
        loss = (x**2)
        loss.backward()
        optimizer.step()
    print(x.allclose(torch.zeros(1)))


if __name__ == '__main__':
    example()

