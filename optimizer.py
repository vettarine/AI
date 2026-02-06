import torch
from torch.optim import Optimizer

class AdaSmooth(Optimizer):

    def __init__(self, params, lr=1e-4, beta=0.9, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                if len(state) == 0:
                    state["avg_grad"] = torch.zeros_like(p)

                avg_grad = state["avg_grad"]

                avg_grad.mul_(beta).add_(grad, alpha=1 - beta)

                p.add_(avg_grad, alpha=-lr)
