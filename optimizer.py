import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                state = self.state[p]
                
                if len(state) == 0:
                    state["step"] = 0
                    state["mt"] = torch.zeros_like(p.data)
                    state["vt"] = torch.zeros_like(p.data)
                
                mt, vt = state['mt'], state['vt']
                beta1, beta2 = group['betas']
                state['step'] += 1
                step = state['step']
                alpha = group["lr"]

                # the underscore versions are executed inplace
                # mt = (mt * beta1) + (1 - beta1) * grad
                mt.mul_(beta1).add_(grad, alpha=(1 - beta1))
                # vt = (vt * beta2) + (1 - beta2) * grad * grad
                vt.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                mt_bias = mt / (1 - (beta1 ** step))
                vt_bias = vt / (1 - (beta2 ** step))

                p.data.addcdiv_(mt_bias, (vt_bias.sqrt().add(group['eps'])), value=-alpha)

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-alpha * group["weight_decay"])




                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, and correct_bias, as saved in
                # the constructor).
                #
                # 1- Update first and second moments of the gradients.
                # 2- Apply bias correction.
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given as the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                ### TODO DONE
                # raise NotImplementedError

        return loss
