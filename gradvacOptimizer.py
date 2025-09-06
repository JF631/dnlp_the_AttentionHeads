import torch
import numpy as np
import random
from typing import List, Tuple

class GradVac:
    """
    Gradient Vaccine (GradVac) wrapper with a PCGrad-like interface.

    Usage:
        gv = GradVac(optimizer, reduction="sum", target=0.0, alpha=0.5)
        gv.gv_backward(loss)               # single-task
        gv.gv_backward([l1, l2, l3, ...])  # multitask
        optimizer.step()

    Summary:
        For multi-task training with losses {L_t}, GradVac adjusts each task's gradient
        directions so that pairwise cosine similarities move toward a target value
        `target` (often 0.0). After one (randomized) sweep of pairwise adjustments, the
        adjusted task gradients are merged by sum/mean and written back to `.grad`.
    """
    def __init__(self, optimizer, reduction: str = "sum", target: float = 0.0, alpha: float = 0.5):
        """
        Initialize the GradVac controller.

        Args:
            optimizer (torch.optim.Optimizer): Wrapped PyTorch optimizer.
            reduction (str): How to merge adjusted task gradients, {"sum", "mean"}.
            target (float): Target cosine similarity for task gradients.
            alpha (float): Step size in (0, 1] controlling adjustment strength.

        Raises:
            ValueError: If `reduction` is not one of {"sum", "mean"}.
        """
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.optimizer = optimizer
        self.reduction = reduction
        self.target = float(target)
        self.alpha = float(alpha)

    def zero_grad(self):
        """
        Set all parameter gradients to zero via the wrapped optimizer.

        Returns:
            Any: Whatever the wrapped optimizer's `zero_grad` returns (often `None`).
        """
        return self.optimizer.zero_grad(set_to_none=True)

    def gv_backward(self, objectives):
        """
        Backward pass with GradVac adjustments.

        Accepts a single scalar loss tensor or a list/tuple of scalar loss tensors.
        Collects per-task gradients, adjusts them pairwise toward the target cosine,
        merges across tasks, and writes the result to `.grad`.

        Parameters:
            objectives (torch.Tensor | list | tuple): A scalar loss tensor or an iterable
                of scalar loss tensors (each must be 0-dim).

        Raises:
            TypeError: If inputs are not scalar tensors or a list/tuple of them.
        """
        if torch.is_tensor(objectives):
            objectives = [objectives]
        elif not isinstance(objectives, (list, tuple)):
            raise TypeError("gv_backward expects a tensor or a list/tuple of tensors.")

        for i, obj in enumerate(objectives):
            if not torch.is_tensor(obj) or obj.dim() != 0:
                raise TypeError(f"Objective at index {i} must be a scalar tensor.")

        flat_grads, shapes = self._collect_per_task_flat_grads(objectives)
        adjusted = self._adjust_all(flat_grads)
        merged = self._reduce(adjusted)
        unflat = self._unflatten(merged, shapes)
        self._set_param_grads(unflat)

    def _collect_per_task_flat_grads(self, objectives: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Size]]:
        """
        For each objective:
            zero_grad -> backward(retain_graph=True) -> capture gradients (flattened).

        Returns:
            tuple:
                - task_grads (list of torch.Tensor): Flattened gradients per task.
                - param_shapes (list of torch.Size): Shapes of parameters for unflattening.
        """
        param_shapes = None
        task_grads = []
        for obj in objectives:
            self.optimizer.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grads, shapes = self._retrieve_param_grads()
            if param_shapes is None:
                param_shapes = shapes
            task_grads.append(self._flatten(grads))
        return task_grads, param_shapes

    def _retrieve_param_grads(self):
        """
        Retrieve current parameter gradients and their shapes.

        Returns:
            tuple:
                - grads (list of torch.Tensor): Per-parameter gradients (zeros if missing).
                - shapes (list of torch.Size): Shapes of each parameter.
        """
        grads, shapes = [], []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    shapes.append(p.shape)
                    grads.append(torch.zeros_like(p, device=p.device))
                else:
                    shapes.append(p.grad.shape)
                    grads.append(p.grad.detach().clone())
        return grads, shapes

    def _flatten(self, grads: List[torch.Tensor]) -> torch.Tensor:
        """
        Flatten per-parameter gradients into a single vector.

        Returns:
            torch.Tensor: 1D tensor containing all parameters' gradients.
        """
        if not grads:
            return torch.tensor([], device=self._first_param_device())
        return torch.cat([g.reshape(-1) for g in grads])

    def _unflatten(self, flat: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
        """
        Split a flattened gradient vector into per-parameter tensors.

        Returns:
            list of torch.Tensor: Gradients reshaped to match `shapes`.
        """
        out, idx = [], 0
        for shp in shapes:
            n = int(np.prod(shp))
            out.append(flat[idx:idx+n].view(shp).clone())
            idx += n
        return out

    def _set_param_grads(self, grads: List[torch.Tensor]) -> None:
        """
        Write gradients back into `.grad` fields of the optimizer parameters.
        """
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                p.grad = grads[idx]
                idx += 1

    def _first_param_device(self):
        """
        Get the device of the first parameter, fallback to CPU if none exist.
        """
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                return p.device
        return torch.device("cpu")

    @staticmethod
    def _unit_and_norm(v: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize a vector and return both the unit vector and its norm.
        """
        n = v.norm()
        return v / (n + eps), n

    def _adjust_pair(self, g1: torch.Tensor, g2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjust a pair of gradient vectors so their cosine similarity moves toward target.
        """
        u1, n1 = self._unit_and_norm(g1)
        u2, n2 = self._unit_and_norm(g2)
        cos = torch.clamp(torch.dot(u1, u2), -1.0, 1.0)
        d1 = (u1 - cos * u2)
        d2 = (u2 - cos * u1)
        step = (self.target - cos) * self.alpha
        g1_new = g1 + step * d1 * n1
        g2_new = g2 + step * d2 * n2
        return g1_new, g2_new

    def _adjust_all(self, grads: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Adjust all task gradients pairwise toward the target cosine similarity.
        """
        T = len(grads)
        if T <= 1:
            return grads

        adjusted = grads[:]
        indices = list(range(T))
        random.shuffle(indices)
        for a in range(T):
            i = indices[a]
            for b in range(a + 1, T):
                j = indices[b]
                gi, gj = adjusted[i], adjusted[j]
                gi_new, gj_new = self._adjust_pair(gi, gj)
                adjusted[i], adjusted[j] = gi_new, gj_new
        return adjusted

    def _reduce(self, grads: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge adjusted gradients across tasks by sum or mean.
        """
        stacked = torch.stack(grads, dim=0)
        if self.reduction == "sum":
            return stacked.sum(dim=0)
        else:
            return stacked.mean(dim=0)