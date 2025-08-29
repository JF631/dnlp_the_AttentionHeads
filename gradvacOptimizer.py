import torch
import numpy as np
import random
from typing import List, Tuple

class GradVac:
    """
    Gradient Vaccine wrapper with the same interface style as PCGrad.
    Usage:
        gv = GradVac(optimizer, reduction="sum", target=0.0, alpha=0.5)
        gv.gv_backward(loss) # single-task
        gv.gv_backward([l1, l2, l3, ...]) # multitask
        optimizer.step()
    """
    def __init__(self, optimizer, reduction: str = "sum", target: float = 0.0, alpha: float = 0.5):
        """
        Args
        ----
        optimizer : torch.optim.Optimizer
        reduction: "sum" or "mean" for merging the adjusted grads across tasks
        target    : target cosine similarity s* (0.0 is common in GradVac)
        alpha     : step size for similarity adjustment (0<alpha<=1)
        """
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be 'sum' or 'mean'")
        self.optimizer = optimizer
        self.reduction = reduction
        self.target = float(target)
        self.alpha = float(alpha)

    def zero_grad(self):
        return self.optimizer.zero_grad(set_to_none=True)

    def gv_backward(self, objectives):
        """
        Accept a scalar tensor or a list/tuple of scalar tensors.
        Computes adjusted gradients and writes them into param .grad fields.
        """
        # Normalize input
        if torch.is_tensor(objectives):
            objectives = [objectives]
        elif not isinstance(objectives, (list, tuple)):
            raise TypeError("gv_backward expects a tensor or a list/tuple of tensors.")

        for i, obj in enumerate(objectives):
            if not torch.is_tensor(obj) or obj.dim() != 0:
                raise TypeError(f"Objective at index {i} must be a scalar tensor.")

        # 1) Collect per-task flattened grads + parameter shapes
        flat_grads, shapes = self._collect_per_task_flat_grads(objectives)

        # 2) Adjust grads pairwise toward target cosine (GradVac)
        adjusted = self._adjust_all(flat_grads)

        # 3) Merge across tasks (sum/mean)
        merged = self._reduce(adjusted)

        # 4) Unflatten and set back to .grad
        unflat = self._unflatten(merged, shapes)
        self._set_param_grads(unflat)

    def _collect_per_task_flat_grads(self, objectives: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Size]]:
        """For each loss: zero -> backward -> capture param grads (flattened)."""
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
        grads, shapes = [], []
        device = None
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if device is None:
                    device = p.device
                if p.grad is None:
                    shapes.append(p.shape)
                    grads.append(torch.zeros_like(p, device=p.device))
                else:
                    shapes.append(p.grad.shape)
                    grads.append(p.grad.detach().clone())
        return grads, shapes

    def _flatten(self, grads: List[torch.Tensor]) -> torch.Tensor:
        if not grads:
            # Should not happen with a normal model; have a safe device fallback
            return torch.tensor([], device=self._first_param_device())
        return torch.cat([g.reshape(-1) for g in grads])

    def _unflatten(self, flat: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
        out, idx = [], 0
        for shp in shapes:
            n = int(np.prod(shp))
            out.append(flat[idx:idx+n].view(shp).clone())
            idx += n
        return out

    def _set_param_grads(self, grads: List[torch.Tensor]) -> None:
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                p.grad = grads[idx]
                idx += 1

    def _first_param_device(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                return p.device
        return torch.device("cpu")

    @staticmethod
    def _unit_and_norm(v: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
        n = v.norm()
        return v / (n + eps), n

    def _adjust_pair(self, g1: torch.Tensor, g2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjust a pair (g1, g2) so that their cosine similarity moves toward self.target.
        This follows the intuition of GradVac: shift directions minimally to hit the target.
        """
        u1, n1 = self._unit_and_norm(g1)
        u2, n2 = self._unit_and_norm(g2)

        # current cosine
        cos = torch.clamp(torch.dot(u1, u2), -1.0, 1.0)

        # Update directions: move away from current similarity toward target
        # direction vectors are orthogonal components
        d1 = (u1 - cos * u2)           # derivative of cos wrt g1 direction
        d2 = (u2 - cos * u1)           # derivative of cos wrt g2 direction

        # Step size scaled by alpha and distance to target
        step = (self.target - cos) * self.alpha

        g1_new = g1 + step * d1 * n1
        g2_new = g2 + step * d2 * n2
        return g1_new, g2_new

    def _adjust_all(self, grads: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        For T tasks:
          - iterate over random pairs (i, j)
          - adjust both toward target cosine
          - for stability, we can do one sweep over all pairs
        """
        T = len(grads)
        if T <= 1:
            return grads

        adjusted = grads[:]  # shallow copy; tensors are cloned per op below
        # one randomized sweep over all unordered pairs
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
        stacked = torch.stack(grads, dim=0)
        if self.reduction == "sum":
            return stacked.sum(dim=0)
        else:
            return stacked.mean(dim=0)
