import torch
import numpy as np
import random

class PCGrad:
    def __init__(self, optimizer, reduction='sum'):
        self.optimizer = optimizer
        self.reduction = reduction

    def zero_grad(self):
        """
        Set the gradients of all parameters to zero.

        Behavior:
            Delegates to the wrapped optimizer with `set_to_none=True` for lower memory
            overhead and clearer "no-grad" semantics (params without grads keep `grad=None`).

        Returns:
            Any: Whatever the wrapped optimizer's `zero_grad` returns (often `None`).
        """
        return self.optimizer.zero_grad(set_to_none=True)

    def pc_backward(self, objectives):
        """
        Backward pass with PCGrad gradient projection.

        Parameters:
            objectives: A single scalar loss tensor, or an iterable (list/tuple)
                of scalar loss tensors, one per task. Each element must be a
                0-dim tensor (i.e., a scalar loss).

        Return:
            None. Computes projected gradients and writes them into `.grad`
            of each parameter. Does not call `optimizer.step()`.

        Raises:
            TypeError: If `objectives` is not a tensor or list/tuple of tensors,
                       or if any element is not a scalar tensor.
        """
        if torch.is_tensor(objectives):
            objectives = [objectives]
        elif not isinstance(objectives, (list, tuple)):
            raise TypeError("pc_backward(objectives) expects a tensor or a list/tuple of tensors.")

        for i, obj in enumerate(objectives):
            if not torch.is_tensor(obj) or obj.dim() != 0:
                raise TypeError(
                    f"Objective at index {i} must be a scalar tensor, got {type(obj)} "
                    f"with dim={getattr(obj, 'dim', lambda: None)()}."
                )

        grads, shapes, has_grads = self.pc_package_grads(objectives)
        pc_grad = self.pc_project_conflicting(grads=grads, has_grads=has_grads)
        pc_grad = self.pc_unflatten_grads(pc_grad, shapes)
        self.pc_set_grad(pc_grad)

    def pc_package_grads(self, objectives):
        """
        Compute per-task gradients, flatten them, and record parameter shapes.

        Parameters:
            objectives: A single scalar loss tensor or an iterable of scalar loss
                tensors (one per task).

        Returns:
            tuple:
                - grads (List[torch.Tensor]): Per-task flattened gradients (length = #tasks),
                  each of shape [P], where P is the total number of parameters.
                - shapes (List[torch.Size]): Original shapes of parameters (in optimizer order).
                - has_grads (List[torch.Tensor]): Per-task flattened 0/1 masks (shape [P])
                  indicating whether each parameter had a gradient for that task.

        Notes:
            - Performs `zero_grad(set_to_none=True)` before each backward to isolate
              the gradient of the current task.
            - Uses `retain_graph=True` to allow multiple backward passes over the same graph.
        """
        grads, has_grads = [], []
        param_shapes = None
        for objective in objectives:
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward(retain_graph=True)
            grad, shape, has_grad = self.pc_retrieve_grad()
            if param_shapes is None:
                param_shapes = shape  # record param shapes once
            grads.append(self.pc_flatten_grads(grad))
            has_grads.append(self.pc_flatten_grads(has_grad))
        return grads, param_shapes, has_grads

    def pc_retrieve_grad(self):
        """
        Collect parameter gradients, their shapes, and a "has_grad" mask.

        Iterates parameters in the wrapped optimizer's `param_groups` order and
        returns:
            - grad: a list of per-parameter gradient tensors (cloned); if a parameter
              has no grad, a zeros-like tensor is used.
            - shape: a list of torch.Size objects (one per parameter), equal to
              `p.grad.shape` when grad is present, otherwise `p.shape`.
            - has_grad: a list of tensors (same shape as the parameter) filled with 1s
              when a grad is present and 0s otherwise.

        Returns:
            tuple:
                (grad: List[torch.Tensor],
                 shape: List[torch.Size],
                 has_grad: List[torch.Tensor])
        """
        grad, shape, has_grad = [], [], []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p, device=p.device))
                    has_grad.append(torch.zeros_like(p, device=p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p, device=p.device))
        return grad, shape, has_grad

    def pc_project_conflicting(self, grads, has_grads):
        """
        Project task gradients to reduce pairwise conflicts, then merge across tasks.

        Algorithm:
            For each task i, iterate over tasks j in random order. If the dot product
            <g_i, g_j> < 0 (i.e., conflicting), project g_i onto the orthogonal plane
            of g_j:
                g_i <- g_i - ( <g_i, g_j> / ||g_j||^2 ) * g_j
            Finally, stack all adjusted task gradients and reduce by sum or mean.

        Parameters:
            grads (List[torch.Tensor]): List of flattened gradient tensors (shape [P]),
                one per task.
            has_grads (List[torch.Tensor]): List of flattened 0/1 masks (shape [P]),
                one per task. Currently not used in the projection step but kept for
                potential extensions (e.g., masking parameters per task).

        Returns:
            torch.Tensor: A single merged flattened gradient tensor of shape [P].
        """
        pc_grad = [g.clone() for g in grads]
        num_task = len(has_grads)

        for i in range(num_task):
            order = list(range(num_task))
            random.shuffle(order)
            for j in order:
                if i == j:
                    continue
                gi, gj = pc_grad[i], grads[j]
                dot = torch.dot(gi, gj)
                if dot < 0:
                    denorm = gj.norm().pow(2)
                    if denorm > 0:
                        pc_grad[i] = gi - (dot / denorm) * gj
        stacked = torch.stack(pc_grad, dim=0)
        return self.compute_reduction(stacked)

    def compute_reduction(self, stacked):
        """
        Reduce a stack of per-task gradients to a single gradient.

        Parameters:
            stacked (torch.Tensor): Tensor of shape [T, P], where T is the number of tasks
                and P is the flattened parameter dimension.

        Returns:
            torch.Tensor: Reduced gradient of shape [P].

        Raises:
            ValueError: If `reduction` is not one of {"sum", "mean"}.
        """
        if self.reduction == "sum":
            merged = stacked.sum(dim=0)
        elif self.reduction == "mean":
            merged = stacked.mean(dim=0)
        else:
            raise ValueError("reduction must be 'sum' or 'mean'")
        return merged

    def pc_flatten_grads(self, grads):
        """
        Flatten a list of per-parameter tensors into a single 1D tensor.

        Parameters:
            grads (List[torch.Tensor]): Per-parameter tensors to flatten and concatenate.

        Returns:
            torch.Tensor: A 1D tensor of length P (sum of all parameter counts).
        """
        return torch.cat([g.flatten() for g in grads]) if len(grads) else torch.tensor(
            [], device=next(self.optimizer.param_groups[0]['params'].__iter__()).device
        )

    def pc_unflatten_grads(self, grads, shapes):
        """
        Inverse of `pc_flatten_grads`: split a flattened gradient into parameter-shaped tensors.

        Parameters:
            grads (torch.Tensor): Flattened gradient of shape [P].
            shapes (List[torch.Size]): Target shapes for each parameter, in the same order
                as produced by `pc_retrieve_grad`.

        Returns:
            List[torch.Tensor]: Per-parameter gradient tensors reshaped to `shapes`.
        """
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = int(np.prod(shape))
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def pc_set_grad(self, grads):
        """
        Write computed gradients back into `.grad` fields of the wrapped optimizer's parameters.

        Parameters:
            grads (List[torch.Tensor]): Per-parameter gradients matching the optimizer's
                parameter order (same order as `param_groups` traversal).

        Returns:
            None
        """
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
