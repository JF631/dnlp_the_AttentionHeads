import torch
import numpy as np
import random

class PCGrad:
    def __init__(self, optimizer, reduction='sum'):
        self.optimizer = optimizer
        self.reduction = reduction

    def zero_grad(self):
        return self.optimizer.zero_grad(set_to_none=True)

    def pc_backward(self, objectives):
        """
        Accept either:
          - a single scalar loss tensor, or
          - an iterable (list/tuple) of scalar loss tensors (one per task)
        Computes projected gradients and writes them into .grad of params.
        """
        # Normalize input to a list of scalar tensors
        if torch.is_tensor(objectives):
            objectives = [objectives]
        elif not isinstance(objectives, (list, tuple)):
            raise TypeError("pc_backward(objectives) expects a tensor or a list/tuple of tensors.")

        for i, obj in enumerate(objectives):
            if not torch.is_tensor(obj) or obj.dim() != 0:
                raise TypeError(f"Objective at index {i} must be a scalar tensor, got {type(obj)} with dim={getattr(obj,'dim',lambda:None)()}.")

        grads, shapes, has_grads = self.pc_package_grads(objectives)
        pc_grad = self.pc_project_conflicting(grads=grads, has_grads=has_grads)
        pc_grad = self.pc_unflatten_grads(pc_grad, shapes)
        self.pc_set_grad(pc_grad)

    def pc_package_grads(self, objectives):
        grads, has_grads = [], []
        param_shapes = None
        for objective in objectives:
            # Reset grads, backprop this objective, capture grads
            self.optimizer.zero_grad(set_to_none=True)
            objective.backward(retain_graph=True)
            grad, shape, has_grad = self.pc_retrieve_grad()
            if param_shapes is None:
                param_shapes = shape  # record param shapes once
            grads.append(self.pc_flatten_grads(grad))
            has_grads.append(self.pc_flatten_grads(has_grad))
        return grads, param_shapes, has_grads

    def pc_retrieve_grad(self):
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
        Project task gradients to remove pairwise conflicts, then reduce
        across tasks by sum/mean.
        grads: list of flattened grad tensors, one per task
        has_grads: list of flattened masks (same length as grads)
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
        if self.reduction == "sum":
            merged = stacked.sum(dim=0)
        elif self.reduction == "mean":
            merged = stacked.mean(dim=0)
        else:
            raise ValueError("reduction must be 'sum' or 'mean'")
        return merged

    def pc_flatten_grads(self, grads):
        return torch.cat([g.flatten() for g in grads]) if len(grads) else torch.tensor([], device=next(self.optimizer.param_groups[0]['params'].__iter__()).device)

    def pc_unflatten_grads(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = int(np.prod(shape))
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def pc_set_grad(self, grads):
        idx = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
