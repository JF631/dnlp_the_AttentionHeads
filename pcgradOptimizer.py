import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


# removed: from fontTools.svgLib.path import shapes
# removed: from torch.optim.adamw import adamw

class PCGrad():
    def __init__(self, optimizer, reduction='sum'):
        self.optimizer = optimizer
        self.reduction = reduction
        return

    '''
    Clear the gradient of the parameters
    '''

    def zero_grad(self):
        return self.optimizer.zero_grad(set_to_none=True)

    '''
    Runs the complete pipeline for the PcGrad process.
    '''

    def pc_backward(self, objectives):
        grads, shapes, has_grads = self.pc_package_grads(
            objectives
        )
        pc_grad = self.pc_project_conflicting(
            grads=grads,
            has_grads=has_grads,
            reduction=self.reduction
        )
        pc_grad = self.pc_unflatten_grads(
            pc_grad,
            shapes
        )
        self.pc_set_grad(
            pc_grad
        )
        return

    '''
    Pack the gradient of the parameters of the network for each objective

    Output:
    - grad: a list of the gradient of the parameters
    - shape: a list of the shape of the parameters
    - has_grad: a list of mask represent whether the parameter has gradient 
    '''

    def pc_package_grads(self, objectives):
        grads, has_grads = [], []
        param_shapes = None
        for objective in objectives:
            self.optimizer.zero_grad(
                set_to_none=True
            )  # Reset the gradients
            objective.backward(
                retain_graph=True
            )  # Run backward propagation on tensor
            grad, shape, has_grad = self.pc_retrieve_grad()
            if param_shapes is None:
                param_shapes = shape  # store shapes once
            grads.append(
                self.pc_flatten_grads(
                    grad,
                    shape
                )
            )
            has_grads.append(
                self.pc_flatten_grads(
                    has_grad,
                    shape)
            )
        return grads, param_shapes, has_grads

    '''
    Get the gradient of the parameters of the network with specific 
    objective

    output:
    - grad: a list of the gradient of the parameters
    - shape: a list of the shape of the parameters
    - has_grad: a list of mask represent whether the parameter has gradient
    '''

    def pc_retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    '''
   This method will compare the projecting direction of the different gradients computed for the multiple tasks. 
   If they do not project in the same direction they will be corrected and merged in the end. 
    '''

    def pc_project_conflicting(self, grads, has_grads, reduction=None, shapes=None):
        if reduction is None:
            reduction = self.reduction
        # Work on a deepcopy so we don't mutate the caller's grads
        pc_grad = [g.clone() for g in grads]
        num_task = len(has_grads)
        # For each task i, project away components that conflict with other
        # tasks' gradients
        for i in range(num_task):
            order = list(range(num_task))
            print("order ", order)
            random.shuffle(order)
            for j in order:
                if i == j:
                    continue  # Edge case handling
                gi, gj = pc_grad[i], grads[j]
                print("gi, gj: ", gi, gj)
                dot = torch.dot(gi, gj)  # provides the direction
                print("dot product: ", dot)
                if dot < 0:  # gradients are pointing in different directions
                    denorm = gj.norm().pow(2)  # simple algebra solves for y and denormalize and makes positive
                    print("denorm: ", denorm)
                    if denorm > 0:
                        # Correct projection, subtract component of gi along gj
                        pc_grad[i] = gi - (dot / denorm) * gj
                        gi = pc_grad[i]  # saving also for gi for next iteration
                        print("gi: ", gi)
        stacked = torch.stack(pc_grad, dim=0)  # concat the corrected gradient
        if reduction == "sum":
            merged = stacked.sum(dim=0)
        elif reduction == "mean":
            merged = stacked.mean(dim=0)
        else:
            raise Exception("reduction must be 'sum' or 'mean'")
        return merged

    '''
    Flattening is a technique that is used to convert multi-dimensional arrays into 1-D array. 
    In this case concat all the gradients to a 1-Dimensional Array
    '''

    def pc_flatten_grads(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    '''
    Expending the gradient tensor in to the desired shape. 
    '''

    def pc_unflatten_grads(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    '''
    Sets the corrected gradients back again to the desired parameters from the orginal
    optimizer.
    '''

    def pc_set_grad(self, grads):
        idx = 0
        for group in self.optimizer.param_groups:  # fixed typo here
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
        return


'''
Neural Network with one simple linear layer that will
be used for testing PcGrad optimization.
'''


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


if __name__ == '__main__':
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)
