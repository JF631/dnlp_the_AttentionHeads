import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self.optimizer = optimizer
        return

    def zero_grad(self):
        return self.optimizer.zero_grad()(set_to_none=True)

    def pack_gradient(self, objectives):
        grads, shapes, has_grads = [], [], []
        for objective in objectives:
            self.optimizer.zero_grad(set_to_none=True)


if __name__ == '__main__':
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)