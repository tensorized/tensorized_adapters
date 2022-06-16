import torch
import torch.nn as nn
from tensorly.random import random_cp
from tensorly.cp_tensor import cp_to_tensor

import tensorly as tl

tl.set_backend('pytorch')

class CPLayer(nn.Module):
    """
    Stores the weights of a fully connected layer in the TT-matrix format
    """
    def __init__(self, shape, rank=2, **kwargs):
        super(CPLayer, self).__init__(**kwargs)
        
        self.rank = rank
        self.shape = shape
        self.weights, self.factors = random_cp(self.shape, rank=self.rank)
        
        # Add and register the factors
        self.weights = nn.Parameter(self.weights, requires_grad=True)
        self.factors = nn.ParameterList([nn.Parameter(f, requires_grad=True) for f in self.factors])
        
        # initialise weights
        
        self.weights.data.uniform_(-0.1, 0.1)
        
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)
            
    def forward(self, x):
        # form full weight matrix
        W = cp_to_tensor((self.weights, self.factors))
        W = W.reshape(x.shape[1], -1)
        # perform regular matrix multiplication
        return torch.matmul(x, W)
    
    def get_num_params(self):
        return sum([p.numel() for p in self.parameters()])