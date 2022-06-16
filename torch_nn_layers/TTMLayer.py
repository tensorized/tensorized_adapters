import torch
import torch.nn as nn    
from tensorly.random import random_tt_matrix
from tensorly.tt_matrix import tt_matrix_to_tensor

import tensorly as tl

tl.set_backend('pytorch')

class TTMLayer(nn.Module):
    """
    Stores the weights of a fully connected layer in the TT-Matrix format
    """
    def __init__(self, shape, rank=2, **kwargs):
        super(TTMLayer, self).__init__(**kwargs)
        
        self.rank = rank
        self.shape = shape
        self.factors = random_tt_matrix(self.shape, rank=self.rank)
        
        # Add and register the factors
        self.factors = nn.ParameterList([nn.Parameter(f, requires_grad=True) for f in self.factors])
        
        # initialise weights  
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)
            
    def forward(self, x):
        W = tt_matrix_to_tensor(self.factors)
        W = W.reshape(x.shape[1], -1)
        return torch.matmul(x, W)
    
    def get_num_params(self):
        return sum([p.numel() for p in self.parameters()])