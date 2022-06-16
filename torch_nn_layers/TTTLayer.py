import torch
import torch.nn as nn
from tensorly.random import random_tt
from tensorly.tt_tensor import tt_to_tensor

import tensorly as tl

tl.set_backend('pytorch')

class TTTLayer(nn.Module):
    """
    Stores the weights of a fully connected layer in the TT-Tensor format
    """
    def __init__(self, shape, rank=2, **kwargs):
        super(TTTLayer, self).__init__(**kwargs)
        
        self.rank = rank
        self.shape = shape
        self.factors = random_tt(self.shape, rank=self.rank)
        # print("\n\n\n\n\n\n\n\n\n\n\n\n Factors is ", self.factors)
        
        # Add and register the factors
        self.factors = nn.ParameterList([nn.Parameter(f, requires_grad=True) for f in self.factors])
        
        
        # initialise weights  
        for f in self.factors:
            # print("\n\n\n\n\n\n\n\n\n\n\n\n Shape is ", f)
            f.data.uniform_(-0.1, 0.1)
            
    def forward(self, x):
        # import pdb; pdb.set_trace() 
        W = tt_to_tensor(self.factors)
        W = W.reshape(x.shape[1], -1)
        return torch.matmul(x, W)
    
    def get_num_params(self):
        return sum([p.numel() for p in self.parameters()])