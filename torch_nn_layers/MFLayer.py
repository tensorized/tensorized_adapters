import torch
import torch.nn as nn
import tensorly as tl

tl.set_backend('pytorch')


class MFLayer(nn.Module):
    """
    Stores the weights of a fully connected layer in the TT-matrix format
    """
    def __init__(self, shape, rank=2, **kwargs):
        super(MFLayer, self).__init__(**kwargs)
        
        self.rank = rank
        self.shape = shape
        self.A, self.B = torch.randn(shape[0], rank), torch.randn(rank, shape[1])
        
        # Add and register the factors
        self.A = nn.Parameter(self.A, requires_grad=True)
        self.B = nn.Parameter(self.B, requires_grad=True)
        
        # initialise weights
        self.A.data.uniform_(-0.1, 0.1)
        self.B.data.uniform_(-0.1, 0.1)
            
    def forward(self, x):
        # form full weight matrix
        W = torch.matmul(self.A, self.B)
        W = W.reshape(x.shape[1], -1)
        
        # perform regular matrix multiplication
        return torch.matmul(x, W)
    
    def get_num_params(self):
        return sum([p.numel() for p in self.parameters()])