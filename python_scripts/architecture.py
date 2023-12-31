import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from python_scripts import forward_pass
import scipy.stats as stats
import concurrent.futures
# Function for cpu or gpu assignment

def to_var(X, CalInGPU):
    if CalInGPU and torch.cuda.is_available():
        X = X.cuda()
    return Variable(X)

class UnfoldedNet_Huber(nn.Module):
    def __init__(self, params = None, model_denoise = None):
        super(UnfoldedNet_Huber, self).__init__()
        
        # Constructor initializes various parameters from the given parameter dictionary
        self.layers = params['layers']
        self.CalInGPU = params['CalInGPU']

        self.n1, self.n2 = params['size1'], params['size2']
        self.rank = params['rank']
        self.iter = params['iter']

        self.U = torch.randn(self.n1, self.rank)
        self.V = torch.randn(self.rank, self.n2)        
        # self.model_denoise = model_denoise

        self.rank = params['rank']
        
        self.c = to_var(torch.tensor(params['initial_c']), self.CalInGPU)
        self.lamda = to_var(torch.tensor(params['initial_lamda']), self.CalInGPU)
        self.mu = to_var(torch.tensor(params['initial_mu']), self.CalInGPU)

        self.sigma = to_var(torch.tensor(params['initial_sigma']), False)

        self.huber_obj = forward_pass.Huber(self.sigma, self.c, self.lamda, self.mu, self.iter, self.layers)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

    # Forward Pass recieves the lowrank noisy matrix
    def forward(self, x):

        # Now initalize the neural architecture of even number of layers where even numbered layers correspond to updating U and odd numbered
        # layers correspond to updating V. Each odd numbered layers will have number of nodes equal to number of rows and each even numbered 
        # column will have number of nodes equal to the number of columns. For now, same parameters either learnable or not throughout

        # Step 1: Compute Forward Pass through all the layers and predict ground truth matrix
        pred_matrix = self.huber_obj.forward(self.U, self.V, x)

        return pred_matrix