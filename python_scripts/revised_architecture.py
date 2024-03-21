import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import scipy.stats as stats
import concurrent.futures

from python_scripts.utils import psi, rho, alpha

def to_var(X, CalInGPU):
    if CalInGPU and torch.cuda.is_available():
        X = X.to('cuda')
    else:
        X = X.to('cpu')
    return Variable(X)
    
class Conv2dC(nn.Module):
    def __init__(self, kernel): # Empirically found to be self.kernel to maintain same shape
        super(Conv2dC, self).__init__()

        # Given a kernel size of 2 dimensions, we calculate the padding through the formula (k[0] - 1)/2 --> this helps maintain the shape as close as possible
        pad0 = int((kernel[0] - 1) / 2)
        pad1 = int((kernel[1] - 1) / 2)
        if torch.cuda.is_available():
            self.convR = nn.Conv2d(1, 1, (kernel[0], kernel[0]), (1, 1), (pad0, pad0), groups = 1).cuda()
        else:
            self.convR = nn.Conv2d(1, 1, (kernel[0], kernel[0]), (1, 1), (pad0, pad0), groups = 1).to('cpu')
        # At groups = in_channels, each input channel is convolved with its own set of filters (of size out_channels/in_channels)

    def forward(self, x):
        # get the height dimension and convert it to int
        n = x.shape[-1]
        # This line creates a new tensor xR by slicing the input tensor along the columns dimension (0:n). The None adds an extra dimension, making xR a 4-dimensional tensor with size (1, 1, H, W).
        # The 1's make sure the consistency with the conv operation which is expecting a in_channels, out_channels, which are set to 1
        xR = x[None, None, :, 0:n].clone()
        xR = self.convR(xR)
        # Removing the extra dimension
        xR = xR.squeeze()
        x = xR
        return x


class Huber(nn.Module):
    def __init__(self, kernel, conv_layers, iter):
        super(Huber, self).__init__()

        self.hubreg_iters = iter
        self.conv_layers = conv_layers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_rows(self, column):
        # returns row indices of non-zero elements in column

        return torch.nonzero(column).squeeze()

    def hub_deriv(self, x):
        # returns the derivative of Equation 2 from Block-Wise Minimization-Majorization Algorithm for Huber'S Criterion: Sparse Learning And Applications
        abs_x = torch.abs(x)
        # print(abs_x, self.c)
        deriv = x * (abs_x <= self.c) + self.c * torch.sign(x) * (abs_x > self.c)
        return deriv

    def hubregv(self, tup_arg):
        # beta: (r, 1), X: (j_i, r), y: (j_i, 1)

        beta, X, y, conv_op = tup_arg[0], tup_arg[1], tup_arg[2], tup_arg[3]
        gamma = 2

        X_plus = torch.linalg.pinv(X) # (r, j_i)
        X_plus_conv = conv_op(X_plus)
        
        # Note: The ill-condition thingy is not caused by hubregv but hubregu

        # norm_value = torch.norm(X_plus_conv_temp, p = 2)
        # X_plus_conv = X_plus_conv_temp / norm_value

        # # Normalize the matrix by its infinity norm
        # X_plus_conv = (X_plus_conv - X_plus_conv.min()) / (X_plus_conv.max() - X_plus_conv.min())


        # For inital estimate beta: might consider nearest neighbour filling thingy

        r = y - torch.mm(X, beta) # y - XBeta
        scale = 1.4815 * torch.median(torch.abs(r - torch.median(r)))

        N = y.shape[0]
        p = beta.shape[0]

        for _ in range(self.hubreg_iters):

            r = y - torch.mm(X, beta)
            r_chi = psi(r / scale) * (r / scale) - rho(r / scale)
            scale = torch.sqrt(((gamma * scale ** 2) / (2 * alpha() * (N - p - 1))) * torch.sum(r_chi))

            r_pseu = psi(r / scale) * scale

            delta = torch.mm(X_plus_conv, r_pseu)

            beta = beta + delta

        return beta

    
    def hubregu(self, tup_arg):
        # beta: (1, r), X: (r, i_j), y: (1, i_j)

        beta, X, y, conv_op = tup_arg[0], tup_arg[1], tup_arg[2], tup_arg[3]
        gamma = 2
        # print("U\n")
        print(X)
        print(X.shape)
        # X = X.reshape(1, -1)
        # print(X.shape)
    
        w, D, v = torch.linalg.svd(X)
        print(D)

        X_plus = torch.linalg.pinv(X) # (i_j, r)
        X_plus_conv = conv_op(X_plus)

        # norm_value = torch.norm(X_plus_conv_temp, p = 2)
        # X_plus_conv = X_plus_conv_temp / norm_value

        # # Normalize the matrix by its infinity norm
        # X_plus_conv = (X_plus_conv - X_plus_conv.min()) / (X_plus_conv.max() - X_plus_conv.min())
        
        r = y - torch.mm(beta, X) # y - XBeta
        scale = 1.4815 * torch.median(torch.abs(r - torch.median(r)))

        N = y.shape[1]
        p = beta.shape[1]

        for _ in range(self.hubreg_iters):

            r = y - torch.mm(beta, X)
            r_chi = psi(r / scale) * (r / scale) - rho(r / scale)
            scale = torch.sqrt(((gamma * scale ** 2) / (2 * alpha() * (N - p - 1))) * torch.sum(r_chi))

            r_pseu = psi(r / scale) * scale

            delta = torch.mm(r_pseu, X_plus_conv)

            beta = beta + delta

        return beta


    def forward(self, lst):
        # runs Algorithm 1 from Robust M-Estimation Based Matrix Completion

        # U = self.U.clone().detach()
        # V = self.V.clone().detach()
        
        X, U, V = lst[0], lst[1], lst[2]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # for layer in range(self.layers):
        for j in range(V.shape[1]):
            rows = self.get_rows(X[:, j]) # row indices for jth column
            V[:, j: j + 1] = self.hubregv((V[:, j: j + 1], U[rows, :], X[rows, j: j + 1], self.conv_layers[j]))

        for i in range(U.shape[0]):
            columns = self.get_rows(X[i, :]) # column indices for ith row
            print("In Forward PASS!\n")
            print(V)
            print("Moving Out\n")
            U[i: i + 1, :] = self.hubregu((U[i: i + 1, :], V[:, columns], X[i: i + 1, columns], self.conv_layers[j + i]))
        return [X, U, V]

        # # for layer in range(self.layers):
        # rows = self.get_rows(X[:, col])
        # tensor_col = self.hubregv((self.V[:, col: col + 1], self.U[rows, :], X[rows, col: col + 1]))

        # columns = self.get_rows(X[row, :]) # column indices for ith row
        # tensor_row = self.hubregu((self.U[row: row + 1, :], self.V[:, columns], X[row: row + 1, columns]))

        # return (tensor_row @ tensor_col).squeeze().to(device) # in inference, construct the matrix from these
    
class UnfoldedNet_Huber(nn.Module):
    def __init__(self, params = None, model_denoise = None):
        super(UnfoldedNet_Huber, self).__init__()

        # Constructor initializes various parameters from the given parameter dictionary
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layers = params['layers']
        self.CalInGPU = params['CalInGPU']

        self.n1, self.n2 = params['size1'], params['size2']
        self.rank = params['rank']
        self.iter = params['hubreg_iters']

        self.U = torch.randn(self.n1, self.rank).to(self.device)
        self.V = torch.randn(self.rank, self.n2).to(self.device)
        
        self.kernel = params['kernel']
        
        # Create lists of Conv2dC instances
        conv_layers_v = [Conv2dC(kernel = self.kernel) for _ in range(self.V.shape[1])]
        conv_layers_u = [Conv2dC(kernel = self.kernel) for _ in range(self.U.shape[0])]

        # Concatenate the two lists
        combined_conv_layers = conv_layers_v + conv_layers_u

        # Create a single ModuleList from the combined list
        self.conv_layers = nn.ModuleList(combined_conv_layers)

        filt = []
        for i in range(self.layers):
            filt.append(Huber(self.kernel, self.conv_layers, self.iter))
            
        self.huber_obj = nn.Sequential(*filt)


    # Forward Pass recieves the lowrank noisy matrix
    def forward(self, X):

        # Now initalize the neural architecture of even number of layers where even numbered layers correspond to updating U and odd numbered
        # layers correspond to updating V. Each odd numbered layers will have number of nodes equal to number of rows and each even numbered
        # column will have number of nodes equal to the number of columns. For now, same parameters either learnable or not throughout

        # Step 1: Compute Forward Pass through all the layers and predict ground truth matrix
        # print('c before call:', self.c)
        X, U, V = self.huber_obj([X, self.U.clone(), self.V.clone()])
        # print('c after call:', self.c)

        return U @ V