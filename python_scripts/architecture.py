import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import scipy.stats as stats
import concurrent.futures
from graphviz import Digraph
# make_dot was moved to https://github.com/szagoruyko/pytorchviz
from torchviz import make_dot

def to_var(X, CalInGPU):
    if CalInGPU and torch.cuda.is_available():
        X = X.to('cuda')
    else:
        X = X.to('cpu')
    return Variable(X)

class Huber(nn.Module):
    def __init__(self, sigma, c, lamda, mu, iter):
        super(Huber, self).__init__()

        self.hubreg_iters = iter
        self.sigma = sigma
        
        self.c = nn.Parameter(c)
        self.lamda = nn.Parameter(lamda)
        self.mu = nn.Parameter(mu)
        
        # self.c_list = nn.Parameter(torch.tensor(self.layers * [params['initial_c']]).to(self.device))
        # print(self.c_list)
        # self.lamda_list = nn.Parameter(torch.tensor(self.layers * [params['initial_lamda']]).to(self.device))
        # self.mu_list = nn.Parameter(torch.tensor(self.layers * [params['initial_mu']]).to(self.device))

        # learnables
        # self.c_list = nn.ParameterList([nn.Parameter(c) for c in c_list])
        # self.lamda_list = nn.ParameterList([nn.Parameter(lamda) for lamda in lamda_list])
        # self.mu_list = nn.ParameterList([nn.Parameter(mu) for mu in mu_list])

        # non-learnables
        # self.sigma = sigma
        # self.hubreg_iters = hubreg_iters
        # self.layers = layers

    def get_rows(self, column):
        # returns row indices of non-zero elements in column

        return torch.nonzero(column).squeeze()

    def hub_deriv(self, x):
        # returns the derivative of Equation 2 from Block-Wise Minimization-Majorization Algorithm for Huber'S Criterion: Sparse Learning And Applications

        return torch.cat((x[abs(x) <= self.c.item()], self.c.item() * torch.sign(x[abs(x) > self.c.item()])))

    def hubregv(self, tup_arg):
        # beta: (r, 1), X: (j_i, r), y: (j_i, 1)

        beta, X, y = tup_arg[0], tup_arg[1], tup_arg[2]
        # print(f'in V: beta.shape: {beta.shape}, X.shape: {X.shape}, y.shape: {y.shape}')

        # print(sigma.requires_grad)
        c = self.c.clone().cpu().detach()

        alpha = ((0.5 * (c * 2) * (1 - stats.chi2.cdf(c * 2, df = 1))) + (0.5 * stats.chi2.cdf(c ** 2, df = 3)))
        # print(alpha.requires_grad)
        try:
            X_plus = torch.linalg.pinv(X)
        except Exception as e:
            print(e)
            # print(X.shape, X.t().shape)
            # print(X, X.t(), X.t() @ X, sep = '\n')
            # temp = X.t() @ X
            # for i in range(len(temp)):
            #     print(i)
            #     print(temp[i, i])
            return None
        # print(X_plus.requires_grad)

        for _ in range(self.hubreg_iters):
            r = y - (X @ beta)
            # print(r.requires_grad)
            tau = torch.norm(self.hub_deriv(r / self.sigma)) / ((2 * len(y) * alpha)**0.5)
            # print(tau.requires_grad)
            self.sigma = tau * self.lamda
            # print(sigma.requires_grad)
            delta = X_plus @ (self.hub_deriv(r / self.sigma).unsqueeze(1) * self.sigma)
            # print(delta.requires_grad)
            beta = beta + (self.mu * delta)
            # print(beta.requires_grad)

        # Return the result and attach gradients
        # print(f'Learnable c: {self.c_list[layer].requires_grad}, Learnable lamda: {self.lamda_list[layer].requires_grad}, Learnable mu: {self.mu_list[layer].requires_grad}')
        return beta
        # return beta
    
    def hubregu(self, tup_arg):
        # beta: (1, r), X: (r, i_j), y: (1, i_j)

        beta, X, y = tup_arg[0], tup_arg[1], tup_arg[2]
        # print(f'in U: beta.shape: {beta.shape}, X.shape: {X.shape}, y.shape: {y.shape}')

        # Detach parameters before using them in the function
        sigma = self.sigma
        c = self.c.clone().cpu().detach()

        alpha = ((0.5 * (c * 2) * (1 - stats.chi2.cdf(c * 2, df = 1))) + (0.5 * stats.chi2.cdf(c ** 2, df = 3)))
        try:
            X_plus = torch.linalg.pinv(X)
        except Exception as e:
            print(e)
            # print(X.shape, X.t().shape)
            # temp = X.t() @ X
            # for i in range(len(temp)):
            #     print(i)
            #     print(temp[i, i])
            return None
        
        for _ in range(self.hubreg_iters):
            r = y - (beta @ X) # (1, j_i)
            tau = torch.norm(self.hub_deriv(r / self.sigma)) / ((2 * len(y) * alpha)**0.5)
            sigma = tau * self.lamda
            delta = (self.hub_deriv(r / sigma).unsqueeze(0) * self.sigma) @ X_plus
            beta = beta + (self.mu * delta) # (1, r)

        # Return the result and attach gradients
        # print(f'Learnable c: {self.c_list[layer].requires_grad}, Learnable lamda: {self.lamda_list[layer].requires_grad}, Learnable mu: {self.mu_list[layer].requires_grad}')
        return beta
        # return beta

    def forward(self, lst):
        # runs Algorithm 1 from Robust M-Estimation Based Matrix Completion

        # U = self.U.clone().detach()
        # V = self.V.clone().detach()
        
        X, U, V = lst[0], lst[1], lst[2]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # for layer in range(self.layers):
        for j in range(V.shape[1]):
            rows = self.get_rows(X[:, j]) # row indices for jth column
            V[:, j: j + 1] = self.hubregv((V[:, j: j + 1], U[rows, :], X[rows, j: j + 1]))

        for i in range(U.shape[0]):
            columns = self.get_rows(X[i, :]) # column indices for ith row
            U[i: i + 1, :] = self.hubregu((U[i: i + 1, :], V[:, columns], X[i: i + 1, columns]))
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
        self.layers = params['layers']
        self.CalInGPU = params['CalInGPU']

        self.n1, self.n2 = params['size1'], params['size2']
        self.rank = params['rank']
        self.iter = params['hubreg_iters']

        self.U = torch.ones(self.n1, self.rank)
        self.V = torch.ones(self.rank, self.n2)
        # self.model_denoise = model_denoise

        self.c = to_var(torch.ones(self.layers) * torch.tensor(params['initial_c']), self.CalInGPU)
        self.lamda = to_var(torch.ones(self.layers) * torch.tensor(params['initial_lamda']), self.CalInGPU)
        self.mu = to_var(torch.ones(self.layers) * torch.tensor(params['initial_mu']), self.CalInGPU)

        self.sigma = to_var(torch.tensor(params['initial_sigma']), self.CalInGPU)

        filt = []
        for i in range(self.layers):
            if i == 0:
                filt.append(Huber(self.sigma, self.c[i], self.lamda[i], self.mu[i], self.iter))
            else:
                filt.append(Huber(self.sigma, self.c[i], self.lamda[i], self.mu[i], self.iter))
        self.huber_obj = nn.Sequential(*filt)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Forward Pass recieves the lowrank noisy matrix
    def forward(self, X):

        # Now initalize the neural architecture of even number of layers where even numbered layers correspond to updating U and odd numbered
        # layers correspond to updating V. Each odd numbered layers will have number of nodes equal to number of rows and each even numbered
        # column will have number of nodes equal to the number of columns. For now, same parameters either learnable or not throughout

        # Step 1: Compute Forward Pass through all the layers and predict ground truth matrix
        # print('c before call:', self.c)
        X, U, V = self.huber_obj([X, self.U, self.V])
        # print('c after call:', self.c)

        return U @ V
    
    def getexp_LS(self):

        c_list = [c.cpu().detach().item() for c in self.c]
        lamda_list = [lamda.cpu().detach().item() for lamda in self.lamda]
        mu_list = [mu.cpu().detach().item() for mu in self.mu]
        
        return c_list, lamda_list, mu_list