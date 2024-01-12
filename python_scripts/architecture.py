import numpy as np
import torch
from torch import nn
import scipy.stats as stats
import concurrent.futures

class Huber(nn.Module):
    def __init__(self, params):
        super(Huber, self).__init__()

        self.n1, self.n2 = params['size1'], params['size2']
        self.rank = params['rank']
        self.U = torch.randn(self.n1, self.rank, requires_grad = True).to(device)
        self.V = torch.randn(self.rank, self.n2, requires_grad = True).to(device)

        self.hubreg_iters = params['hubreg_iters']
        self.layers = params['layers']
        self.sigma = torch.tensor(params['initial_sigma']).to(device)
        c_list = (torch.ones(self.layers) * torch.tensor(params['initial_c'])).to(device)
        lamda_list = (torch.ones(self.layers) * torch.tensor(params['initial_lamda'])).to(device)
        mu_list = (torch.ones(self.layers) * torch.tensor(params['initial_mu'])).to(device)

        # learnables
        self.c_list = nn.ParameterList([nn.Parameter(c) for c in c_list])
        self.lamda_list = nn.ParameterList([nn.Parameter(lamda) for lamda in lamda_list])
        self.mu_list = nn.ParameterList([nn.Parameter(mu) for mu in mu_list])

        # non-learnables
        # self.sigma = sigma
        # self.hubreg_iters = hubreg_iters
        # self.layers = layers

    def get_rows(self, column):
        # returns row indices of non-zero elements in column

        return torch.nonzero(column).squeeze()

    def hub_deriv(self, x, c):
        # returns the derivative of Equation 2 from Block-Wise Minimization-Majorization Algorithm for Huber'S Criterion: Sparse Learning And Applications

        return torch.cat((x[abs(x) <= c], c * torch.sign(x[abs(x) > c])))

    def hubregv(self, tup_arg):
        # beta: (r, 1), X: (j_i, r), y: (j_i, 1)

        beta, X, y, layer = tup_arg[0], tup_arg[1], tup_arg[2], tup_arg[3]
        # print(f'in V: beta.shape: {beta.shape}, X.shape: {X.shape}, y.shape: {y.shape}')

        # Detach parameters before using them in the function
        sigma = self.sigma
        # print(sigma.requires_grad)
        c = self.c_list[layer].clone().cpu().detach()
        # print(c.requires_grad)
        lamda = self.lamda_list[layer]
        # print(lamda.requires_grad)
        mu = self.mu_list[layer]
        # print(mu.requires_grad)

        alpha = ((0.5 * (c * 2) * (1 - stats.chi2.cdf(c * 2, df = 1))) + (0.5 * stats.chi2.cdf(c ** 2, df = 3)))
        # print(alpha.requires_grad)
        try:
            X_plus = torch.inverse(X.t() @ X) @ X.t()
        except Exception as e:
            print(e)
            print(X.shape, X.t().shape)
            print(X, X.t(), X.t() @ X, sep = '\n')
            return None
        # print(X_plus.requires_grad)

        for _ in range(self.hubreg_iters):
            r = y - (X @ beta)
            # print(r.requires_grad)
            tau = torch.norm(self.hub_deriv(r / sigma, c)) / ((2 * len(y) * alpha)**0.5)
            # print(tau.requires_grad)
            sigma = tau * lamda
            # print(sigma.requires_grad)
            delta = X_plus @ (self.hub_deriv(r / sigma, c).unsqueeze(1) * sigma)
            # print(delta.requires_grad)
            beta = beta + (mu * delta)
            # print(beta.requires_grad)

        # Return the result and attach gradients
        return beta.clone().cpu().detach()

    def hubregu(self, tup_arg):
        # beta: (1, r), X: (r, i_j), y: (1, i_j)

        beta, X, y, layer = tup_arg[0], tup_arg[1], tup_arg[2], tup_arg[3]
        # print(f'in U: beta.shape: {beta.shape}, X.shape: {X.shape}, y.shape: {y.shape}')

        # Detach parameters before using them in the function
        sigma = self.sigma
        c = self.c_list[layer].clone().cpu().detach()
        lamda = self.lamda_list[layer]
        mu = self.mu_list[layer]

        alpha = ((0.5 * (c * 2) * (1 - stats.chi2.cdf(c * 2, df = 1))) + (0.5 * stats.chi2.cdf(c ** 2, df = 3)))
        try:
            X_plus = torch.inverse(X.t() @ X) @ X.t()
        except Exception as e:
            print(e)
            print(X.shape, X.t().shape)
            temp = X.t() @ X
            for i in range(len(temp)):
                print(i)
                print(temp[i, i])
            return None
        
        for _ in range(self.hubreg_iters):
            r = y - (beta @ X) # (1, j_i)
            tau = torch.norm(self.hub_deriv(r / sigma, c)) / ((2 * len(y) * alpha)**0.5)
            sigma = tau * lamda
            delta = (self.hub_deriv(r / sigma, c).unsqueeze(0) * sigma) @ X_plus
            beta = beta + (mu * delta) # (1, r)

        # Return the result and attach gradients
        return beta.clone().cpu().detach()

    def forward(self, X):
        # runs Algorithm 1 from Robust M-Estimation Based Matrix Completion

        U = self.U.clone().detach()
        V = self.V.clone().detach()

        for layer in range(self.layers):
            for j in range(V.shape[1]):
                rows = self.get_rows(X[:, j]) # row indices for jth column
                V[:, j:j+1] = self.hubregv((V[:, j:j+1], U[rows, :], X[rows, j:j+1], layer))

            for i in range(U.shape[0]):
                columns = self.get_rows(X[i, :]) # column indices for ith row
                U[i:i+1, :] = self.hubregu((U[i:i+1, :], V[:, columns], X[i:i+1, columns], layer))
        return U @ V
    
    def getexp_LS(self):

        c_list = [c.cpu().detach().item() for c in self.c_list]
        lamda_list = [lamda.cpu().detach().item() for lamda in self.lamda_list]
        mu_list = [mu.cpu().detach().item() for mu in self.mu_list]
        
        return c_list, lamda_list, mu_list