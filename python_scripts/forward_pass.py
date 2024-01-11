import torch
import numpy as np
from torch import nn
import scipy.stats as stats
import concurrent.futures

class Huber(nn.Module):

    def __init__(self, sigma, c, lamda, mu, hubreg_iters = 2, layers = 3):
        super(Huber, self).__init__()

        # learnables
        self.c = nn.ParameterList([nn.Parameter(c_ele) for c_ele in c])
        self.lamda = nn.ParameterList([nn.Parameter(lambda_ele) for lambda_ele in c])
        self.mu = nn.ParameterList([nn.Parameter(mu_ele) for mu_ele in c])

        # non-learnables
        self.sigma = sigma
        self.hubreg_iters = hubreg_iters
        self.layers = layers

    def get_rows(self, column):
        # returns row indices of non-zero elements in column
        return torch.nonzero(column).squeeze()

    def hub_deriv(self, x, c):
        # returns the derivative of Equation 2 from Block-Wise Minimization-Majorization Algorithm for Huber'S Criterion: Sparse Learning And Applications

        return torch.cat((x[abs(x) <= c], c * torch.sign(x[abs(x) > c])))

    def hubregv(self, tup_arg):
        # beta: (1, r), X: (r, j_i), y: (1, j_i)
        beta, X, y, layer = tup_arg[0], tup_arg[1], tup_arg[2], tup_arg[3]

        # Detach parameters before using them in the function
        sigma = self.sigma.detach().clone()
        c = self.c[layer].detach().clone()
        lamda = self.lamda[layer].detach().clone()
        mu = self.mu[layer].detach().clone()

        alpha = (0.5 * (c.numpy() * 2) * (1 - stats.chi2.cdf(c.numpy() * 2, df = 1))) + (0.5 * stats.chi2.cdf(c.numpy() ** 2, df = 3))
        temp = torch.eye(X.shape[1]) * torch.tensor(1e-5)
        inv_matrix = (X.t() @ X) + temp
        print(f'Inv Matrix V: {inv_matrix}, diagonal_elements: {torch.diagonal(inv_matrix)}')
        X_plus = torch.inverse(inv_matrix) @ X.t() # Future Purpose Note: Computationally Expensive

        for _ in range(self.hubreg_iters):
            r = y - (X @ beta)
            tau = torch.norm(self.hub_deriv(r / sigma, c)) / ((2 * len(y) * alpha)**0.5)
            sigma = tau * lamda
            delta = X_plus @ (self.hub_deriv(r / sigma, c) * sigma)
            beta = beta + (mu * delta)

        # Return the result and attach gradients
        return beta.detach().requires_grad_()

    def hubregu(self, tup_arg):
        # beta: (r, 1), X: (j_i, r), y: (j_i, 1)

        beta, X, y, layer = tup_arg[0], tup_arg[1], tup_arg[2], tup_arg[3]

        # Detach parameters before using them in the function
        sigma = self.sigma.detach().clone()
        c = self.c[layer].detach().clone()
        lamda = self.lamda[layer].detach().clone()
        mu = self.mu[layer].detach().clone()

        alpha = (0.5 * (c.numpy() * 2) * (1 - stats.chi2.cdf(c.numpy() * 2, df = 1))) + (0.5 * stats.chi2.cdf(c.numpy() ** 2, df = 3))
        temp = torch.eye(X.shape[1]) * torch.tensor(1e-5)
        inv_matrix = (X.t() @ X) + temp
        print(f'Inv Matrix U: {inv_matrix}, diagonal_elements: {torch.diagonal(inv_matrix)}')
        X_plus = torch.inverse(inv_matrix) @ X.t() # Future Purpose Note: Computationally Expensive # (j_i, r)

        for _ in range(self.hubreg_iters):
            r = y - (beta @ X) # (1, j_i)
            tau = torch.norm(self.hub_deriv(r / sigma, c)) / ((2 * len(y) * alpha)**0.5)
            sigma = tau * lamda
            delta = (self.hub_deriv(r / sigma, c) * sigma) @ X_plus
            beta = beta + mu * delta # (1, r)

        # Return the result and attach gradients
        return beta.detach().requires_grad_()
        # So use multiprocessing here if possible

        # # Basic syntax
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     # Define a list of arguments here for e.g a list of u_rows or v_cols and somehow pass it to repsective huber cell. Rest is done hopefully
        #     # check this link for more details: https://arnabocean.com/frontposts/2020-10-01-python-multicore-parallel-processing/
        # return None

    def forward(self, U, V, X):
        # runs Algorithm 1 from Robust M-Estimation Based Matrix Completion

        self.U = U
        self.V = V
        self.X = X

        v_rows = [self.get_rows(self.X[:, j]) for j in range(self.V.shape[1])]
        argslist_v = ((self.V[:, j].detach(), self.U[rows, :].detach(), self.X[rows, j].detach()) for j, rows in enumerate(v_rows))

        u_rows = [self.get_rows(self.X[i, :]) for i in range(self.U.shape[0])]
        argslist_u = ((self.U[i, :].t().detach(), self.V[:, rows].t().detach(), self.X[i, rows].t().detach()) for i, rows in enumerate(u_rows))

        for layer in range(self.layers):

            # with concurrent.futures.ProcessPoolExecutor() as executor:
            #     resultsv = list(executor.map(self.hubreg, argslist_v))
            #     print(resultsv)
            #     for idx, result in enumerate(resultsv):
            #         self.V[:, idx] = result.requires_grad_()
            #     print("V Multiprocess Done! \n")
            #     try:
            #         resultsu = list(executor.map(self.hubreg, argslist_u))
            #         print(resultsu)
            #     except Exception as e:
            #         print(f"Error in U processing: {e}")
            #     # print(results)
            #     for idx, result in enumerate(resultsu):
            #         self.U[idx, :] = result.t().requires_grad_()
            #     print("U Multiprocess Done! \n")

            for j in range(self.V.shape[1]):
                rows = self.get_rows(self.X[:, j]) # row indices for jth column
                self.V[:, j] = self.hubregv((self.V[:, j], self.U[rows, :], self.X[rows, j], layer)) # beta: (r, 1), X: (j_i, r), y: (j_i, 1)

            for i in range(self.U.shape[0]):
                rows = self.get_rows(self.X[i, :]) # column indices for ith row
                self.U[i, :] = self.hubregu((self.U[i, :], self.V[:, rows], self.X[i, rows], layer)) # beta: (r, 1), X: (j_i, r), y: (j_i, 1)

        return torch.matmul(self.U, self.V)