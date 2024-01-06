import torch
import numpy as np
from torch import nn
import scipy.stats as stats
import concurrent.futures
class Huber(nn.Module):

    def __init__(self, sigma, c = 1.345, lamda = 1, mu = 0, hubreg_iters = 2, layers = 3):
        super(Huber, self).__init__()
        # learnables
        self.c = nn.Parameter(c)
        self.lamda = nn.Parameter(lamda)
        self.mu = nn.Parameter(mu)

        # non-learnables
        self.sigma = sigma
        self.hubreg_iters = hubreg_iters
        self.layers = layers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def get_rows(self, column):
        # returns row indices of non-zero elements in column
        return torch.nonzero(column).squeeze()

    def hub_deriv(self, x):
        # returns the derivative of Equation 2 from Block-Wise Minimization-Majorization Algorithm for Huber'S Criterion: Sparse Learning And Applications

        return torch.cat((x[abs(x) <= self.c], self.c * torch.sign(x[abs(x) > self.c])))

    def hubreg(self, tup_arg):

        beta, X, y = tup_arg[0], tup_arg[1], tup_arg[2]
        # runs Algorithm 1 from Block-Wise Minimization-Majorization Algorithm for Huber'S Criterion: Sparse Learning And Applications
        alpha = (0.5 * (self.c.detach().numpy() ** 2) * (1 - stats.chi2.cdf(self.c.detach().numpy() ** 2, df = 1))) + (0.5 * stats.chi2.cdf(self.c.detach().numpy() ** 2, df = 3))
        X_plus = torch.inverse(X.t() @ X) @ X.t()
        
        for _ in range(self.hubreg_iters):
            r = y - (X @ beta)
            tau = torch.norm(self.hub_deriv(r / self.sigma)) / ((2 * len(y) * alpha)**0.5)
            self.sigma *= tau**self.lamda
            delta = X_plus @ (self.hub_deriv(r / self.sigma) * self.sigma)
            beta += self.mu * delta
        return beta

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
        argslist_v = ((self.V[:, j], self.U[rows, :], self.X[rows, j]) for j, rows in enumerate(v_rows))
        
        u_rows = [self.get_rows(self.X[i, :]) for i in range(self.U.shape[0])]
        argslist_u = ((self.U[i, :].t(), self.V[:, rows].t(), self.X[i, rows].t()) for i, rows in enumerate(u_rows))
        # argslist = ((file, arg1, arg2, arg3) for file in fileslist)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        # results = executor.map(loopfunc, argslist)

        # for rs in results:
        #     print(rs)

        for _ in range(self.layers):
            
            # Run multiprocess for v
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(self.hubreg, argslist_v))
                print("V Multiprocess Done! \n")
                
                for idx, result in enumerate(results):
                    self.V[:, idx] = result

            # Run multiprocess for u
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(self.hubreg, argslist_u)
                print("U Multiprocess Done! \n")
                
                for idx, result in enumerate(results):
                    self.V[idx, :] = result.t()

            # for j in range(self.V.shape[1]):
            #     rows = self.get_rows(self.X[:, j]) # row indices for jth column
            #     self.V[:, j] = self.hubreg(self.V[:, j], self.U[rows, :], self.X[rows, j]) # beta: (r, 1), X: (j_i, r), y: (j_i, 1)

            # for i in range(self.U.shape[0]):
            #     rows = self.get_rows(self.X[i, :]) # column indices for ith row
            #     self.U[i, :] = self.hubreg(self.U[i, :].t(), self.V[:, rows].t(), self.X[i, rows].t()).t() # beta: (r, 1), X: (j_i, r), y: (j_i, 1)

        return torch.matmul(self.U, self.V)
