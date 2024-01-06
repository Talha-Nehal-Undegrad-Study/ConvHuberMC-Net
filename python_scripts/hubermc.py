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

# class HuberCellV(nn.Module):
#     # Constructor initalizes all the parameters that were passed to it from the unfolded net. Note: v, neta, lamda1/2, S are different for each layer. coef_gamma is constant
#     def __init__(self, c, w, lamda, sigma, mu, delta, tau, iter, CalInGPU = True):
#         super(HuberCellV,self).__init__()

#         self.c = nn.Parameter(c)
#         self.w = nn.Parameter(w)
#         self.lamda = nn.Parameter(lamda)
#         self.sigma = nn.Parameter(sigma)
#         self.mu = nn.Parameter(mu)
#         self.delta = nn.Parameter(delta)
#         self.tau = nn.Parameter(tau)
#         self.iter = iter
        
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
#         self.CalInGPU = CalInGPU
#         self.relu = nn.ReLU()
#         self.sig = nn.Sigmoid()
    
#     def get_row_col_indices(matrix): # returns the indices of the sampling matrix where row entries or column entries are non-zero
#     # Get dimensions of matrix
#         n1, n2 = matrix.shape
        
#         # First Column wise
#         indices_col = [torch.where(matrix[:, i] == 1)[0] for i in range(n2)]
#         indices_col = [tensor.tolist() for tensor in indices_col]

#         # Now Row wise
#         indices_row = [np.where(matrix[i, :] == 1)[0] for i in range(n1)]
#         indices_row = [tensor.tolist() for tensor in indices_row]

#         return indices_col, indices_row

#     # Forward Pass recieves a list of 3 elements, data tensor of shape (2, 160, 320), a mask of shape (160, 320) which is True wherever there is
#     # missing value in the matrix at index data[0], and rank of lowrank matrix
    
#     def get_filter_mat(self, matrix, U, indices_col, col_num):
#         lst = indices_col[col_num]
#         mat_filter = torch.tensor([matrix[sub, col_num] for sub in lst]).unsqueeze(1)
#         u_filter = ([U[sub, :] for sub in lst])
#         u_filter = torch.stack(u_filter, dim = 1).T
#         return mat_filter, u_filter
    
#     def score_func(self, x):
#         return torch.where(torch.abs(x) <= self.c, x, self.c * torch.sign(x))
    
#     def weighted_dot_product(self, weight_vec, a_vec, b_vec):
#         return torch.sum(weight_vec * a_vec * b_vec)

#     def forward(self, lst):
#         X_Omega, sampling_mat, rank, U, v_col, col_num = lst[0], lst[1], lst[2], lst[3], lst[4], lst[5]
#         indices_col, indices_row = self.get_row_col_indices(sampling_mat)
#         n1, n2 = X_Omega.shape

#         # Get alpha for Fischer Consistency
#         alpha = (self.c ** 2 / 2) * (1 - stats.chi2.pdf(self.c ** 2, 1)) + (1 / 2) * stats.chi2.pdf(self.c ** 2, 3)
        
#         # Get U_Ij Dagger
#         mat_filter, U_filter = self.get_filter_mat(matrix = X_Omega, U = U, indices_col = indices_col, col_num = col_num)
#         U_dagger = np.linalg.inv(U_filter.T @ U_filter) @ U_filter.T

#         # Apply Hubreg Algorithm for as many as there are iterations.
#         for _ in range(0, self.iter):
#             # Get Residual Vector
#             res_vec = mat_filter - np.dot(U_filter, v_col)
#             # Update Step size for scale
#             self.lamda = self.lamda + torch.log(torch.norm(self.score_func((res_vec / (self.sigma * (self.tau ** self.lamda))) * (1 / torch.sqrt(2 * alpha * len(U_filter)))), p = 2))
#             # Updated Scale
#             self.sigma = self.sigma * (self.tau ** self.lamda)
#             # Update Direction for update vj - delta
#             self.delta = np.dot(U_dagger, self.score_func(res_vec / self.sigma) * self.sigma)
#             # Get Regression Z = Xdelta
#             z = np.dot(U_filter, self.delta)
#             # Update step size for vj --> mu by first finding appropriate weighting throught IRWLS
#             w_vec = self.w * ((res_vec - self.mu * z) / self.sigma)
#             self.mu = self.weighted_dot_product(w_vec, z, z) * self.weighted_dot_product(w_vec, res_vec, z)
#             # Update Vj
#             v_col = v_col + self.mu * self.delta

#         return v_col
    

# class HuberCellU(nn.Module):
#     # Constructor initalizes all the parameters that were passed to it from the unfolded net. Note: v, neta, lamda1/2, S are different for each layer. coef_gamma is constant
#     def __init__(self, c, w, lamda, sigma, mu, delta, tau, iter, CalInGPU = True):
#         super(HuberCellV,self).__init__()

#         self.c = nn.Parameter(c)
#         self.w = nn.Parameter(w)
#         self.lamda = nn.Parameter(lamda)
#         self.sigma = nn.Parameter(sigma)
#         self.mu = nn.Parameter(mu)
#         self.delta = nn.Parameter(delta)
#         self.tau = nn.Parameter(tau)
#         self.iter = iter
        
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
#         self.CalInGPU = CalInGPU
#         self.relu = nn.ReLU()
#         self.sig = nn.Sigmoid()

#     def get_row_col_indices(matrix): # returns the indices of the sampling matrix where row entries or column entries are non-zero
#     # Get dimensions of matrix
#         n1, n2 = matrix.shape
        
#         # First Column wise
#         indices_col = [torch.where(matrix[:, i] == 1)[0] for i in range(n2)]
#         indices_col = [tensor.tolist() for tensor in indices_col]

#         # Now Row wise
#         indices_row = [np.where(matrix[i, :] == 1)[0] for i in range(n1)]
#         indices_row = [tensor.tolist() for tensor in indices_row]

#         return indices_col, indices_row

#     # Forward Pass recieves a list of 3 elements, data tensor of shape (2, 160, 320), a mask of shape (160, 320) which is True wherever there is
#     # missing value in the matrix at index data[0], and rank of lowrank matrix
    
#     def get_filter_mat(self, matrix, V, indices_row, row_num):
#         lst = indices_row[row_num]
#         mat_filter = torch.tensor([matrix[row_num, sub] for sub in lst]).unsqueeze(1).T
#         v_filter = ([V[:, sub] for sub in lst])
#         v_filter = torch.stack(v_filter, dim = 1)
#         return mat_filter, v_filter
    
#     def score_func(self, x):
#         return torch.where(torch.abs(x) <= self.c, x, self.c * torch.sign(x))
    
#     def weighted_dot_product(self, weight_vec, a_vec, b_vec):
#         return torch.sum(weight_vec * a_vec * b_vec)

#     def forward(self, lst):
#         X_Omega, sampling_mat, rank, V, u_row, row_num = lst[0], lst[1], lst[2], lst[3], lst[4], lst[5]
#         indices_col, indices_row = self.get_row_col_indices(sampling_mat)
#         n1, n2 = X_Omega.shape

#         # Get alpha for Fischer Consistency
#         alpha = (self.c ** 2 / 2) * (1 - stats.chi2.pdf(self.c ** 2, 1)) + (1 / 2) * stats.chi2.pdf(self.c ** 2, 3)
        
#         # Get U_Ij Dagger
#         mat_filter, v_filter = self.get_filter_mat(matrix = X_Omega, V = V, indices_col = indices_row, col_num = row_num)
#         v_dagger = np.linalg.inv(v_filter.T @ v_filter) @ v_filter.T

#         # Apply Hubreg Algorithm for as many as there are iterations.
#         for _ in range(0, self.iter):
#             # Get Residual Vector
#             res_vec = (mat_filter) - np.dot(u_row, v_filter)
#             # Update Step size for scale
#             self.lamda = self.lamda + torch.log(torch.norm(self.score_func((res_vec / (self.sigma * (self.tau ** self.lamda))) * (1 / torch.sqrt(2 * alpha * len(v_filter)))), p = 2))
#             # Updated Scale
#             self.sigma = self.sigma * (self.tau ** self.lamda)
#             # Update Direction for update vj - delta
#             self.delta = np.dot(self.score_func(res_vec / self.sigma) * self.sigma, v_dagger) # Opposite
#             # Get Regression Z = Xdelta
#             z = np.dot(self.delta, v_filter) # Opposite
#             # Update step size for vj --> mu by first finding appropriate weighting throught IRWLS
#             w_vec = self.w * ((res_vec - self.mu * z) / self.sigma)
#             self.mu = self.weighted_dot_product(w_vec, z, z) * self.weighted_dot_product(w_vec, res_vec, z)
#             # Update Vj
#             u_row = u_row + self.mu * self.delta

#         return u_row

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
        
        self.c = to_var(params['initial_c'], self.CalInGPU)
        self.lamda = to_var(params['initial_lamda'], self.CalInGPU)
        self.mu = to_var(params['initial_mu'], self.CalInGPU)

        self.sigma = to_var(params['initial_sigma'], False)

        self.huber_obj = forward_pass.Huber(self.sigma, self.c, self.lamda, self.mu, self.iter, self.layers)

        # Learning parameters for hubreg per layer

        # self.c = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_c'], self.CalInGPU)

        # self.w = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_w'], self.CalInGPU)
        # self.initial_lamda = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_lamda'], self.CalInGPU)

        # self.initial_sigma = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_sigma'], self.CalInGPU)

        # self.initial_mu = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_mu'], self.CalInGPU)
        # self.initial_delta = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_delta'], self.CalInGPU)
        # self.initial_tau = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_tau'], self.CalInGPU)

        # For matrix if needed
        # self.S = to_var(torch.ones((self.layers, params['size1'], params['size1']), requires_grad = True) * params['initial_S'], self.CalInGPU, True)
        # self.y1 = nn.Parameter(to_var(torch.ones((params['size1'], params['size2']), requires_grad = True) * params['initial_y1'], self.CalInGPU))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

        # self.indices_col, self.indices_row = self.get_row_col_indices()
        # self.init_model = self.makelayers()

    # Function which intializes num_layers ISTA cells by passing it those parameters that are learnt per layer like lambda1/2, neta, S and those that are fixed like coef_gamma
    # def makelayers(self):
    #     filt = []
    #     for layer_idx in range(self.layers):
    #         if (layer_idx + 1) % 2 != 0: # --> V Update Layer - n2 nodes
    #             layer_nodes = nn.ModuleList([HuberCellV(self.c[layer_idx], self.w[layer_idx], self.initial_lamda[layer_idx], 
    #                                                     self.initial_sigma[layer_idx], self.initial_mu[layer_idx], 
    #                                                     self.initial_delta[layer_idx], self.initial_tau[layer_idx], self.iter) 
    #                                                     for col in range(self.n2)])
    #             filt.append(layer_nodes)
    #         else: # --> U Update Layer - n1 nodes
    #             layer_nodes = nn.ModuleList([HuberCellU(self.c[layer_idx], self.w[layer_idx], self.initial_lamda[layer_idx], 
    #                                                     self.initial_sigma[layer_idx], self.initial_mu[layer_idx], 
    #                                                     self.initial_delta[layer_idx], self.initial_tau[layer_idx], self.iter) 
    #                                                     for col in range(self.n1)])
    #             filt.append(layer_nodes)
    #     return nn.Sequential(*filt)
    # Forward Pass recieves the lowrank noisy matrix
    def forward(self, x):

        # Now initalize the neural architecture of even number of layers where even numbered layers correspond to updating U and odd numbered
        # layers correspond to updating V. Each odd numbered layers will have number of nodes equal to number of rows and each even numbered 
        # column will have number of nodes equal to the number of columns. For now, same parameters either learnable or not throughout

        # sampling_mat = torch.where(x != 0, torch.tensor(1), torch.tensor(0))
        # v_args = [self.V[:, i] for i in range(self.V.shape[1])]
        # u_args = [self.U[i, :] for i in range(self.U.shape[0])]
        # rank = self.rank

       #  X_Omega, sampling_mat, rank, V, u_row, row_num send these arguments some to huber cell
       # but the problem is the model is sequential and different arguments for V or U Huber Cell. So use multiprocessing here if possible
        
        # arglist_v = ((x, sampling_mat, self.rank, self.U, ))


        # Basic syntax
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Define a list of arguments here for e.g a list of u_rows or v_cols and somehow pass it to repsective huber cell. Rest is done hopefully
            # check this link for more details: https://arnabocean.com/frontposts/2020-10-01-python-multicore-parallel-processing/
        return None
    
    # Ignore the below function for now.

    def getexp_LS(self):
        neta = self.neta
        lamda1 = self.lamda1
        lamda2 = self.lamda2

        v = self.v
        S = self.S
        rho = self.rho
        
        coef_gamma = self.coef_gamma
        # exp_tau used for svt threshold of Z/X (eq 13 of original paper)
        exp_tau = self.sig(v) * coef_gamma

        if torch.cuda.is_available():
          neta = neta.cpu().detach().numpy()
          v = v.cpu().detach().numpy()
          lamda1 = lamda1.cpu().detach().numpy()
          lamda2 = lamda2.cpu().detach().numpy()
          S = S.cpu().detach().numpy()
          rho = rho.cpu().detach().numpy()
          coef_gamma = coef_gamma.cpu().detach().numpy()
          exp_tau = exp_tau.cpu().detach().numpy()
        else:
          neta = neta.detach().numpy()
          v = v.detach().numpy()
          lamda1 = lamda1.detach().numpy()
          lamda2 = lamda2.detach().numpy()
          S = S.detach().numpy()
          rho = rho.detach().numpy()
          coef_gamma = coef_gamma.detach().numpy()
          exp_tau = exp_tau.detach().numpy()
        return neta, v, lamda1, lamda2, S, rho, coef_gamma, exp_tau
