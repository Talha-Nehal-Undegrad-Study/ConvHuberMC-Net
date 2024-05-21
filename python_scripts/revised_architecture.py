import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import scipy.stats as stats

from python_scripts.utils import psi, rho, alpha


#  Creating a class which recives the lagrange multiplier y of shape (49, 60) and ouputs two learnable matrices W and B both of shape (49, 60) as in eq (9) of paper.
# The process for getting the output is very similar to Conv2dC above where we add extra dimensions for convolution and then after the operation we remove those extra dimensions

# class LearnableMatrices(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super(LearnableMatrices, self).__init__()
#         self.conv_W = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 3, stride = 1, padding = 1)
#         self.conv_B = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 3, stride = 1, padding = 1)

#     def forward(self, y):
#         temp = y[None, None, :, 0:y.shape[-1]]
#         W = self.conv_W(temp)
#         B = self.conv_B(temp)
#         return W.squeeze(), B.squeeze()

    # # Alternative to this maybe
    # self.W = nn.Parameter(torch.ones((400, 500), device = torch.device(self.device), requires_grad = CalInGPU))
    # self.B = nn.Parameter(torch.zeros((400, 500), device = torch.device(self.device), requires_grad = CalInGPU))


# class PseudoInverse(nn.Module):
#     def __init__(self, a, b, c):
#         super(PseudoInverse, self).__init__()
#         # Initialize W and B as learnable matrices with appropriate dimensions
#         # W has shape (c, a) and B has shape (b, c) to ensure the dot product will have shape (b, a)
#         self.W = nn.Linear(a, c, bias = False)
#         self.B = nn.Linear(c, b, bias = False)

#     def forward(self):
#         # Compute the dot product of W and B to approximate the pseudo-inverse
#         # The result will be of shape (b, a), assuming matrix multiplication rules
#         pseudo_inverse = self.B(self.W.weight).t()
#         return pseudo_inverse

# Example usage:
"""
a = 49  # Dimension of the input matrix's height
b = 60  # Dimension of the input matrix's width
c = 50  # Intermediate dimension size

model = PseudoInverse(a, b, c)
"""

# class PseudoInverse(nn.Module):
#     def __init__(self, rank):
#         super(PseudoInverse, self).__init__()
#         self.W = None
#         self.B = None

#     def forward(self, input_dim, output_dim):
#         # Dynamically create W and B with the required dimensions if they are not already created
#         if self.W is None or self.B is None or self.W.shape[1] != input_dim or self.B.shape[0] != output_dim:
#             # self.inter_dim = np.random.randint(1, 11)
#             # self.W = nn.Parameter(torch.randn(input_dim, self.inter_dim) * 0.01)
#             # self.B = nn.Parameter(torch.randn(self.inter_dim, output_dim) * 0.01)
            
#             self.W = nn.Parameter(torch.randn(input_dim, output_dim) * np.random.uniform(0.01, 0.05))
#             self.B = nn.Parameter(torch.randn(input_dim, output_dim) * np.random.uniform(0.01, 0.05))

#         # Compute the dot product of W and B to approximate the pseudo-inverse
#         # pseudo_inverse = torch.matmul(self.W, self.B)
#         # Compute hadamard product
#         pseudo_inverse = torch.multiply(self.W, self.B)
#         return pseudo_inverse



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
            self.convR = nn.Conv2d(1, 1, (kernel[0], kernel[0]), (1, 1), (pad0, pad1), groups = 1).cuda()
        else:
            self.convR = nn.Conv2d(1, 1, (kernel[0], kernel[0]), (1, 1), (pad0, pad1), groups = 1).to('cpu')
        
        # Initialize weights to zero
        self.convR.weight.data.zero_()

        # For a 3x3 kernel, set the center value to 1 to approximate an identity operation
        if kernel[0] == 3 and kernel[1] == 3:
            self.convR.weight.data[0, 0, 1, 1] = 1

        # Set bias to zero
        self.convR.bias.data.zero_()
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
    def __init__(self, kernel, conv_layers, iter, layers):
        super(Huber, self).__init__()

        self.hubreg_iters = iter
        self.conv_layers = conv_layers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layers = layers


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

        # beta, X, y, matrix_op, conv_op, layer = tup_arg[0], tup_arg[1], tup_arg[2], tup_arg[3], tup_arg[4], tup_arg[5]
        beta, X, y, conv_op, layer = tup_arg[0], tup_arg[1], tup_arg[2], tup_arg[3], tup_arg[4]
        gamma = 2
        
       
        # X_plus_approx = matrix_op.forward(input_dim = beta.shape[0], output_dim = X.shape[0])

        X_plus = torch.linalg.pinv(X) # (r, j_i)
        X_plus_conv = conv_op(X_plus)
        # if torch.isnan(X_plus_conv).any():
        #     print(f'Reconstructed Conv Matrix V {conv_op.convR} \n')
        # print(f'Reconstructed Conv Matrix V {conv_op.convR.weight.data} Has nans: {torch.isnan(X_plus_conv).any()} \n')
        
        # Note: The ill-condition thingy is not caused by hubregv but hubregu

        # norm_value = torch.norm(X_plus_conv_temp, p = 2)
        # X_plus_conv = X_plus_conv_temp / norm_value

        # # Normalize the matrix by its infinity norm
        # X_plus_conv = (X_plus_conv - X_plus_conv.min()) / (X_plus_conv.max() - X_plus_conv.min())


        # For inital estimate beta: might consider nearest neighbour filling thingy

        r = y - torch.matmul(X, beta) # y - XBeta
        scale = 1.4815 * torch.median(torch.abs(r - torch.median(r)))

        N = y.shape[0]
        p = beta.shape[0]

        for _ in range(self.hubreg_iters):

            r = y - torch.matmul(X, beta)
            r_chi = psi(r / scale) * (r / scale) - rho(r / scale)
            scale = torch.sqrt(((gamma * scale ** 2) / (2 * alpha() * (N - p - 1))) * torch.sum(r_chi))

            r_pseu = psi(r / scale) * scale

            delta = torch.matmul(X_plus_conv, r_pseu)

            beta = beta + delta
            # beta += delta.clone()
            # beta = beta.add(delta)

        return beta

    
    def hubregu(self, tup_arg):
        # beta: (1, r), X: (r, i_j), y: (1, i_j)

        # beta, X, y, matrix_op, conv_op, layer = tup_arg[0], tup_arg[1], tup_arg[2], tup_arg[3], tup_arg[4], tup_arg[5]
        beta, X, y, conv_op, layer = tup_arg[0], tup_arg[1], tup_arg[2], tup_arg[3], tup_arg[4]
        gamma = 2
        # print("U\n")
        # print(X)
        # print(X.shape)
        # X = X.reshape(1, -1)
        # print(X.shape)
    
        # w, D, v = torch.linalg.svd(X)
        # print(D)

        # X_plus_approx = matrix_op.forward(input_dim = X.shape[1], output_dim = beta.shape[1])

        X_plus = torch.linalg.pinv(X) # (i_j, r)
        X_plus_conv = conv_op(X_plus)
        
        # print(f'Reconstructed Conv Matrix U {conv_op.convR.weight.data} Has nans: {torch.isnan(X_plus_conv).any()} \n')
        # norm_value = torch.norm(X_plus_conv_temp, p = 2)
        # X_plus_conv = X_plus_conv_temp / norm_value

        # # Normalize the matrix by its infinity norm
        # X_plus_conv = (X_plus_conv - X_plus_conv.min()) / (X_plus_conv.max() - X_plus_conv.min())
        
        r = y - torch.matmul(beta, X) # y - XBeta
        scale = 1.4815 * torch.median(torch.abs(r - torch.median(r)))

        N = y.shape[1]
        p = beta.shape[1]

        for _ in range(self.hubreg_iters):

            r = y - torch.matmul(beta, X)
            r_chi = psi(r / scale) * (r / scale) - rho(r / scale)
            scale = torch.sqrt(((gamma * scale ** 2) / (2 * alpha() * (N - p - 1))) * torch.sum(r_chi))

            r_pseu = psi(r / scale) * scale

            delta = torch.matmul(r_pseu, X_plus_conv)

            beta = beta + delta
            # beta += delta.clone()
            # beta = beta.add(delta)

        return beta


    def forward(self, lst):
        # runs Algorithm 1 from Robust M-Estimation Based Matrix Completion

        # U = self.U.clone().detach()
        # V = self.V.clone().detach()
        
        X, U, V, layer = lst[0], lst[1], lst[2], lst[3]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # for layer in range(self.layers):
        for j in range(V.shape[1]):
            rows = self.get_rows(X[:, j]) # row indices for jth column
            # V[:, j: j + 1] = self.hubregv((V[:, j: j + 1], U[rows, :], X[rows, j: j + 1], self.conv_layers[j]))
            # for psuedo self.matrix_layers[layer * V.shape[1] + j], 
            new_V_col = self.hubregv((V[:, j: j + 1], U[rows, :], X[rows, j: j + 1], 
                                      self.conv_layers[j], 
                                      layer))
            V = torch.cat((V[:, :j], new_V_col, V[:, j + 1:]), dim = 1)

        for i in range(U.shape[0]):
            columns = self.get_rows(X[i, :]) # column indices for ith row
            # print("In Forward PASS!\n")
            # print(V)
            # print("Moving Out\n")
            # U[i: i + 1, :] = self.hubregu((U[i: i + 1, :], V[:, columns], X[i: i + 1, columns], self.conv_layers[j + i]))
            # for psuedo: self.matrix_layers[(V.shape[1] * self.layers) + (layer * U.shape[0] + i)]
            new_U_row = self.hubregu((U[i: i + 1, :], V[:, columns], X[i: i + 1, columns], 
                                      self.conv_layers[j + i], 
                                      layer))
            U = torch.cat((U[:i, :], new_U_row, U[i + 1:, :]), dim = 0)
        
        layer += 1

        # print("Forward Pass Done!")
        # print(f'Reconstructed V: {V}. Has nans: {torch.isnan(V).any()} \n')
        # print(f'Reconstructed U: {U}. Has nans: {torch.isnan(U).any()} \n')
        return [X, U, V, layer]

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

        # Initialize W and B matrices for each column of V and each row of U per each layer
        # W_B_matrices_v = [PseudoInverse(rank = self.rank)] * self.V.shape[1] * self.layers
        
        # W_B_matrices_u = [PseudoInverse(rank = self.rank)] * self.U.shape[0] * self.layers

        # Concatenate all the learnable conv layer
        combined_conv_layers = conv_layers_v + conv_layers_u
        # Create a single ModuleList from the combined list
        self.conv_layers = nn.ModuleList(combined_conv_layers)

        # Concatenate all the learnable matrices layer
        # combined_matrix_layers = W_B_matrices_v + W_B_matrices_u
        # Create a single ModuleList from the combined list of matrices
        # self.matrix_layers = nn.ModuleList(combined_matrix_layers)

        filt = []
        for i in range(self.layers):
            filt.append(Huber(self.kernel, self.conv_layers, self.iter, self.layers))
            
        self.huber_obj = nn.Sequential(*filt)


    # Forward Pass recieves the lowrank noisy matrix
    def forward(self, X):

        # Now initalize the neural architecture of even number of layers where even numbered layers correspond to updating U and odd numbered
        # layers correspond to updating V. Each odd numbered layers will have number of nodes equal to number of rows and each even numbered
        # column will have number of nodes equal to the number of columns. For now, same parameters either learnable or not throughout

        # Step 1: Compute Forward Pass through all the layers and predict ground truth matrix
        # print('c before call:', self.c)
        X, U, V, layer = self.huber_obj([X, self.U.clone(), self.V.clone(), 0])
        # print('c after call:', self.c)

        return U @ V
    


class LP1(nn.Module):
    def __init__(self, kernel, conv_layers, iter):
        super(LP1, self).__init__()

        self.inner_iter = iter
        self.conv_layers = conv_layers
        self.kernel = kernel
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def findNonZeroIndices(self, M_Omega):
        """
        Finds the indices of non-zero elements in a matrix and
        returns them in a specified 2x(number of non-zero elements) format.

        Parameters:
        - M_Omega: A matrix to search for non-zero elements.

        Returns:
        - array_Omega: A 2xM matrix where the first row contains the row indices and
        the second row contains the column indices of the non-zero elements in M_Omega.
        """
        # Find the row and column indices of non-zero elements in M_Omega
        row_indices, col_indices = torch.nonzero(M_Omega, as_tuple = True)
        
        # Combine row and column indices into the array_Omega format
        array_Omega = torch.stack((row_indices, col_indices), dim = 0)

        return array_Omega.to(self.device)

    def lp1v(self, tup_arg):

        beta, U_I, b_I, conv_op, layer = tup_arg[0], tup_arg[1], tup_arg[2], tup_arg[3], tup_arg[4]
        U_I, b_I, beta = U_I.to(self.device), b_I.to(self.device), beta.to(self.device)

        if layer == 1:
            W = torch.eye(len(b_I)).to(self.device)
        else:
            ksi = U_I @ beta - b_I
            W = torch.diag(1 / (ksi.abs() ** 2 + 0.0001) ** (1 / 4))

        for inner_iter in range(self.inner_iter):
            X_plus = torch.linalg.pinv(U_I.T @ W.T @ W @ U_I)
            X_plus_conv = conv_op(X_plus)
            beta = X_plus_conv @ U_I.T @ W.T @ W @ b_I
            ksi = U_I @ beta - b_I
            W = torch.diag(1 / (ksi.abs() ** 2 + 0.0001) ** (1 / 4))

        return beta

    def lp1u(self, tup_arg):

        beta, V_I, b_I, conv_op = tup_arg[0], tup_arg[1], tup_arg[2], tup_arg[3]
        beta, V_I, b_I = beta.to(self.device), V_I.to(self.device), b_I.to(self.device)

        if b_I.size(0) > 0:  # Ensure b_I is not empty
            ksi = (beta @ V_I - b_I).clone()
            W = torch.diag(1 / (ksi.abs() ** 2 + 0.0001) ** (1 / 4)).clone()

            for inner_iter in range(self.inner_iter):
                X_plus = torch.linalg.pinv(V_I @ W @ W.T @ V_I.T)
                X_plus_conv = conv_op(X_plus)
                updated_beta = b_I @ W @ W.T @ V_I.T @ X_plus_conv
                beta = updated_beta.clone()
                ksi = (beta @ V_I - b_I).clone()
                W = torch.diag(1 / (ksi.abs() ** 2 + 0.0001) ** (1 / 4)).clone()

        return beta
    
    def forward(self, lst):
        
        X, U, V, layer = lst[0], lst[1], lst[2], lst[3]
        Omega = self.findNonZeroIndices(X)

        U, V = U.to(self.device), V.to(self.device)

        for j in range(V.shape[1]):
            row_indices = (Omega[1, :] == j).nonzero(as_tuple = True)[0]
            row = Omega[0, row_indices]
            U_I = U[row, :]
            b_I = X[row, j]
            beta = V[:, j]

            new_V_col = self.lp1v((beta, U_I, b_I, self.conv_layers[j], layer))
            # V = torch.cat((V[:, :j], new_V_col, V[:, j + 1:]), dim = 1)
            V[:, j] = new_V_col.squeeze()

        for i in range(U.shape[0]):
            col_indices = (Omega[0, :] == i).nonzero(as_tuple = True)[0]
            col = Omega[1, col_indices]
            V_I = V[:, col]
            b_I = X[i, col]
            
            beta = U[i, :]

            new_U_row = self.lp1u((beta, V_I, b_I, self.conv_layers[j + i]))
            # U = torch.cat((U[:i, :], new_U_row, U[i + 1:, :]), dim = 0)
            U[i, :] = new_U_row.squeeze()
        
        layer += 1

        return [X, U, V, layer]

class UnfoldedNet_LP1(nn.Module):
    def __init__(self, params = None, model_denoise = None):
        super(UnfoldedNet_LP1, self).__init__()

        # Constructor initializes various parameters from the given parameter dictionary
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.layers = params['layers']
        self.CalInGPU = params['CalInGPU']

        self.n1, self.n2 = params['size1'], params['size2']
        self.rank = params['rank']
        self.iter = params['inner_iters'] # analagous to 't' in LP1 script.py

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
            filt.append(LP1(self.kernel, self.conv_layers, self.iter))
            
        self.lp1_obj = nn.Sequential(*filt)


    # Forward Pass recieves the lowrank noisy matrix
    def forward(self, X):

        # Now initalize the neural architecture of even number of layers where even numbered layers correspond to updating U and odd numbered
        # layers correspond to updating V. Each odd numbered layers will have number of nodes equal to number of rows and each even numbered
        # column will have number of nodes equal to the number of columns. For now, same parameters either learnable or not throughout

        # Step 1: Compute Forward Pass through all the layers and predict ground truth matrix
        # print('c before call:', self.c)
        X, U, V, layer = self.lp1_obj([X, self.U.clone(), self.V.clone(), 0])
        # print('c after call:', self.c)

        return U @ V