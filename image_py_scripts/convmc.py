import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
# Function for cpu or gpu assignment
def to_var(X, CalInGPU):
    if CalInGPU and torch.cuda.is_available():
        X = X.cuda()
    return Variable(X)


class ISTACell_admm(nn.Module):
    # Constructor initalizes all the parameters that were passed to it from the unfolded net. Note: v, neta, lamda1/2, S are different for each layer. coef_gamma is constant
    def __init__(self, neta, v, lamda1, lamda2, S, rho, coef_gamma, CalInGPU):
        super(ISTACell_admm,self).__init__()

        self.v = nn.Parameter(v)
        self.neta = nn.Parameter(neta)
        self.lamda1 = nn.Parameter(lamda1)
        self.lamda2 = nn.Parameter(lamda2)

        self.rho = nn.Parameter(rho)
        self.S = nn.Parameter(S)
        self.coef_gamma = coef_gamma
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.CalInGPU = CalInGPU
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.softshrink = None

    # Forward Pass recieves a list of 4 elements, data tensor of shape (2, 49, 60), a mask of shape (49, 60) which is True wherever there is no missing value in the matrix at index data[0], the temporal
    # matrix D_tilda, and the lagrange multiplier.
    def forward(self, lst):
        # Get every element of the list
        data = lst[0]
        entries_mask = lst[1]
        D_tilda = lst[2]
        th_P = lst[3]

        # Get the lowrank and the L matrix which is for now (49, 60) zeros
        input = data[0]
        L = data[1]

        # get the dimension and the parameters
        H, U = input.shape
        th_incomplete_tau = self.sig(self.v) * self.coef_gamma
        th_neta = self.neta
        th_lamda1 =  self.lamda1
        th_lamda2 =  self.lamda2
        th_rho = self.rho
        th_S = self.S
        coef_gamma = self.coef_gamma

        # Get Q_tilda which is the diagonal of the flattened Q - sampling matrix
        Q = entries_mask.view(H, U).float()
        Q_tilda = torch.diag(Q.view(Q.shape[0] * Q.shape[1]))

        # Get S_tilda which is the kronecker product of I_N identity matrix (where N is 60) and S*S
        S_tilda = torch.kron(torch.eye(U, device = torch.device(self.device), requires_grad = self.CalInGPU), torch.matmul(th_S.T, th_S))

        # The Reconstruction Layer - Eq (12 in original paper)
        # Second half of equation - L is Z in the paper
        vec = th_rho * (L.view(H, U) - th_P.view(H, U)) + torch.where(entries_mask.view(H, U), input.view(H, U), 0)
        # First half of equation
        Xtmp1 = torch.linalg.inv(Q_tilda + th_lamda1 * D_tilda + th_lamda2 * S_tilda + th_rho * torch.eye(H * U, device = torch.device(self.device), requires_grad = self.CalInGPU))
        # Combining the two
        Xtmp = Xtmp1 @ vec.view(H * U)

        # The Non-Linear Transform Layer - Eq (13 in original paper)
        Ltmp = self.svtC(Xtmp.view(H, U) + th_P.view(H, U), th_incomplete_tau)

        # The Multiplier Update Layer - Eq (14 in original paper)
        Ptmp = th_P.view(H, U) + th_neta * (Xtmp.view(H, U) - Ltmp.view(H, U))

        # Update data tensor with the reconstructed prediction and return the list. The list gets passed to the next ISTA cell and then after all layers returns the list.
        data[1] = Ltmp.view(H, U)
        return [data, entries_mask, D_tilda, Ptmp.view(H,U)]

    def svtC(self, x, th):
        m, n = x.shape
        U, S, V = torch.svd(x)
        tmp = to_var(torch.zeros_like(S), self.CalInGPU)
        tmp = S - (th * S[0])
        tmp2 = to_var(torch.zeros_like(tmp), self.CalInGPU)
        tmp2 = self.relu(tmp)
        VS = to_var(torch.zeros(n, m), self.CalInGPU)
        stmp = to_var(torch.zeros(m), self.CalInGPU)
        stmp[0 : tmp2.shape[-1]] = tmp2
        minmn = min(m, n)
        VS[:, 0:minmn] = V[:, 0:minmn]
        y = torch.zeros_like(x)
        y = (U * stmp) @ VS.t()
        return y

class UnfoldedNet3dC_admm(nn.Module):
    def __init__(self, params = None):
        super(UnfoldedNet3dC_admm, self).__init__()


        # Constructor initializes various parameters from the given parameter dictionary
        self.layers = params['layers']
        self.CalInGPU = params['CalInGPU']

        self.params = params

        self.rho = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_rho'], self.CalInGPU)
        self.coef_gamma = to_var(torch.tensor(params['coef_gamma'], dtype = torch.float), self.CalInGPU)
        self.neta = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_neta'], self.CalInGPU)
        self.v = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_v'], self.CalInGPU)
        self.lamda1 = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_lamda1'], self.CalInGPU)
        self.lamda2 = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_lamda2'], self.CalInGPU)

        self.S = to_var(torch.ones((self.layers, params['size1'], params['size1']), requires_grad = True) * params['initial_S'], self.CalInGPU, True) # For spatial correlation

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.filter = self.makelayers()

    # Function which intializes num_layers ISTA cells by passing it those parameters that are learnt per layer like lambda1/2, neta, S and those that are fixed like coef_gamma
    def makelayers(self):
        filt = []
        for i in range(self.layers):
          filt.append(ISTACell_admm(self.neta[i], self.v[i], self.lamda1[i] ** (i + 1), self.lamda2[i] ** (i + 1), self.S[i], self.rho[i] ** (i + 1), self.coef_gamma, self.CalInGPU))
        return nn.Sequential(*filt)

    # Forward Pass recieves a list containing only one element for now and that is the Lowrank(denoised maybe) component
    def forward(self, x):
        # We intialize a tensor 'data' of shape (2, 49, 60) and assign the first index of it with the lowrank component passed i.e. first element of x
        data = to_var(torch.zeros([2] + list(x[0].shape)), self.CalInGPU)
        data[0] = x[0]

        # Get the dimensions i.e. H = 49, U = 60 and then intialize a mask of shape (49, 60) which is True wherever there is not a missing value in x/L
        H, U = x[0].shape
        # entries_mask = ~(torch.isnan(data[0]))
        entries_mask = (data[0] != 0)
        
        # Intialize the temporal matrix D which is such that its of shape (U, U - 1) and fill the main diagonal with -1's and the second diagonal with 1's such that XD = [x2 - x1, x3 - x2, ...., xN - XN-1]
        D = torch.zeros((U, U - 1),device = torch.device(self.device), requires_grad = False)
        D.fill_diagonal_(-1)
        D[1:].fill_diagonal_(1)
        D = to_var(D, self.CalInGPU)

        # Intialize D_tilda (eq 9 from original ADMM paper) as the kronecker product of DD* with I_M identity matrix where * represents transpose, M equals 49
        Dtilda = torch.kron(torch.matmul(D, D.T), torch.eye(H, device = torch.device(self.device), requires_grad = self.CalInGPU))

        # Initalize the scaled lagrange multiplier of size (49, 60). Note: Here we dont see any concept of learning alternative auxillary matrices as we do in convmc (W and B)
        P = to_var(torch.ones((self.params['size1'], self.params['size2']), requires_grad = True) * self.params['initial_P'], self.CalInGPU)

        # Then forward pass to the ISTA cell, the data tensor, the mask, D_tilda, and lagrange multiplier.
        ans = self.filter([data, entries_mask, Dtilda, P])
        data = ans[0]
        return ans

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

class UnfoldedNet2dC_convmc(nn.Module):
    def __init__(self, params = None):
        super(UnfoldedNet2dC_convmc, self).__init__()

        # Note: Constants that dont change througout the layers i.e. coef_mu_inverse and those that change are initalized to shape (num_layers, ) and are leanable like y1
        self.layers = params['layers']
        self.kernel = params['kernel']
        self.CalInGPU = params['CalInGPU']
        self.coef_mu_inverse = to_var(torch.tensor(params['coef_mu_inverse'], dtype = torch.float), self.CalInGPU)
        self.mu_inverse = to_var(torch.ones(self.layers, requires_grad = True) * params['initial_mu_inverse'], self.CalInGPU) # tensor([0., 0., 0., 0., 0.], device='cuda:0')
        self.y1 = nn.Parameter(to_var(torch.ones((params['size1'], params['size2']), requires_grad = True) * params['initial_y1'], self.CalInGPU)) # A (49, 60) shape tensor with each (i, j) index containing
        # Get W and B matrices and pass it to the ISTA cell

        # input_channels = 1
        # output_channels = 1  # Use 1 for W and 1 for B
        # model = LearnableMatrices(input_channels, output_channels)

        # W, B = model(y1)
        
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.filter = self.makelayers()
        
    # For each of the layers in the unfolded nn, we create ISTA cell block with the kernel, mu_inverse, reosy1,....., all of the parameters. And then combine all layers (5 in our case) using
    # nn.Sequential forming the overall architecture
    def makelayers(self):
        filt = []
        for i in range(self.layers):
            filt.append(ISTACell_convmc(i, self.kernel[i], self.mu_inverse[i], self.coef_mu_inverse, self.CalInGPU))
        return nn.Sequential(*filt)

    # The Forward Pass recieves inputs as a list i.e. model([inputs1])
    def forward(self, x):
        data = to_var(torch.zeros([2] + list(x[0].shape)), self.CalInGPU)
        data[0] = x[0]
        # The data matrix is x[0]
        # H, U = x[0].shape
        # print(H, U)

        # entries_mask = ~(torch.isnan(data[0]))
        entries_mask = (data[0] != 0)
        ans = self.filter([data, entries_mask, self.y1])
        data = ans[0]
        return ans

    def getexp_LS(self):
        mu_inverse = self.mu_inverse
        y1 = self.y1
        coef_mu_inverse = self.coef_mu_inverse
        exp_L = self.sig(self.mu_inverse) * coef_mu_inverse

        if torch.cuda.is_available():
          mu_inverse = mu_inverse.cpu().detach().numpy()
          y1 = y1.cpu().detach().numpy()
          coef_mu_inverse = coef_mu_inverse.cpu().detach().numpy()
          exp_L = exp_L.cpu().detach().numpy()
        else:
          mu_inverse = mu_inverse.detach().numpy()
          y1 = y1.detach().numpy()
          coef_mu_inverse = coef_mu_inverse.detach().numpy()
          exp_L = exp_L.detach().numpy()
        return mu_inverse, y1, exp_L

# Conv2D class
class Conv2dC(nn.Module):
    def __init__(self, kernel):
        super(Conv2dC, self).__init__()

        # Given a kernel size of 2 dimensions, we calculate the padding through the formula (k[0] - 1)/2
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
        nh = int(n) # 60

        # This line creates a new tensor xR by slicing the input tensor along the columns dimension (0:n). The None adds an extra dimension, making xR a 4-dimensional tensor with size (1, 1, H, W).
        # The 1's make sure the consistency with the conv operation which is expecting a in_channels, out_channels, which are set to 1
        xR = x[None, None, :, 0:n].clone()
        xR = self.convR(xR)
        # Removing the extra dimension
        xR = xR.squeeze()
        x = xR
        return x

# Creating a class which recives the lagrange multiplier y of shape (49, 60) and ouputs two learnable matrices W and B both of shape (49, 60) as in eq (9) of paper.
# The process for getting the output is very similar to Conv2dC above where we add extra dimensions for convolution and then after the operation we remove those extra dimensions

class LearnableMatrices(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LearnableMatrices, self).__init__()
        self.conv_W = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 3, stride = 1, padding = 1)
        self.conv_B = nn.Conv2d(in_channels = input_channels, out_channels = output_channels, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, y):
        temp = y[None, None, :, 0:y.shape[-1]]
        W = self.conv_W(temp)
        B = self.conv_B(temp)
        return W.squeeze(), B.squeeze()

# ISTA Cell Class
class ISTACell_convmc(nn.Module):
    # Constuctor initializes the parameters that have to be learnt per layer like mu_inverse and some that stay constant throughout like coef_mu_inverse
    def __init__(self, layer_num, kernel, mu_inverse, coef_mu_inverse, CalInGPU):
        super(ISTACell_convmc,self).__init__()
        self.conv1 = Conv2dC(kernel)
        self.conv2 = Conv2dC(kernel)
        self.conv3 = Conv2dC(kernel)
        self.CalInGPU = CalInGPU
        self.mu_inverse = nn.Parameter(mu_inverse)
        self.layer_num = layer_num
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.W = nn.Parameter(torch.ones((150, 300), device = torch.device(self.device), requires_grad = CalInGPU))
        self.B = nn.Parameter(torch.zeros((150, 300), device = torch.device(self.device), requires_grad = CalInGPU))

        # self.W = to_var(nn.Parameter(torch.ones((49, 60), requires_grad = True)), self.CalInGPU)
        # self.B = to_var(nn.Parameter(torch.zeros((49, 60), requires_grad = True)), self.CalInGPU)
                
        self.coef_mu_inverse = coef_mu_inverse
        
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        self.softshrink = None

    # Performs Singular Value Thresholding on x with the threshold 'th' - essentially takes the diagonal matrix from the svd of x, pulls each of the singular values towards zero.
    def svtC(self, x, th):
        m, n = x.shape
        U, S, V = torch.svd(x)
        tmp = to_var(torch.zeros_like(S), self.CalInGPU)
        tmp = S - (th * S[0])
        tmp2 = to_var(torch.zeros_like(tmp), self.CalInGPU)
        tmp2 = self.relu(tmp)
        VS = to_var(torch.zeros(n, m), self.CalInGPU)
        stmp = to_var(torch.zeros(m), self.CalInGPU)
        stmp[0 : tmp2.shape[-1]] = tmp2
        minmn = min(m, n)
        VS[:, 0:minmn] = V[:, 0:minmn]
        y = torch.zeros_like(x)
        y = (U * stmp) @ VS.t()
        return y

    def forward(self, lst):
        # Step 1: Get data of shape (2, 49, 60) (which contains lowrank matrix at index 0) and the boolean mask (specifiying missing values if any) at index 1 of list and largrange multipler at index 2
        data = lst[0]
        entries_mask = lst[1]
        th_y1 = lst[2]

        # Step 2: seperately identify the two indices in data as x --> index 0 and L --> index 1. Note x is the sample/noised(maybe) version of the groundtruth. L as of now is just zeros of shape (49, 60).
        # Further get the dimensions of x
        x = data[0]
        L = data[1]
        H, U = x.shape

        # Step 3: Scale mu_inverse by sigmoid and coef_mu_inverse to be used later for the iterative step. Also get some parameters
        th_mu_inverse = self.sig(self.mu_inverse) * self.coef_mu_inverse
        th_W = self.W
        th_B = self.B

        # Step 4: Carry out Eq (9)
        # Part 1: Performs the convolutional operation on the matrix x by replacing it with 0 where it contains missing values and then added it with L (which initially is just zeros of shape (49, 60))
        part1_eq = L + self.conv1(torch.where(entries_mask, x, torch.tensor(0.0, device = torch.device(self.device), requires_grad = self.CalInGPU)))
        # Part 2: Performs convolutional operation on the matrix L by first replacing it with 0 where it contains missing values then added with part1
        part2_eq = self.conv2(torch.where(entries_mask, L, torch.tensor(0.0, device = torch.device(self.device), requires_grad = self.CalInGPU)))
        # Part 3: Performs element wise multiplication between W (scaled by 1/mu) and lagrange multiplier (again first mapped its missing values to 0) + a bias

        part3_eq = torch.mul(torch.where(entries_mask, th_y1.view(H, U), torch.tensor(0.0, device = torch.device(self.device), requires_grad = self.CalInGPU)), th_W * th_mu_inverse) + th_B

        # Overall result:
        Ltmp = part1_eq + part2_eq + part3_eq # shape (49, 60)

        # Step 5: Perform singular value thresholding on Ltemp
        L = self.svtC(Ltmp.view(H, U), th_mu_inverse)

        # Step 6: Update lagrange multiplier using gradient ascent

        # Intialize an empty matrix same size as y1 and then assign it the current largrange multiplier
        y1tmp = torch.zeros((H, U), device = torch.device(self.device), requires_grad = True)
        y1tmp = th_y1.clone()
        # Perform Step 4 of Algorithm 1 - self explanatory (however its mu in eq not 1/mu - ask shoaib bhai)
        y1tmp = y1tmp + (1/th_mu_inverse) * (torch.where(entries_mask, x, torch.tensor(0.0, device = torch.device(self.device), requires_grad = self.CalInGPU)) -
                                              torch.where(entries_mask, L, torch.tensor(0.0, device = torch.device(self.device), requires_grad = self.CalInGPU)))

        # Step 7: Update index 1 of data with the updated L and then pass the mask, data, and updated lagrange multiplier to the second layer/second ISTA cell as a list
        data[1] = L.view(H, U)
        return [data, entries_mask, y1tmp]
