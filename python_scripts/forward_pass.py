class Huber(nn.Module):
    def __init__(self, epsilon, sigma, c = 1.345, lamda = 1, mu = 0, hubreg_iters = 2, layers = 3):
        super(Huber, self).__init__()
        # learnables
        self.c = nn.Parameter(c)
        self.lamda = nn.Parameter(lamda)
        self.mu = nn.Parameter(mu)

        # non-learnables
        self.epsilon = epsilon
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

    def hubreg(self, beta, X, y):
        # runs Algorithm 1 from Block-Wise Minimization-Majorization Algorithm for Huber'S Criterion: Sparse Learning And Applications
        
        alpha = (0.5 * (self.c**2) * (1 - stats.chi2.cdf(self.c**2, df = 1))) + (0.5 * stats.chi2.cdf(self.c**2, df = 3))
        X_plus = torch.inverse(X.t() @ X) @ X.t()
        
        for _ in range(self.hubreg_iters):
            r = y - (X @ beta)
            tau = torch.norm(self.hub_deriv(r / self.sigma)) / ((2 * len(y) * alpha)**0.5)
            self.sigma *= tau**self.lamda
            delta = X_plus @ (self.hub_deriv(r / self.sigma) * self.sigma)
            beta += self.mu * delta
        
        return beta

    def forward(self, U, V, X):
        # runs Algorithm 1 from Robust M-Estimation Based Matrix Completion
        
        self.U = U
        self.V = V
        self.X = X

        for _ in range(self.layers):
            for j in range(self.V.shape[1]):
                rows = self.get_rows(self.U[:, j]) # row indices for jth column
                self.V[:, j] = self.hubreg(self.V[:, j], self.U[rows, :], self.X[rows, j]) # beta: (r, 1), X: (j_i, r), y: (j_i, 1)

            for i in range(self.U.shape[1]):
                rows = self.get_rows(self.V[i, :]) # column indices for ith row
                self.U[i, :] = self.hubreg(self.U[i, :].t(), self.V[:, rows].t(), self.X[i, rows].t()).t() # beta: (r, 1), X: (j_i, r), y: (j_i, 1)
