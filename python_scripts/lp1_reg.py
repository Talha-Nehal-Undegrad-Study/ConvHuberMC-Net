import numpy as np

def LP1(M_Omega, Omega, rak, maxiter = 500):
    """
    L1-regularized method for Matrix Completion.

    Parameters:
    - M_Omega: m x n matrix of observations/data.
    - Omega: The subset of indices in M_Omega that are observed.
    - rak: Rank for the approximation.
    - maxiter: Maximum number of iterations (default: 500).
    
    Returns:
    - Out_X: The completed matrix.
    """
    
    iter = 0
    n1, n2 = M_Omega.shape
    U = np.random.randn(n1, rak)
    V = np.zeros((rak, n2))  # Initialize V as it's used before assignment
    t = 2

    while True:
        iter += 1
        for j in range(n2):
            row_i = [i for i, val in enumerate(Omega[1, :]) if val == j]
            row = [Omega[0, x] for x in row_i]
            U_I = U[row, :]
            b_I = M_Omega[row, j]

            if iter == 1:
                W = np.eye(len(b_I))
            else:
                ksi = U_I @ V[:, j] - b_I
                W = np.diag(1 / (np.abs(ksi)**2 + 0.0001)**(1/4))

            for inner_iter in range(t):
                V[:, j] = np.linalg.pinv(U_I.T @ W.T @ W @ U_I) @ U_I.T @ W.T @ W @ b_I
                ksi = U_I @ V[:, j] - b_I
                W = np.diag(1 / (np.abs(ksi)**2 + 0.0001)**(1/4))

        for i in range(n1):
            col_i_new = [i for i, val in enumerate(Omega[0, :]) if val == i]
            col = [Omega[1, x] for x in col_i_new]
            V_I = V[:, col]
            b_I = M_Omega[i, col]

            if len(b_I) > 0:  # Ensure b_I is not empty
                ksi = U[i, :] @ V_I - b_I
                W = np.diag(1 / (np.abs(ksi)**2 + 0.0001)**(1/4))

                for inner_iter in range(t):
                    U[i, :] = b_I @ W @ W.T @ V_I.T @ np.linalg.pinv(V_I @ W @ W.T @ V_I.T)
                    ksi = U[i, :] @ V_I - b_I
                    W = np.diag(1 / (np.abs(ksi)**2 + 0.0001)**(1/4))

        X = U @ V
        if iter > maxiter:
            break

    return X
