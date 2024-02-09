import numpy as np

def LP2(M_Omega, Omega, rak, maxiter = 500):
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
    V = np.zeros((rak, n2))  # Initialize V

    while True:
        iter += 1
        for j in range(n2):
            row_i = [i for i, val in enumerate(Omega[1, :]) if val == j]
            row = [Omega[0, x] for x in row_i]
            U_I = U[row, :]
            b_I = M_Omega[row, j]

            if len(row) > 0:  # Ensure row is not empty
                V[:, j] = np.linalg.pinv(U_I) @ b_I

        for i in range(n1):
            col_i_new = [i for i, val in enumerate(Omega[0, :]) if val == i]
            col = [Omega[1, x] for x in col_i_new]
            V_I = V[:, col]
            b_I = M_Omega[i, col]

            if len(col) > 0:  # Ensure col is not empty
                U[i, :] = b_I @ np.linalg.pinv(V_I)

        X = U @ V
        if iter > maxiter:
            break

    return X
