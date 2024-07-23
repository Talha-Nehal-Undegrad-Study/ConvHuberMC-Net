import numpy as np
import torch
from scipy.fftpack import dct
import random
import os
from python_scripts import utils



# Make a function which extracts the first Q samples from the dataset in a random mi anner for a speicifc split (Test, Train)
def get_sample(split, Q):
    pass

def attention_based_algo(A, D, X, K, lambda_1, lambda_2, c):
    """
    Perform attention-based algorithm for sparse representation and matrix recovery.
    
    Parameters:
    - A: Measurement sensing matrix of shape (m x n)
    - D: Dictionary matrix initialized by DCT of shape (n x d)
    - X: Observed matrix of shape (m x T)
    - K: Number of outer iterations
    - lambda_1, lambda_2, c: Hyperparameters
    
    Returns:
    - S: Reconstructed matrix of shape (n x T)
    """
    m, n, d, T = A.shape[0], A.shape[1], D.shape[1], X.shape[1]

    # Initialize H which is d x T
    H = np.zeros((d, T))

    # Precompute matrices that do not change within the inner loop
    DTD = np.dot(D.T, D)
    A_T = A.T
    A_TA = np.dot(A_T, A)
    D_TA_T = np.dot(D.T, A_T)
    AD = np.dot(A, D)
    prod = np.dot(D_TA_T, AD)

    for k in range(K):
        # Compute beta_u values outside the loop for efficiency
        beta_values = np.array([utils.weighted_softmax(u) for u in range(T)])

        # Compute exponent_terms
        H_T = H.T
        exponent_terms = np.einsum('ij,ij->j', np.dot(DTD, H), H)
        exp_values = np.exp(exponent_terms)

        # Compute G numerator and denominator using vectorized operations
        G_numerator = np.sum(beta_values[:, None] * exp_values[:, None] * H.T, axis=0)
        G_denominator = np.sum(beta_values * exp_values)

        # Compute y_t
        if G_denominator != 0:
            G = G_numerator / G_denominator
        else:
            G = np.zeros(d)

        y_t = lambda_2 * G

        # Compute z_t for all time periods
        I_d = np.eye(d)
        eye_minus_prod = I_d - (1 / c * prod)
        z_t = np.dot(eye_minus_prod, y_t) + (1 / c) * np.dot(D_TA_T, X)

        # Apply soft thresholding
        H = np.apply_along_axis(lambda h: utils.soft_thres(lambda_1, c, h), 0, z_t)

    # Compute S = DH
    S = np.dot(D, H)

    return S