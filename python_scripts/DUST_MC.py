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
    # A is measurment sensing matrix (m x n where m << n) and D is initialized by DCT (n x d where d >> n) and X is observed matrix (m x T)
    # Form of x_t = As_t + e_t where s_t is groundtruth (m x 1) signal and x_t is observed noisy signal (represents 1, 2, ...., T columns of X)
    # Form of s_t = Dh_t where h_t is sparse representation of s_t (d x 1) intialized to 0's
    # K is the number of outer iterations. Inner iterations are for T time periods.
    # lambda_1, lambda_2, c are hyperparameters 
    
    m, n, d, T = A.shape[0], A.shape[1], D.shape[1], X.shape[1]
    # initailize H which is d x T and is h_t's concatenated
    H = np.zeros((d, T))

    for k in range(K):
        for t in range(T):
            h_t = H[:, t]
            x_t = np.dot(A, X[:, t])
            
            # Compute G numerator and denominator
            G_numerator = np.zeros_like(h_t)
            G_denominator = 0

            for u in range(T):
                h_u_prev = H[:, u]
                beta_u = utils.beta_u(u, H)
                exponent_term = np.dot(h_u_prev.T, np.dot(D.T, np.dot(D, h_u_prev)))
                exp_value = np.exp(exponent_term)
                G_numerator += beta_u * exp_value * h_u_prev
                G_denominator += beta_u * exp_value

            # Compute y_t
            if G_denominator != 0:
                G = G_numerator / G_denominator
            else:
                G = np.zeros_like(G_numerator)
                
            y_t = lambda_2 * G
            prod = np.dot(np.dot(D.T, A.T), np.dot(A, D))

            z_t = np.dot((np.eye(d) - (1/c * prod)), y_t) + ((1/c) * np.dot(np.dot(D.T, A.T), x_t))

            h_t = utils.soft_thresholding(z_t, lambda_1/c)

            # Here you can update h_t based on y_t and other factors if needed

            # Store the updated h_t back to H
            H[:, t] = h_t
    
    S = np.dot(D, H)
    return S


