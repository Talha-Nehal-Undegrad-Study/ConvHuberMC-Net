import numpy as np
import torch
from scipy.fftpack import dct
import random
import os
from python_scripts import add_gaussian_noise




# Make a function which extracts the first Q samples from the dataset in a random manner for a speicifc split (Test, Train)
def get_sample(split, Q):
    pass
    

def attention_based_algo(A, D, X, K, lambda_1, lambda_2, c):
    # A is measurment sensing matrix (m x n where n << m) and D is initialized by DCT (n x d where d >> n) and X is observed matrix (m x T)
    # Form of x_t = As_t + e_t where s_t is groundtruth (m x 1) signal and x_t is observed noisy signal (represents 1, 2, ...., T columns of X)
    # Form of s_t = Dh_t where h_t is sparse representation of s_t (d x 1) intialized to 0's
    # K is the number of outer iterations. Inner iterations are for T time periods.
    # lambda_1, lambda_2, c are hyperparameters 

    for k in range(K):
        for t in range(X.shape[1]):
            


