import numpy as np

# Define the parameters and 1/0 mask - sampling projection matrix
r, c, rak = 150, 300, 10
num_train_instances = 40
num_test_instances = 20
per = 0.5  # Obervation ratio
dB = 9      # GMM Noise
array_Omega = np.random.choice([1, 0], (r, c), True, [per, 1 - per])

# Initialize arrays to store the training and test data
M_train = np.zeros((num_train_instances, r, c))
M_Omega_train = np.zeros((num_train_instances, r, c))
M_test = np.zeros((num_test_instances, r, c))
M_Omega_test = np.zeros((num_test_instances, r, c))

# Loop to generate training data
for i in range(1, num_train_instances):
    M = np.random.normal(size = (r, rak)) * np.random.normal(size = (rak, c))
    M_Omega = M * array_Omega
    omega = [(row_idx, col_idx) for row_idx, row in enumerate(array_Omega) for col_idx, value in enumerate(row) if array_Omega[row_idx, col_idx]]
    noise = Gaussian_noise(M_Omega[omega], 'GM', dB)
    Noise = np.zeros(M_Omega.shape)
    Noise[omega] = noise
    M_Omega = M_Omega + Noise
    
    M_train[i, :, :] = M
    M_Omega_train[i, :, :] = M_Omega

# Loop to generate test data
for i in range(1, num_test_instances):
    M = np.random.normal(size = (r, rak)) * np.random.normal(size = (rak, c))
    M_Omega = M .* array_Omega
    omega = [(row_idx, col_idx) for row_idx, row in enumerate(array_Omega) for col_idx, value in enumerate(row) if array_Omega[row_idx, col_idx]]
    noise = Gaussian_noise(M_Omega[omega], 'GM', dB)
    Noise = np.zeros(M_Omega.shape)
    Noise[omega] = noise
    M_Omega = M_Omega + Noise
    
    M_test[i, :, :] = M
    M_Omega_test[i, :, :] = M_Omega

# Save the data to .mat files
# save('M_train1.mat', 'M_train')
# save('M_Omega_train1.mat', 'M_Omega_train')
# save('M_test1.mat', 'M_test')
# save('M_Omega_test1.mat', 'M_Omega_test')