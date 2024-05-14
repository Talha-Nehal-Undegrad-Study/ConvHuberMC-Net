import numpy as np
from image_py_scripts import gaussian_noise

# Define the parameters and 1/0 mask - sampling projection matrix
def generate(r, c, rak, num_train_instances, num_test_instances, sampling_rate, dB):
    array_Omega = np.random.choice([1, 0], (r, c), True, [sampling_rate, 1 - sampling_rate])

    # Initialize arrays to store the training and test data
    M_train = np.zeros((num_train_instances, r, c))
    M_Omega_train = np.zeros((num_train_instances, r, c))
    M_test = np.zeros((num_test_instances, r, c))
    M_Omega_test = np.zeros((num_test_instances, r, c))

    # Loop to generate training data
    for i in range(num_train_instances):
        M = np.dot(np.random.normal(size = (r, rak)), np.random.normal(size = (rak, c)))
        M_Omega = np.multiply(M, array_Omega)

        omega = np.where(array_Omega == 1)

        noise = gaussian_noise.gaussian_noise(M_Omega[omega], 'GM', dB)
        Noise = np.zeros(M_Omega.shape)
        Noise[omega] = noise.reshape(noise.shape[0], )
        M_Omega = M_Omega + Noise
        
        M_train[i, :, :] = M
        M_Omega_train[i, :, :] = M_Omega

    # Loop to generate test data
    for i in range(num_test_instances):
        M = np.dot(np.random.normal(size = (r, rak)), np.random.normal(size = (rak, c)))
        M_Omega = np.multiply(M, array_Omega)
        omega = np.where(array_Omega == 1)
        noise = gaussian_noise.gaussian_noise(M_Omega[omega], 'GM', dB)
        Noise = np.zeros(M_Omega.shape)
        Noise[omega] = noise.reshape(noise.shape[0], )
        M_Omega = M_Omega + Noise
        
        M_test[i, :, :] = M
        M_Omega_test[i, :, :] = M_Omega

    return M_train, M_Omega_train, M_test, M_Omega_test