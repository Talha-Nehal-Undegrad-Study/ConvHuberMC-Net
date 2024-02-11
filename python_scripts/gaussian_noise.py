import numpy as np

def gaussian_noise(signal, model, SNR):
    signal_size = signal.shape
    total_samples = 1

    for dim_idx in range(len(signal_size)):
        total_samples = total_samples * signal_size[dim_idx]

    signal_power = (np.sum(np.abs(signal.flatten()) ** 2)) / total_samples

    s_n = 10 ** (SNR / 10)

    if model == 'GW':
        noise_power = signal_power/s_n
        noise = np.sqrt(noise_power) * np.random.normal(size = signal_size)
    else:
        c_1 = 0.9
        c_2 = 0.1

        sigma_v_2 = signal_power / s_n
        sigma_1 = np.sqrt(sigma_v_2 / (c_1 + ((10 ** 2) * c_2)))

        p = c_1/1

        sigma = np.array([[100 * sigma_1, sigma_1]])

        flag = np.random.binomial(1, p, size = signal_size)

        noise = np.multiply(sigma[0, 0] * np.random.normal(size = signal_size), (np.ones(shape = signal_size) - flag)) + np.multiply(sigma[0, 1] * np.random.normal(size = signal_size), flag)
    
    return noise