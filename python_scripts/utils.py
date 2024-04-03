import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import re

c = 1.345

def get_nonzeros(matrix):
    
    return np.array(np.nonzero(matrix))

def rho(x):
    x = torch.as_tensor(x)
    subset = torch.abs(x) <= c
    return 0.5 * x**2 * subset + (~subset) * (c * torch.abs(x) - 0.5 * c**2)

def psi(x):
    x = torch.as_tensor(x)
    subset = torch.abs(x) <= c
    return subset * x + (~subset) * c * torch.sign(x)

def alpha():
    return 0.7102


def get_all_text_files(folder_path):
    # Create a pattern to match all .txt files in the folder and subfolders
    pattern = f"{folder_path}/**/*.txt"
    # Use glob to find all files matching the pattern, with recursive search
    return glob.glob(pattern, recursive=True)

# Usage example
"""
folder_path = 'C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/ConvHuberMC/HuberMC_Data'
text_files = get_all_text_files(folder_path)
print(text_files, len(text_files))
"""

def extract_min_losses(file_paths):
    losses = []
    for path in file_paths:
        with open(path, 'r') as file:
            text = file.read()
            # match = re.search(r'Min Loss = ([\d\.]+e[+\-]\d+)', text)
            match = re.search(r'Epoch \[20/20\], Mean Training Loss:.*?, Mean Validation Loss:(\d+\.\d+e[+\-]\d+)', text)
            if match:
                loss = float(match.group(1))
                losses.append(loss)
            else:
                # Append NaN if no loss is found
                losses.append(np.nan)

    # Ensure we have exactly 28 losses to reshape into a 4x7 array
    if len(losses) == 28:
        return np.array(losses).reshape(4, 7)
    else:
        raise ValueError(f'Expected 28 loss values, but got {len(losses)}.')
    
# Example Usage
"""

# Example usage, replace 'fixed_paths' with your actual list of file paths
# fixed_paths = ['path/to/your/file1.txt', 'path/to/your/file2.txt', ...]
losses_array = extract_min_losses(text_files)
print(losses_array)
unfolded_results = losses_array.reshape(7, 4).T
"""

def normalize_paths(file_paths):
    # Iterate over the file paths and replace '//' with '/'
    # return [re.sub(r'\+', '/', path) for path in file_paths]
    return [path.replace('\\', '/') for path in file_paths]

# Example Usage
"""
# text_files = normalize_paths(text_files)
"""                         

# MAT file from MATLAB for M_Est results
# data = """
# 24.5695    0.0529    0.0307    0.0197    0.0148    0.0135    0.0107
# 11.6203    0.0391    0.0209    0.0114    0.0100    0.0077    0.0068
# 14.0508    0.0271    0.0148    0.0095    0.0084    0.0061    0.0057
# 6.2097    0.0152    0.0083    0.0049    0.0041    0.0032    0.0025
# """

# # Convert the string to a list of numbers, then reshape into a 4x7 array
# array = np.fromstring(data, sep=' ').reshape(4, 7)
# print(array)

# Plotting Results for comparison
# Sampling rates

# sampling_rates = np.arange(0.2, 0.9, 0.1)

# SNR/DB levels
# snr_db_levels = [3, 5, 6, 9]

# Plotting
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# for i in range(4):
#     ax = axs[i // 2, i % 2]
#     ax.plot(sampling_rates, array[i], marker='o', label='M-Estimation Loss')
#     ax.plot(sampling_rates, unfolded_results[i], marker='x', label='Unfolded M-Est Loss')
#     ax.set_title(f'Losses at SNR/DB = {snr_db_levels[i]}')
#     ax.set_xlabel('Sampling Rate')
#     ax.set_ylabel('Loss')
#     ax.legend()

# plt.tight_layout()
# plt.show()