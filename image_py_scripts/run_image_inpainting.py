import numpy as np
from skimage import io, color, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import os
import matplotlib.pyplot as plt
from image_py_scripts import gaussian_noise
from pathlib import Path

ROOT = 'C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/convmc-net/'
DATA_PATH = os.path.join(ROOT, 'Image_Inpainting_Data')
SYNTHETIC_DATA_PATH = os.path.join(ROOT, 'Image_Inpainting_Dataset')


# Function to read and preprocess images
def read_and_preprocess_image(image_path, r, c):
    image = io.imread(image_path)
    image = color.rgb2gray(img_as_float(image))
    image = np.resize(image, (r, c))
    return image

# Function to add GMM noise to the image
def add_gmm_noise(image, per, dB):
    r, c = 150, 300
    array_Omega = np.random.choice([1, 0], (r, c), True, [per, 1 - per])
    M_Omega = np.multiply(image, array_Omega)

    omega = np.where(array_Omega == 1)

    noise = gaussian_noise.gaussian_noise(M_Omega[omega], 'GM', dB)
    Noise = np.zeros(M_Omega.shape)
    Noise[omega] = noise
    M_Omega = M_Omega + Noise

    return torch.from_numpy(M_Omega)

# Function to evaluate image inpainting
def evaluate_inpainting(M, M_Omega, model_path, model):
    model_path = model_path.replace('\\', '/')
    model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
    model.eval()
    with torch.no_grad():
        M_omega_reconstructed = (model([M_Omega]))[0][1]
    M_omega_reconstructed = M_omega_reconstructed.numpy()
    M_omega_reconstructed = M_omega_reconstructed.astype(np.float32)
    PSNR = psnr(M, M_omega_reconstructed)
    SSIM = ssim(M, M_omega_reconstructed, data_range = 1)
    return PSNR, SSIM

# Function to plot PSNRs and SSIMs
def plot_metrics(PSNRs, SSIMs, save_directory):
    # Plotting PSNRs
    plt.figure(figsize = (10, 6))
    for idx, psnr_values_per_model in enumerate(PSNRs):
        plt.plot(range(1, 9), psnr_values_per_model, label = f"Sampling {20 + 10 * idx}%")
    plt.xlabel("Image")
    plt.ylabel("PSNR")
    plt.title("PSNR Comparison (5DB)")
    plt.legend()
    plt.grid(True)
    psnr_plot_path = os.path.join(save_directory, "psnr_plot.png")
    plt.savefig(psnr_plot_path)
    plt.show()

    # Plotting SSIMs
    plt.figure(figsize=(10, 6))
    for idx, ssim_values_per_model in enumerate(SSIMs):
        plt.plot(range(1, 9), ssim_values_per_model, label = f"Sampling {20 + 10 * idx}%")
    plt.xlabel("Image")
    plt.ylabel("SSIM")
    plt.title("SSIM Comparison (5DB)")
    plt.legend()
    plt.grid(True)
    ssim_plot_path = os.path.join(save_directory, "ssim_plot.png")
    plt.savefig(ssim_plot_path)
    plt.show()

# Function to orchestrate the image inpainting pipeline
def run_image_inpainting_pipeline(file_paths, save_directory, model):
    # Parameters
    r, c, rak = 150, 300, 10
    dB = 5
    per = 0.5
    PSNRs = []
    SSIMs = []

    # Loop through models and images
    for idx, file_path in enumerate(file_paths):
        psnr_per_model = []
        ssim_per_model = []
        for i in range(1, 9):
            # Read Image and convert into black and white and reshape into 150 x 300
            image_path = os.path.join(DATA_PATH, f"{i}.jpg")
            M = read_and_preprocess_image(image_path, r, c)

            # Add GMM noise
            M_Omega = add_gmm_noise(M, per, dB)
            maxiter = 50

            PSNR, SSIM = evaluate_inpainting(M, M_Omega, file_path, model)
            print(f'Model: {"ConvMC-Net"}, PSNR of Image {i} = {PSNR}, SSIM of Image {i} = {SSIM}')

            psnr_per_model.append(PSNR)
            ssim_per_model.append(SSIM)

        PSNRs.append(psnr_per_model)
        SSIMs.append(ssim_per_model)

    # Plot PSNRs and SSIMs
    plot_metrics(PSNRs, SSIMs, save_directory)
