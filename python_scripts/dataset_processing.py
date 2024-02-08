import numpy as np
import torch
import torch.utils.data as data
from python_scripts import generate_synthetic_data
from python_scripts import format_data

ROOT = 'C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/ConvHuberMC/HuberMC_Data'
# ROOT = 'C:/Users/HP/Documents/GitHub/ConvHuberMC-Net/HuberMC_Data'
# def preprocess(L, D, size1, size2, size3):

#     A = max(np.max(np.abs(L)), np.max(np.abs(D)))
#     if A == 0:
#         A = 1
#     L = L/A
#     D = D/A

#     return L, D

class ImageDataset(data.Dataset):


    def __init__(self, NumInstances, shape, split, transform = None):
        self.shape = shape

        # dummy image loader
        images_L = torch.zeros(tuple([NumInstances]) + self.shape) # --> shape: (400, 49, 60)
        images_D = torch.zeros(tuple([NumInstances]) + self.shape) # --> shape: (400, 49, 60)

        # TRAIN
        if split == 0:
            for n in range(NumInstances):
                L = np.load(ROOT + '/lowrank/train/L_mat_MC_train' + str(n + 1) + '.npy')
                print(L.shape)
                D = np.load(ROOT + '/groundtruth/train/ground_mat_MC_train' + str(n + 1) + '.npy')
                # L, D = preprocess(L, D, None, None, None)

                images_L[n] = torch.from_numpy(L)
                images_D[n] = torch.from_numpy(D)

         # TEST
        if split == 1:
            for n in range(NumInstances):
                L = np.load(ROOT + '/lowrank/test/L_mat_MC_test' + str(n + 1) + '.npy')
                D = np.load(ROOT + '/groundtruth/test/ground_mat_MC_test' + str(n + 1) + '.npy')
                # L, D = preprocess(L, D, None, None, None)

                images_L[n] = torch.from_numpy(L)
                images_D[n] = torch.from_numpy(D)


        self.transform = transform
        self.images_L = images_L
        self.images_D = images_D

    def __getitem__(self, index):
        L = self.images_L[index]
        D = self.images_D[index]
        return L, D

    def __len__(self):
        return len(self.images_L)


def get_dataloaders(params_net, hyper_param_net, sampling_rate, db):
    M_train, M_Omega_train, M_test, M_Omega_test = generate_synthetic_data.generate(
        params_net['size1'], params_net['size2'], params_net['rank'], 
        hyper_param_net['TrainInstances'], hyper_param_net['ValInstances'], sampling_rate, db)

    # Format and Save Data
    format_data.format(M_train, M_Omega_train, M_test, M_Omega_test)

    # Create DataLoaders
    train_dataset = ImageDataset(40, (params_net['size1'], params_net['size2']), 0)
    train_loader = data.DataLoader(train_dataset, batch_size = 5, shuffle = True)
    test_dataset = ImageDataset(20, (params_net['size1'], params_net['size2']), 1)
    test_loader = data.DataLoader(test_dataset, batch_size = 5)

    return train_loader, test_loader