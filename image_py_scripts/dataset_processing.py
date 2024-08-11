import numpy as np
import torch
import torch.utils.data as data
from image_py_scripts import generate_synthetic_data
from image_py_scripts import format_data

from image_py_scripts import utils
from pathlib import Path
import os

# DATA_ROOT = Path('C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/convmc-net/Image_Inpainting_Data/BSDS300/images')
DATA_ROOT = Path('C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/ConvHuberMC/Image_Inpainting_Data/BSDS300/images')
train_dir = DATA_ROOT / 'train'
test_dir = DATA_ROOT / 'test'

class SyntheticDataset(data.Dataset):
    def __init__(self, NumInstances, shape, split, ROOT, transform = None):
        self.shape = shape

        # dummy image loader
        images_L = torch.zeros(tuple([NumInstances]) + self.shape) # --> shape: (400, 49, 60)
        images_D = torch.zeros(tuple([NumInstances]) + self.shape) # --> shape: (400, 49, 60)

        # TRAIN
        if split == 0:
            for n in range(NumInstances):
                L = np.load(ROOT + '\\lowrank/train/L_mat_MC_train' + str(n + 1) + '.npy')
                D = np.load(ROOT + '\\groundtruth/train/ground_mat_MC_train' + str(n + 1) + '.npy')
                # L, D = preprocess(L, D, None, None, None)

                images_D[n] = torch.from_numpy(D)
                images_L[n] = torch.from_numpy(L)

         # TEST
        if split == 1:
            for n in range(NumInstances):
                L = np.load(ROOT + '\\lowrank/test/L_mat_MC_test' + str(n + 1) + '.npy')
                D = np.load(ROOT + '\\groundtruth/test/ground_mat_MC_test' + str(n + 1) + '.npy')
                # L, D = preprocess(L, D, None, None, None)
                
                images_D[n] = torch.from_numpy(D)
                images_L[n] = torch.from_numpy(L)


        self.transform = transform
        self.images_L = images_L
        self.images_D = images_D

    def __getitem__(self, index):
        L = self.images_L[index]
        D = self.images_D[index]
        return L, D

    def __len__(self):
        return len(self.images_D)



class ImageDataset(data.Dataset):
    def __init__(self, shape, split, path, transform = None, dust = True):
        self.shape = shape
        self.dust = dust
        
        # TRAIN
        if split == 0:
            # dummy image loader
            images_L = torch.zeros(tuple([200]) + self.shape) # --> shape: (200, shape)
            images_D = torch.zeros(tuple([200]) + self.shape) # --> shape: (200, shape)
            for n in range(200):
                if not dust:
                    L = np.load(os.path.join(path, f'lowrank/lowrank_image_MC_train_' + str(n) + '.npy'))
                    images_L[n] = torch.from_numpy(L)
                D = np.load(os.path.join(path, f'groundtruth/ground_image_MC_train_' + str(n) + '.npy'))
                images_D[n] = torch.from_numpy(D)

         # TEST
        if split == 1:
            images_L = torch.zeros(tuple([100]) + self.shape) # --> shape: (200, shape)
            images_D = torch.zeros(tuple([100]) + self.shape) # --> shape: (200, shape)
            for n in range(100):
                if not dust:
                    L = np.load(os.path.join(path, f'lowrank/lowrank_image_MC_test_' + str(n) + '.npy'))
                    images_L[n] = torch.from_numpy(L)
                D = np.load(os.path.join(path, f'groundtruth/ground_image_MC_test_' + str(n) + '.npy'))
                images_D[n] = torch.from_numpy(D)


        self.transform = transform
        if not dust:
            print('gg')
            self.images_L = images_L
        self.images_D = images_D

    def __getitem__(self, index):
        if not self.dust:
            L = self.images_L[index]
        D = self.images_D[index]
        return D if self.dust else L, D

    def __len__(self):
        return len(self.images_D)


def get_dataloaders(params_net, hyper_param_net, sampling_rate, db, ROOT, synthetic = True, dust = True):
    if synthetic:
        M_train, M_Omega_train, M_test, M_Omega_test = generate_synthetic_data.generate(
            params_net['size1'], params_net['size2'], params_net['rank'], 
            hyper_param_net['TrainInstances'], hyper_param_net['ValInstances'], sampling_rate, db)

        # Format and Save Data
        format_data.format(M_train, M_Omega_train, M_test, M_Omega_test, ROOT)

        # Create DataLoaders
        train_dataset = SyntheticDataset(hyper_param_net['TrainInstances'], (params_net['size1'], params_net['size2']), 0, ROOT)
        train_loader = data.DataLoader(train_dataset, batch_size = hyper_param_net['BatchSize'], shuffle = True)
        test_dataset = SyntheticDataset(hyper_param_net['ValInstances'], (params_net['size1'], params_net['size2']), 1, ROOT)
        test_loader = data.DataLoader(test_dataset, batch_size = hyper_param_net['ValBatchSize'])

    else:
        
        utils.make_imgs(split = 'train', shape = (params_net['size1'], params_net['size2']), sampling_rate = sampling_rate, dB = db)
        utils.make_imgs(split = 'test', shape = (params_net['size1'], params_net['size2']), sampling_rate = sampling_rate, dB = db) 

        train_dataset = ImageDataset((params_net['size1'], params_net['size2']), 0, train_dir, dust = dust)
        train_loader = data.DataLoader(train_dataset, batch_size = hyper_param_net['BatchSize'], shuffle = True)
        test_dataset = ImageDataset((params_net['size1'], params_net['size2']), 1, test_dir, dust = dust)
        test_loader = data.DataLoader(test_dataset, batch_size = hyper_param_net['ValBatchSize'])
    
    return train_loader, test_loader