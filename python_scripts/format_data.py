import os 
import numpy as np
from pathlib import Path


# ROOT = Path('C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/ConvHuberMC/HuberMC_Data')
# ROOT = Path('C:/Users/HP/Git/ConvHuberMC-Net/HuberMC_Data')

def format(M_train, M_Omega_train, M_test, M_Omega_test, ROOT):
    os.makedirs(Path(ROOT), exist_ok = True)

    ground_or_pred_dir = (Path(ROOT) / 'groundtruth')
    os.makedirs(ground_or_pred_dir, exist_ok = True)

    lowrank_dir = (Path(ROOT) / 'lowrank')
    os.makedirs(lowrank_dir, exist_ok = True)

    ground_train_dir = (ground_or_pred_dir / 'train')
    os.makedirs(ground_train_dir, exist_ok = True)

    ground_test_dir = (ground_or_pred_dir / 'test')
    os.makedirs(ground_test_dir, exist_ok = True)

    low_train_dir = (lowrank_dir / 'train')
    os.makedirs(low_train_dir, exist_ok = True)

    low_test_dir = (lowrank_dir / 'test')
    os.makedirs(low_test_dir, exist_ok = True)

    for i in range(M_train.shape[0]):
        np.save(str(low_train_dir) + '/L_mat_MC_train' + str(i + 1) + '.npy', M_Omega_train[i])
        np.save(str(ground_train_dir) + '/ground_mat_MC_train' + str(i + 1) + '.npy', M_train[i])

    for i in range(M_test.shape[0]):
        np.save(str(low_test_dir) + '/L_mat_MC_test' + str(i + 1) + '.npy', M_Omega_test[i])
        np.save(str(ground_test_dir) + '/ground_mat_MC_test' + str(i + 1) + '.npy', M_test[i])