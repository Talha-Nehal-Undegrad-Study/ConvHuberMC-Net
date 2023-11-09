import numpy as np
import subprocess
import sys
import os
import shutil
import torch
import torch.utils.data as data
from torch import nn

try:
    import convmc, dataset_processing, logs_and_results, training
except ImportError:
    print("[INFO] Cloning the repository and importing convmc & dataset_preprocessing script...")
    subprocess.run(["git", "clone", "https://github.com/TalhaAhmed2000/convmc-net.git"])
    subprocess.run(["mv", "convmc-net/python_scripts", "py_scripts"])
    sys.path.append('py_scripts')
    import convmc, dataset_processing, logs_and_results, training


from convmc import to_var, UnfoldedNet2dC_convmc, UnfoldedNet3dC_admm
from dataset_processing import ImageDataset
from logs_and_results import get_current_time, get_noise_str, get_q_str, make_predictions_dir
from training import get_hyperparameter_grid, get_model, train_step, test_step

ROOT = '/home/gcf/Desktop/Talha_Nehal Sproj/Tahir Sproj Stuff/SPROJ_ConvMC_Net/Sensor_Data'

# Another small helper function 'get_model_from_dict' which takes a model diretory and returns a model from it
def get_model_from_dict(model_dict_path, model_obj, device):
  # Load the model dictionary
  model_state_dict = torch.load(model_dict_path, map_location = torch.device(device))
  # Load the state dictionary into your model
  model_obj.load_state_dict(model_state_dict)
  # return model
  return model_obj.to(device)

def make_and_store_predictions(model_dict_path, q, sigma, params_net, hyper_param_net, train_loader, val_loader, device, SESSION):
    # Get train and test dir to store predictions in
    ProjectName = 'Best_Model_Predictions' + ' ' + get_current_time() + ' ' + hyper_param_net['Model'] + ' ' + 'Sampling Rate: ' + get_q_str(q) + ' and Noise Variance ' + get_noise_str(sigma)
    train_dir, test_dir = make_predictions_dir(ProjectName, q, sigma, 'Predictions', params_net, hyper_param_net, SESSION)
    CalInGPU = params_net['CalInGPU']
    # Get model from dict
    # Create an instance of your model class
    model = UnfoldedNet2dC_convmc(params_net)
    model = get_model_from_dict(model_dict_path, model, device)
    # Put model in eval mode
    model.eval()

    # Perform Inference on Train Dataset
    with torch.inference_mode():
      for batch, (D, L) in enumerate(train_loader):
        for mat in range(hyper_param_net['BatchSize']):
          # Get inputs and targets and forward pass
          inputsv1 = to_var(D[mat], CalInGPU)
          targets_Lv = to_var(L[mat], CalInGPU)
          lst_2 = model([inputsv1])  # Forward
          # Get predicted L and save it in the corresponding dir
          pred_L = (lst_2[0][1]).cpu()
          np.save(train_dir + '/pred_mat_MC_train' + str(batch) + '.npy', pred_L)

    # Perform Inference on Test Dataset
    with torch.inference_mode():
      for batch, (D, L) in enumerate(val_loader):
        for mat in range(hyper_param_net['ValBatchSize']):
          # Get inputs and targets and forward pass
          inputsv1 = to_var(D[mat], CalInGPU)
          targets_Lv = to_var(L[mat], CalInGPU)
          lst_2 = model([inputsv1])  # Forward
          # Get predicted L and save it in the corresponding dir
          pred_L = lst_2[0][1].cpu()
          np.save(test_dir + '/pred_mat_MC_test' + str(batch) + '.npy', pred_L)

# Another helper function which when given the same arguement as make_and_store_predictions except that of train and test dir, performs inference and gets the train_loss and validation_loss mean
# Note: This is essentially what our functions train_step and test_step are doing so we will be using those

def evaluate_each_model(model_dict_path, train_loader, val_loader, CalInGPU, param_net, hyper_param_net, device):
    # Get model from dict
    if hyper_param_net['Model'] == 'ConvMC-Net':
        model = UnfoldedNet2dC_convmc(param_net)
        model = get_model_from_dict(model_dict_path, model, device)
        CalInGPU = param_net['CalInGPU']
        
        # Set up loss and optimizer
        floss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = hyper_param_net['Lr'])
        scheduler2 =  torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.97, verbose = True)
        
        # Get train loss mean from train step
        loss_mean, loss_lowrank_mean = train_step(model, train_loader, floss, optimizer, CalInGPU, hyper_param_net['Alpha'], hyper_param_net['TrainInstances'], hyper_param_net['BatchSize'], inference = True)
        # Get test loss mean from test step
        loss_val_mean, loss_val_lowrank_mean = test_step(model, val_loader, floss, optimizer, CalInGPU, hyper_param_net['Alpha'], hyper_param_net['TrainInstances'], hyper_param_net['ValBatchSize'])
        # Return the tuple of loss_lowrank_mean and loss_val_lowrank_mean
        return (loss_lowrank_mean, loss_val_lowrank_mean)
    else:
        model = UnfoldedNet3dC_admm(param_net)
        model = get_model_from_dict(model_dict_path, model, device)
        CalInGPU = param_net['CalInGPU']
    
        # Set up loss and optimizer
        floss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = hyper_param_net['Lr'])
        scheduler2 =  torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma = 0.97, verbose = True)
        # Get train loss mean from train step
        loss_mean, loss_lowrank_mean = train_step(model, train_loader, floss, optimizer, CalInGPU, hyper_param_net['Alpha'], hyper_param_net['TrainInstances'], hyper_param_net['BatchSize'], inference = True)
        # Get test loss mean from test step
        loss_val_mean, loss_val_lowrank_mean = test_step(model, val_loader, floss, optimizer, CalInGPU, hyper_param_net['Alpha'], hyper_param_net['TrainInstances'], hyper_param_net['ValBatchSize'])
        # Return the tuple of loss_lowrank_mean and loss_val_lowrank_mean
        return (loss_lowrank_mean, loss_val_lowrank_mean)

# Another helper function when giving a dictionary where the key's are a string (specifying the path to the model) and the value is a tuple of size 2 containing train and val_loss. The function will
# find the index in the dictionary where the tuple has minimum ratio

def find_min_train_val_loss(dict_loss):
  min_loss = float('inf')  # Initialize with positive infinity
  min_loss_index = None

  for index, (numerator, denominator) in enumerate(dict_loss.values()):
      loss = denominator

      if loss < min_loss:
          min_loss = loss
          min_loss_index = index

  return min_loss_index, min_loss

# Another helper function. Given a session, q, sigma, and the model, we get all the models made that session, perform inference on it and find best performing model of that session and then rename that
# as 'best_model.....'

def search_and_save_best_model(SESSION, params_net, q, sigma, model, device):

    # Get param and hyperparam_net
    hyper_param_net = get_hyperparameter_grid(model, TrainInstances = 400, ValInstances = 68, BatchSize = 20, ValBatchSize = 4, Alpha = 1.0, num_epochs = 40, learning_rate = 0.012)
    CalInGPU = params_net['CalInGPU']
    
    # Get the train and the val dataloaders
    train_dataset = ImageDataset(round(hyper_param_net['TrainInstances']), (49, 60), 0, q, sigma)
    train_loader = data.DataLoader(train_dataset,batch_size = hyper_param_net['BatchSize'], shuffle = True)
    val_dataset = ImageDataset(round(hyper_param_net['ValInstances']), (49, 60), 1, q, sigma)
    val_loader = data.DataLoader(val_dataset, batch_size = hyper_param_net['ValBatchSize'], shuffle = True)
    
    # Initalize an empty dictionary which will contain the path to all complete models run in a particular session as keys and values of tuples of size 2
    dict_loss = {}
    q_exp = f'/Q is {q * 100}%'
    noise_exp = f'/Noise Variance {float(sigma)}'
    
    # Get the folder where all the model dicts are saved
    model_path = ROOT + q_exp + noise_exp + '/Saved Models - Dict/' + model + '/' + SESSION
    
    # Get all files in the folder
    all_models = os.listdir(model_path)
    
    # We want to get all models which are complete i.e. have run all 40/40 epochs
    filter_str = '40_out_of_40'
    complete_models_lst = [file for file in all_models if filter_str in file]
    
    # Now taking each model and joining it with the model_path, we will perform inference and find which gives the best ratio of val_loss to train_loss and we will declare that as the best model
    for model_file in complete_models_lst:
        final_model_path = os.path.join(model_path, model_file)
        # Now get the tuple of train_loss and val_loss
        loss_tuple = evaluate_each_model(final_model_path, train_loader, val_loader, CalInGPU, params_net, hyper_param_net, device)
        # Now store this in the dictionary defined in the beginning
        dict_loss[final_model_path] = loss_tuple
    
    # After performing inference on all models of a specific session and storing their results we get the path to the model which has the minimum val_loss to train_loss ratio
    min_loss_index, min_loss = find_min_train_val_loss(dict_loss)
    
    # Get the best model path
    best_model_path = (list(dict_loss.keys()))[min_loss_index]
    
    # Finding the best performing model, we move on to storing its predictions on the train and test dataset as per our 'make_and_store_predictions' function
    make_and_store_predictions(best_model_path, q, sigma, params_net, hyper_param_net, train_loader, val_loader, device, SESSION)
    
    # Once the best models predictions are stored, we move ahead with storing the model as a 'best_model' for better reference
    source_dir = os.path.dirname(best_model_path)
    saving_best_model_path = os.path.join(source_dir, 'best_model.pth')
    shutil.copy(best_model_path, saving_best_model_path)

    return min_loss
