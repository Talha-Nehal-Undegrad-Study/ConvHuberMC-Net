import datetime
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# ROOT = '/home/gcf/Desktop/Talha_Nehal Sproj/Tahir Sproj Stuff/SPROJ_ConvMC_Net/Sensor_Data'
# ROOT = 'C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/ConvHuberMC/HuberMC_Data'
# ROOT = 'C:/Users/HP/Git/ConvHuberMC-Net/HuberMC_Data'

# Small Helper Function to modularize training loop
# Short functions to get sampling rate and noise variables as strings. make_dir is a function which takes the appropriate noise and sampling rate directory and makes there a directory for either logs or
# saved models or plots (for both convmc and admm). get_modularized_record() takes the project name, hyperparameters, q and db and returns its file path either that be a log file or a title for a plot
def get_q_str(q):
    q_exp = f'{q * 100}%'
    return q_exp

def get_noise_str(db):
    noise_exp = f'{float(db)}'
    return noise_exp

def make_dir(q, db, new_entry, hyper_param_net, ROOT):
    new_dir = ROOT + '/Q ' + get_q_str(q) + '/DB ' + get_noise_str(db) + '/' + new_entry
    os.makedirs(new_dir, exist_ok = True)

    hubermc_dir = new_dir + '/HuberMC-Net'
    os.makedirs(hubermc_dir, exist_ok = True)

    return hubermc_dir

# Making a small function which makes directories for different tries for different models
def make_session(q, db, new_entry, hyper_param_net, ROOT, SESSION):
    # Get dir of activity/record to make sessions of
    dir = make_dir(q, db, new_entry, hyper_param_net, ROOT)

    # Now make a session corresponding to whichever try is going on currently
    session_dir = dir + '/' + SESSION
    os.makedirs(session_dir, exist_ok = True)

    return session_dir

# Making a small function to get date and time according to our time zone. Will be used to create the logs
def get_current_time():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts)

    # Add 5 hours to the datetime according to our time zone
    new_time = st + datetime.timedelta(hours = 5)

    formatted_new_time = new_time.strftime('%Y-%m-%d %H:%M:%S')

    return formatted_new_time

def make_predictions_dir(project_name, q, db, new_entry, params_net, hyper_param_net, ROOT, SESSION):
    # Get Directory with ADMM-Net/ConvMC-Net --> Session directories built
    dir = make_session(q, db, new_entry, hyper_param_net, ROOT, SESSION)

    # Make a children directory after Session --> ProjectName + parameters --> Train/Test which will contain the final predictions on our train and test data used to evaluate the model after training phase
    param_project_dir = (f'{dir}/{project_name} Layers_{params_net["layers"]}_TrainInstances_{hyper_param_net["TrainInstances"]}_Epochs_{hyper_param_net["Epochs"]}_lr_{hyper_param_net["Lr"]}')
    os.makedirs(param_project_dir, exist_ok = True)

    # Train/Test Dir dir
    train_dir = param_project_dir + '/train'
    test_dir = param_project_dir + '/test'

    os.makedirs(train_dir, exist_ok = True)
    os.makedirs(test_dir, exist_ok = True)

    # Return Train and Test Directories to store data in
    return train_dir, test_dir

def get_modularized_record(project_name, q, db, new_entry, hyper_param_net, params_net, ROOT, SESSION, current_epoch = None):
    # Get directory
    dir = make_session(q, db, new_entry, hyper_param_net, ROOT, SESSION)
    if current_epoch == None:
        log_path = (f'{dir}/{project_name} Layers_{params_net["layers"]}_TrainInstances_{hyper_param_net["TrainInstances"]}_Epochs_{hyper_param_net["Epochs"]}_lr_{hyper_param_net["Lr"]}.txt')
        return log_path
    elif new_entry == 'Plots':
        model_path = (f'{dir}/{project_name} Layers_{params_net["layers"]}_TrainInstances_{hyper_param_net["TrainInstances"]}_Epochs_[{current_epoch}_out_of_{hyper_param_net["Epochs"]}]_lr_{hyper_param_net["Lr"]}.png')
        return model_path
    else:
        model_path = (f'{dir}/{project_name} Layers_{params_net["layers"]}_TrainInstances_{hyper_param_net["TrainInstances"]}_Epochs_[{current_epoch}_out_of_{hyper_param_net["Epochs"]}]_lr_{hyper_param_net["Lr"]}.pth')
        return model_path

def plot_and_save_mse_vs_epoch(epochs_vec, lossmean_vec, lossmean_val_vec, dir):
    fig = plt.figure(figsize = (8, 6), dpi = 100)
    plt.plot(epochs_vec, lossmean_vec, '-*', label = 'loss')
    plt.plot(epochs_vec, lossmean_val_vec, '-*', label = 'loss_val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(ymin = 0)
    plt.title("MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(dir)