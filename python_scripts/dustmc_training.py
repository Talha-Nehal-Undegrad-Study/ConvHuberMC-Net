import torch
from torch import nn
from torch.autograd import Variable

from python_scripts import dustmc_unrolled
from python_scripts import utils

def get_hyperparameter_grid(Model, TrainInstances, ValInstances, BatchSize, ValBatchSize, num_epochs, learning_rate,
                            K, mu, sigma, m, n, d, T):
        
    hyper_param = {}
    hyper_param['Model'] = Model
    hyper_param['TrainInstances'] = TrainInstances
    hyper_param['ValInstances'] = ValInstances
    hyper_param['BatchSize'] = BatchSize
    hyper_param['ValBatchSize'] = ValBatchSize
    hyper_param['Epochs'] = num_epochs
    hyper_param['Lr'] = learning_rate
    hyper_param['K'] = K
    hyper_param['mu'] = mu
    hyper_param['sigma'] = sigma
    hyper_param['m'] = m
    hyper_param['n'] = n
    hyper_param['d'] = d
    hyper_param['T'] = T

    return hyper_param

# Defining a fucntion for loading/instantiating the model based on some boolean values and updates log
def get_model(params_net, hyper_param_net, log, path_whole = None, path_dict = None, loadmodel = False, load_frm = None):
    
    # Construct Model
    print('Configuring Network...')
    log.write('Configuring network...\n')

    if not loadmodel:
        print('Instantiating Model...')
        log.write('Instantiating Model...\n')
        if hyper_param_net['Model'] == 'DUSTMC-Net':
            net = dustmc_unrolled.DustNet(hyper_param_net)
            print('Model Instantiated...')
            log.write('Model Instantiated...\n')

    else:
        print('Loading Model...')
        log.write('Loading Model...\n')
        if hyper_param_net['Model'] == 'DUSTMC-Net':
            if load_frm == 'state_dict':
                torch.autograd.set_detect_anomaly(True)
                net = dustmc_unrolled.DustNet(params_net)
                state_dict = torch.load(path_dict, map_location = 'cpu')
                net.load_state_dict(state_dict)
                print('Model loaded from state dict...')
                log.write('Model loaded from state dict...\n')
            elif load_frm == 'whole_model':
                net = torch.load(path_whole)
                print('Whole model loaded...')
                log.write('Whole model loaded...\n')
            net.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    return net

# Training functions and Plotting Functions

def train_step(model, dataloader, loss_fn, optimizer, TrainInstances, batch, inference = False):
    # Put model in train mode
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()

    # Initalize loss for lowrank matrices which will be calculated per batch for each epoch
    loss_mean = 0
  
    for _, (image, label) in enumerate(dataloader): # usually _, (D, L)
        # set the gradients to zero at the beginning of each epoch
        optimizer.zero_grad()
        image = image.squeeze()
        with torch.autograd.set_detect_anomaly(False):
            for mat in range(batch):
                inputs = image[mat].to(device)
                # targets_L = L[mat].to(device)
                outputs_D = model(inputs)
                loss = utils.columnwise_mse_loss(outputs_D, inputs)
                loss_mean += loss.item()
        if not inference:
            loss.backward()
            optimizer.step()
    loss_mean = loss_mean/TrainInstances

    return loss_mean

def test_step(model, dataloader, loss_fn, ValInstances, batch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    loss_val_mean = 0

    # Validation
    with torch.no_grad():
        for _, (image, label) in enumerate(dataloader):
            for mat in range(batch):
                inputs = D[mat].to(device)   # "mat"th picture
                # targets_L = L[mat].to(device)

                outputs_D = model(inputs)
                loss_val = utils.columnwise_mse_loss(outputs_D, inputs)
                loss_val_mean += loss_val.item()

    loss_val_mean = loss_val_mean/ValInstances

    return loss_val_mean