import torch
from torch import nn
from torch.autograd import Variable
# make_dot was moved to https://github.com/szagoruyko/pytorchviz
# try:
#     import hubermc, dataset_processing, logs_and_results, training
# except ImportError:
#     print("[INFO] Cloning the repository and importing convmc & dataset_preprocessing script...")
#     subprocess.run(["git", "clone", "https://github.com/Talha-Nehal-Undegrad-Study/ConvHuberMC-Net.git"])
#     subprocess.run(["mv", "ConvHuberMC-Net/python_scripts", "py_scripts"])
#     sys.path.append('py_scripts')
#     import hubermc, dataset_processing, logs_and_results, training

from python_scripts import architecture

def get_hyperparameter_grid(Model, TrainInstances, ValInstances, BatchSize, ValBatchSize, num_epochs, learning_rate):
  hyper_param = {}

  hyper_param['Model'] = Model
  hyper_param['TrainInstances'] = TrainInstances
  hyper_param['ValInstances'] = ValInstances
  hyper_param['BatchSize'] = BatchSize
  hyper_param['ValBatchSize'] = ValBatchSize
  hyper_param['Epochs'] = num_epochs
  hyper_param['Lr'] = learning_rate

  return hyper_param

# Defining a fucntion for loading/instantiating the model based on some boolean values and updates log
def get_model(params_net, hyper_param_net, log, path_whole = None, path_dict = None, loadmodel = False, load_frm = None):
    
    # Construct Model
    print('Configuring Network...\n')
    log.write('Configuring network...\n')

    if not loadmodel:
        print('Instantiating Model...\n')
        log.write('Instantiating Model...\n')
        if hyper_param_net['Model'] == 'HuberMC-Net':
            net = architecture.UnfoldedNet_Huber(params_net)
            print('Model Instantiated...\n')
            log.write('Model Instantiated...\n')

    else:
        print('Loading Model...\n')
        log.write('Loading Model...\n')
        if hyper_param_net['Model'] == 'HuberMC-Net':
            if load_frm == 'state_dict':
                torch.autograd.set_detect_anomaly(True)
                net = architecture.UnfoldedNet_Huber(params_net)
                state_dict = torch.load(path_dict, map_location = 'cpu')
                net.load_state_dict(state_dict)
                print('Model loaded from state dict...\n')
                log.write('Model loaded from state dict...\n')
            elif load_frm == 'whole_model':
                net = torch.load(path_whole)
                print('Whole model loaded...\n')
                log.write('Whole model loaded...\n')
            net.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)

    return net

# Training functions and Plotting Functions

def train_step(model, dataloader, loss_fn, optimizer, CalInGPU, TrainInstances, batch, inference = False):
  # Put model in train mode
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.train()

  # Initalize loss for lowrank matrices which will be calculated per batch for each epoch
  loss_mean = 0
  loss_lowrank_mean = 0
  # loss_fn.requires_grad = True
  
  for _, (D, L) in enumerate(dataloader):
    # set the gradients to zero at the beginning of each epoch
    optimizer.zero_grad()
    with torch.autograd.set_detect_anomaly(True):
        for mat in range(batch):
            
            inputs = D[mat].to(device)
            targets_L = L[mat].to(device)
            outputs_L = model(inputs)
            loss = (loss_fn(outputs_L, targets_L))/torch.square(torch.norm(targets_L, p = 'fro'))
            # loss = torch.square(torch.norm(targets_L, p = 'fro')) / loss_fn(outputs_L, targets_L)
            # loss = (loss_fn(outputs_L, targets_L)) / (targets_L.shape[0] * targets_L.shape[1])
            # loss = ((loss_val * loss_fn(outputs_L, targets_L)) / (targets_L.shape[0] * targets_L.shape[1])).item()
            if not inference:
              loss.backward()
              # print(loss)
            #   # print(loss.grad)

            # loss_mean += ((loss * loss_fn(outputs_L, targets_L)) / (targets_L.shape[0] * targets_L.shape[1])).item()
            loss_mean += loss.item()
    if not inference:
        # loss_mean.backward()
        optimizer.step()
  loss_mean = loss_mean/TrainInstances
  # loss_lowrank_mean = loss_lowrank_mean/TrainInstances

  # return loss_mean, loss_lowrank_mean
  return loss_mean
def test_step(model, dataloader, loss_fn, CalInGPU, ValInstances, batch):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  model.eval()
  loss_val_mean = 0

  # Validation
  with torch.no_grad():
    for _, (D, L) in enumerate(dataloader):
      for mat in range(batch):
        inputs = D[mat].to(device)   # "mat"th picture
        targets_L = L[mat].to(device)

        # for i in range(targets_L.shape[0]):
        #   for j in range(targets_L.shape[1]):
        #     output_ij = model(inputs, i, j)
        #     # loss = loss_fn(torch.tensor(output_ij.item()), torch.tensor(targets_L[i][j]))/torch.square(torch.norm(targets_L, p = 'fro'))
        #     loss_val = loss_fn(output_ij, targets_L[i][j])/torch.square(torch.norm(targets_L, p = 'fro'))
        #     loss_val_mean += loss_val.item()
        outputs_L = model(inputs)
        # loss_val = (loss_fn(outputs_L, targets_L))/torch.square(torch.norm(targets_L, p = 'fro'))
        # loss_val = torch.square(torch.norm(targets_L, p = 'fro')) / loss_fn(outputs_L, targets_L)
        loss = nn.MSELoss(reduction = 'sum')
        loss_val = (loss(outputs_L, targets_L)) / (targets_L.numel())
        # loss_val_mean += ((loss_val * loss_fn(outputs_L, targets_L)) / (targets_L.shape[0] * targets_L.shape[1])).item()
        loss_val_mean += loss_val.item()

        # Alternative IDea:
        # loss = torch.sum(loss_fn()

  loss_val_mean = loss_val_mean/ValInstances
  # loss_val_lowrank_mean = loss_val_lowrank_mean/ValInstances

  return loss_val_mean