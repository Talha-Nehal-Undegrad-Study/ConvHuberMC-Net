import numpy as np
from python_scripts import lp1_reg, lp2_reg, generate_synthetic_data

np.random.seed(42)

def compare(lp_reg, m_estimation, lo_bcd, params_net, hyper_param_net, sampling_rate, db):
    M_train, M_Omega_train, M_test, M_Omega_test = generate_synthetic_data.generate(
        params_net['size1'], params_net['size2'], params_net['rank'], 
        hyper_param_net['TrainInstances'], hyper_param_net['ValInstances'], sampling_rate, db)
    
    if lp_reg:
        for sample_rate in sampling_rate:
            for db in db:
                
