from rail_training import Network, Network_CE, Network_mask1, Network_mask2
from rail_data_preprocess_original import get_preprocessed_data
import torch
import numpy as np
import time
import datetime
from rail_fun import cost_per_step, build_stacked_state, build_delta_vector
from rail_rl_env import RailNet, d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot, mdl_feasible, action_dict, gurobi_minlp, gurobi_lp_presolve
import sys

x = datetime.datetime.now()
date_str = '%.2d%.2d_%.2d%.2d' %(x.month, x.day, x.hour, x.minute)
print('date_time: ' + date_str)

testing = True

start_time = time.time()

N=40
# opt = 'minlp_ol'
opt = sys.argv[1]
threshold_counts = 25
output_get_preprocessed_data = get_preprocessed_data(opt, threshold_counts, N)

mipgap= 1e-3
log= 0
device='cpu'

if testing == True:
    timeout=5*60
    timelimit=30
    n_threads=8
    torch.set_num_threads(n_threads)
else:
    timeout = 24*60*60 - 30*60
    timelimit= 240
    n_threads = 1
    torch.set_num_threads(n_threads)

N = output_get_preprocessed_data[0]
N_control = output_get_preprocessed_data[1]
# stacked_states_train=output_get_preprocessed_data[2]
# stacked_states_val=output_get_preprocessed_data[3]
# stacked_actions_reduced_train=output_get_preprocessed_data[4]
stacked_actions_reduced_val=output_get_preprocessed_data[5]
list_masks=output_get_preprocessed_data[6]
# stacked_states_train_tensor=output_get_preprocessed_data[7]
stacked_states_val_tensor=output_get_preprocessed_data[8]
state_min_reduced=output_get_preprocessed_data[9]
state_max_reduced=output_get_preprocessed_data[10]
input_size=output_get_preprocessed_data[11]
total_action_set=output_get_preprocessed_data[12]

num_actions = total_action_set.shape[0]

def build_network(network_type, input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout, list_masks=list_masks, device=device):
    if network_type == 'Network':
        network = Network(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout)
    elif network_type == 'Network_CE':
        network = Network_CE(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout)
    elif network_type == 'Network_mask1':
        network = Network_mask1(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout, list_masks, device)
    elif network_type == 'Network_mask2':
        network = Network_mask2(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout, list_masks, device)
    else:
        raise KeyError('Network type has not been found.')
    
    return network

list_idx_networks = list(range(48))
del list_idx_networks[12:24] # exclude training with cross-entropy loss

list_networks_minlp_ol = []
h0 = []
c0 = []
for i in list_idx_networks:
    network_info = np.load('training_data//' + opt + '_N%d_%.3d_' %(N,i) + 'info.npy', allow_pickle=True).item()
    network_type = network_info['network_type']
    input_size = network_info['input_size']
    hidden_size = network_info['hidden_size']
    num_layers = network_info['num_layers']
    lr = network_info['lr']
    # num_actions = network_info['num_actions']
    # batch_size = network_info['batch_size']
    batch_size = 1
    dropout = network_info['dropout']
    # list_masks = network_info['list_masks']
    # device = network_info['device']
    
    network = build_network(network_type, input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout)
    network.load_state_dict(torch.load('training_data//' + opt + '_N%d_%.3d_' %(N,i) + 'weight'))
    
    h0.append(torch.zeros(num_layers, 1, hidden_size))
    c0.append(torch.zeros(num_layers, 1, hidden_size))
    
    list_networks_minlp_ol.append(network)
    
num_networks_minlp_ol = len(list_idx_networks)
    
list_networks_tscript_minlp_ol = []
for i in range(num_networks_minlp_ol):
    list_networks_tscript_minlp_ol.append(torch.jit.optimize_for_inference(torch.jit.script(list_networks_minlp_ol[i].eval())))
    
action_dict_reduced = {}
for i in total_action_set:
    action_dict_reduced[str(i)] = action_dict[str(i)]

Env = RailNet(N)

list_minlp = []

list_learning_minlp = []
for j in range(num_networks_minlp_ol):
    list_learning_minlp.append([])
    
k=0

with torch.no_grad():
    while time.time() < start_time + timeout :
        elapsed_time = (time.time()-start_time)/3600
        print('\nk=%d \t elapsed_time(hrs): %.3f' %(k,elapsed_time))
        Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)
        state_learning = build_stacked_state(Env.state_n, Env.state_rho, Env.state_depot, Env.state_l, input_size, N_control, state_min_reduced, state_max_reduced)
                        
        # minlp (warm start, early term)
        early_term = 1
        warm_start = 1
        a_values, d_values, r_values, l_values, y_values, delta_minlp, n_values, n_after_values,  mdl = gurobi_minlp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit, early_term, warm_start, n_threads)
        cost_minlp = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values,  N)
        list_minlp.append([mdl._Runtime, cost_minlp, 0, mdl.status])
        print('minlp_net \t runtime: %.2f \t cost: %.2f \t opt_gap: 0%% \t status: %d' %(mdl._Runtime, cost_minlp, mdl.status))
        
        for i in range(num_networks_minlp_ol):
                     
            output_net = list_networks_minlp_ol[i](state_learning, h0[i], c0[i])  
            # output_net = list_networks_tscript_minlp_ol[i](state_learning, h0[i], c0[i])
            action_idx = total_action_set[torch.max(output_net, dim=2)[1].squeeze().numpy()]
            delta_SL = build_delta_vector(action_idx, N_control, action_dict_reduced)
        
            a_values, d_values, r_values, l_values, y_values, n_values, n_after_values,  mdl = gurobi_lp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, delta_SL, mipgap, log, timelimit, n_threads)
            
            if mdl_feasible(mdl)==True:
                cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values,  N)
                perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
                list_learning_minlp[i].append([mdl.Runtime, cost, perc_minlp_cost, mdl.Status, list_idx_networks[i], True])
                print('learning+LP \t runtime: %.2f \t cost: %.2f \t opt_gap: %.2f \t status: %d \t idx: %d' %(mdl.Runtime, cost, perc_minlp_cost, mdl.Status, list_idx_networks[i]))
            else:
                list_learning_minlp[i].append([mdl.Runtime, False, False, mdl.Status, list_idx_networks[i], False])
                print('solution not feasible. idx: %d' %(list_idx_networks[i]))
        k+=1
                
array_minlp = np.array(list_minlp)
array_learning_minlp = np.array(list_learning_minlp)

dict_arrays = {
    'array_minlp': array_minlp,
    'array_learning_minlp': array_learning_minlp,
    'info': 'Each array has a list containing [mdl.Runtime, cost_minlp, perc_minlp_cost, mdl.status, idx_network, Feasible(bool)]'
}

x = datetime.datetime.now()
date_str = '%.2d%.2d_%.2d%.2d' %(x.month, x.day, x.hour, x.minute)
print('date_time: ' + date_str)

if testing == True:
    np.save('tests//tests_networks_ol' + opt + '_N%.2d.npy' %(N), dict_arrays)
else:
    np.save('/scratch/cfoliveiradasi/railway_largescale/tests/' + 'tests_networks_ol' + opt + '_N%.2d.npy' %(N), dict_arrays)

print('total time: %f' %(time.time()-start_time))
print('test completed!')