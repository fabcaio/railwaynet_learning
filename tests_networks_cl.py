from rail_training import Network, Network_CE, Network_mask1, Network_mask2, Network_mask3
from rail_data_preprocess_original import get_preprocessed_data
import torch
import numpy as np
import time
import datetime
from rail_fun import cost_per_step, build_stacked_state, build_delta_vector
from rail_rl_env import RailNet, d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot, mdl_feasible, action_dict, gurobi_minlp, gurobi_milp, gurobi_lp_presolve
import sys
import copy

x = datetime.datetime.now()
date_str = '%.2d%.2d_%.2d%.2d' %(x.month, x.day, x.hour, x.minute)
print('date_time: ' + date_str)

testing = False

start_time = time.time()

mipgap= 1e-3
log= 0
device='cpu'

if testing == True:
    opt = 'milp_cl'
    timeout=10*60
    timelimit=30
    n_threads=8
    torch.set_num_threads(n_threads)
else:
    opt = sys.argv[1]
    timeout = 24*60*60 - 90*60
    timelimit = 120
    n_threads = 1
    torch.set_num_threads(n_threads)

N=40
threshold_counts = 25
output_get_preprocessed_data = get_preprocessed_data(opt, threshold_counts, N)
batch_size = 1
device = "cpu"

# N = output_get_preprocessed_data[0]
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
    elif network_type == 'Network_mask3':
        network = Network_mask3(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout, list_masks, device)
    else:
        raise KeyError('Network type has not been found.')
    
    return network

list_idx_networks = list(range(48))
# del list_idx_networks[12:24] # exclude training with cross-entropy loss

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

j=0 # number of data points
k=0 # step size count
Env_learning = RailNet(N)
    
with torch.no_grad():
    
    while time.time() < start_time + timeout :
        elapsed_time = (time.time()-start_time)/3600
        print('\nnum_data_points = %d \t elapsed_time(hrs): %.3f' %(j,elapsed_time))
        
        k=0
        early_term = 1
        warm_start = 1
        Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)
        print('\nnew episode')
        # Env.step(0, d_pre, rho_whole, mipgap, log, timelimit, early_term, warm_start, n_threads, opt='minlp')[-1]
        
        while (Env.terminated or Env.truncated)==False and k < 30:
            
            Env_old = copy.deepcopy(Env)            
            state_learning = build_stacked_state(Env_old.state_n, Env_old.state_rho, Env_old.state_depot, Env_old.state_l, input_size, N_control, state_min_reduced, state_max_reduced)
          
            # minlp (warm start, early term)
            early_term = 1
            warm_start = 1
            a_values, d_values, r_values, l_values, y_values, delta_minlp, n_values, n_after_values,  mdl = gurobi_minlp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit, early_term, warm_start, n_threads)
            cost_minlp = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, 1)
            list_minlp.append([mdl._Runtime, cost_minlp, 0, mdl.status])
            print('minlp_net \t runtime: %.2f \t cost: %.2f \t opt_gap: 0%% \t status: %d' %(mdl._Runtime, cost_minlp, mdl.status))
            Env.step(delta_minlp, d_pre, rho_whole, mipgap, log, timelimit, early_term, warm_start, n_threads, opt='lp')
            
            # # minlp (warm start, early term)
            # info = Env.step(0, d_pre, rho_whole, mipgap, log, timelimit, early_term, warm_start, n_threads, opt='minlp')[-1]
            # mdl_minlp = info['mdl']
            # state_n = np.expand_dims(Env.state_n, axis=1)
            # state_n_after = np.expand_dims(Env.state_n_after, axis=1)
            # state_d = np.expand_dims(Env.state_d[:,0,:], axis=1)
            # state_l = np.expand_dims(Env.state_l[:,0,:], axis=1)
            # cost_minlp = cost_per_step(state_n, state_n_after, Env.d_pre_cut_old, state_d, state_l, 1)
            # list_minlp.append([mdl_minlp._Runtime, cost_minlp, 0, mdl_minlp.status])
            # print('minlp_net \t runtime: %.2f \t cost: %.2f \t opt_gap: 0%% \t status: %d' %(mdl_minlp._Runtime, cost_minlp, mdl_minlp.status))  
            
            for i in range(num_networks_minlp_ol):
            # for i in range(12):
                
                # Env_learning_current = copy.deepcopy(Env_previous)
                
                output_net = list_networks_minlp_ol[i](state_learning, h0[i], c0[i])  
                # output_net = list_networks_tscript_minlp_ol[i](state_learning, h0[i], c0[i])
                action_idx = total_action_set[torch.max(output_net, dim=2)[1].squeeze().numpy()]
                delta_SL = build_delta_vector(action_idx, N_control, action_dict_reduced)
            
                a_values, d_values, r_values, l_values, y_values, n_values, n_after_values,  mdl = gurobi_lp_presolve(N, Env_old.d_pre_cut, Env_old.state_rho, Env_old.state_a, Env_old.state_d, Env_old.state_r, Env_old.state_l, Env_old.state_y, Env_old.state_n, Env_old.state_depot, delta_SL, mipgap, log, timelimit, n_threads)        
                
                # info = Env_learning_current.step(delta_SL, d_pre, rho_whole, mipgap, log, timelimit, early_term, warm_start, n_threads, opt='lp')[-1]
                # mdl = info['mdl']
                
                if mdl_feasible(mdl)==True:
                    # state_n = np.expand_dims(Env_learning_current.state_n, axis=1)
                    # state_n_after = np.expand_dims(Env_learning_current.state_n_after, axis=1)
                    # state_d = np.expand_dims(Env_learning_current.state_d[:,0,:], axis=1)
                    # state_l = np.expand_dims(Env_learning_current.state_l[:,0,:], axis=1)
                    # cost = cost_per_step(state_n, state_n_after, Env_learning_current.d_pre_cut_old, state_d, state_l, 1)
                    cost = cost_per_step(n_values, n_after_values, Env_old.d_pre_cut_old, d_values, l_values, 1)
                    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
                    list_learning_minlp[i].append([mdl.Runtime, cost, perc_minlp_cost, mdl.Status, list_idx_networks[i], True])
                    print('learning+LP \t runtime: %.2f \t cost: %.2f \t opt_gap: %.2f \t status: %d \t idx: %d' %(mdl.Runtime, cost, perc_minlp_cost, mdl.Status, list_idx_networks[i]))
                else:
                    list_learning_minlp[i].append([mdl.Runtime, False, False, mdl.Status, list_idx_networks[i], False])
                    print('solution not feasible. idx: %d' %(list_idx_networks[i]))
                    
            k+=1
            j+=1
                
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
    np.save('tests//tests_networks_cl' + opt + '_N%.2d_test.npy' %(N), dict_arrays)
else:
    np.save('/scratch/cfoliveiradasi/railway_largescale/tests/' + 'tests_networks_cl_' + opt + '_N%.2d.npy' %(N), dict_arrays)

print('total time: %f' %(time.time()-start_time))
print('test completed!')