from rail_fun import cost_per_step, build_stacked_state, build_delta_vector
import numpy as np
from rail_rl_env import RailNet, d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot, mdl_feasible, action_dict, gurobi_nlp_presolve, gurobi_lp_presolve
import time
from rail_data_preprocess_original import get_preprocessed_data
from rail_training import Network, Network_CE, Network_mask1, Network_mask2
import torch
import sys
import datetime

testing = True

x = datetime.datetime.now()
date_str = '%.2d%.2d_%.2d%.2d' %(x.month, x.day, x.hour, x.minute)
print('date_time: ' + date_str)

start_loop = time.time()

# add by Xiaoyu
# 
match_nlp = np.zeros(3, dtype=int)
match_lp = np.zeros(3, dtype=int)
match_constant = np.zeros(3, dtype=int)
match_constant[0] = 21 # 4 x 21: time for the train to return the first station (from the original timetable)
match_constant[1] = 27
match_constant[2] = 23

mipgap = 1e-3
log = 0
early_term = 1

if testing:    
    opt_data = 'milp_ol'
    N=40
    job_idx = 999
    n_threads = 8  
    timeout = 3*60 #minutes
    timelimit = 10
    timelimit_reference = 20 # sets the time for the longest minlp (with warm-start)
else:
    opt_data=sys.argv[1]
    N=int(sys.argv[2])
    job_idx = int(sys.argv[3])
    timelimit_job = int(sys.argv[4])
    n_threads = int(sys.argv[5])
    timeout = (timelimit_job-2.5)*60*60
    timelimit = int(sys.argv[6])
    timelimit_reference = 600 # sets the time for the longest minlp (with warm-start)

batch_size=1
device = "cpu"

Env = RailNet(N)

print('N: %d \t mipgap: %.4f \t timelimit: %.2f \t n_threads: %d \t early_term: %d' %(N, mipgap, timelimit, n_threads, early_term))

############################################################

match N:
    case 20:
        threshold_counts = 50
    case 40:
        threshold_counts = 25

print('opt_data: ' + opt_data + '\t threshold counts: %d' %threshold_counts)
print('timeout: %.2f,\t' %timeout, 'mipgap %.4f' %mipgap)

output_get_preprocessed_data = get_preprocessed_data(opt_data, threshold_counts, N)

N = output_get_preprocessed_data[0]
N_control = output_get_preprocessed_data[1]
stacked_actions_reduced_val=output_get_preprocessed_data[5]
list_masks=output_get_preprocessed_data[6]
stacked_states_val_tensor=output_get_preprocessed_data[8]
state_min_reduced=output_get_preprocessed_data[9]
state_max_reduced=output_get_preprocessed_data[10]
input_size=output_get_preprocessed_data[11]
total_action_set=output_get_preprocessed_data[12]

action_dict_reduced = {}
for i in total_action_set:
    action_dict_reduced[str(i)] = action_dict[str(i)]

num_actions = total_action_set.shape[0]
seq_len=N_control

#builds the mask
mask = torch.tensor(np.array(list_masks, dtype=np.int32)).unsqueeze(axis=0).to(device)
masks_tensor = mask
for i in range(1,batch_size):
    masks_tensor = torch.concatenate((masks_tensor, mask), dim=0)
mask = masks_tensor

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

match N:
    case 20:
        hidden_size=256
        num_layers=1
        lr=1e-4
        model1_str = 'milp_ol_hs256_N20_tc50_0808_1638_weight'
        model2_str = 'milp_ol_hs256_N20_tc50_s04_0908_1310_weight'
        model3_str = 'milp_ol_hs256_N20_tc50_s07_0908_1337_weight'
        model4_str = 'milp_ol_hs256_N20_tc50_s11_0908_1435_weight'
        model5_str = 'milp_ol_hs256_N20_tc50_s42_0908_1407_weight'
        model_str_list = [model1_str, model2_str, model3_str, model4_str, model5_str]
        model_list = []
        h0 = []
        c0 = []
        for i in range(len(model_str_list)):
            model_list.append(Network(input_size, hidden_size, num_layers, lr, num_actions, batch_size))
            tmp_str = model_str_list[i]
            model_list[i].load_state_dict(torch.load('training_data//' + tmp_str, map_location=torch.device('cpu')))
            h0.append(torch.zeros(num_layers, 1, hidden_size))
            c0.append(torch.zeros(num_layers, 1, hidden_size))
    case 40:
        if opt_data=='minlp_ol':
            idx_best_networks = [9, 31, 7, 33, 43, 8, 24, 5, 42, 28, 32, 4, 29, 2, 26]
        elif opt_data=='milp_ol':
            idx_best_networks = [31, 7, 10, 11, 26, 3, 2, 6, 33, 27, 43, 30, 8, 9, 42]
        model_list = []
        h0 = []
        c0 = []
        for i in idx_best_networks:
            network_info = np.load('training_data//' + opt_data + '_N%d_%.3d_' %(N,i) + 'info.npy', allow_pickle=True).item()
            network_type = network_info['network_type']
            input_size = network_info['input_size']
            hidden_size = network_info['hidden_size']
            num_layers = network_info['num_layers']
            lr = network_info['lr']
            dropout = network_info['dropout']
            
            network = build_network(network_type, input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout)
            network.load_state_dict(torch.load('training_data//' + opt_data + '_N%d_%.3d_' %(N,i) + 'weight', weights_only=True))
            
            h0.append(torch.zeros(num_layers, 1, hidden_size))
            c0.append(torch.zeros(num_layers, 1, hidden_size))
            
            model_list.append(network)
            
        num_networks_minlp_ol = len(idx_best_networks)

# torch script is not working for Network_mask1
# model_tscript_list = []
# for i in range(len(model_list)):
#     model_tscript_list.append(torch.jit.optimize_for_inference(torch.jit.script(model_list[i].eval())))

"""
net: no early termination
ws:  warm start
10m: timelimit_gurobi = 10minutes
"""

list_minlp_net_ws_10m = []
list_minlp_net = []
list_minlp_net_ws = []
list_minlp = []
list_minlp_ws = []
list_milp = []
list_learning_nlp = []
list_learning_lp = []

############################################################

Env_minlp_net_ws_10m = RailNet(N)
Env_minlp_net_ws_10m.name = 'minlp_net_ws_10m'
Env_minlp_net = RailNet(N)
Env_minlp_net.name = 'minlp_net'
Env_minlp_net_ws = RailNet(N)
Env_minlp_net_ws.name = 'minlp_net_ws'
Env_minlp = RailNet(N)
Env_minlp.name = 'minlp'
Env_minlp_ws = RailNet(N)
Env_minlp_ws.name = 'minlp_ws'
Env_milp = RailNet(N)
Env_milp.name = 'milp'
Env_learning_nlp = RailNet(N)
Env_learning_nlp.name = 'learning_nlp'
Env_learning_lp = RailNet(N)
Env_learning_lp.name = 'learning_lp'

##########################################################

# 1 train  -> 00
# 2 trains -> 10
# 3 trains -> 01
# 4 trains -> 11

delta_heuristic1 = np.zeros((12,))
delta_heuristic1[[2,6,10]] = 0     # 1st bit train composition
delta_heuristic1[[3,7,11]] = 0     # 2nd bit train composition
delta_heuristic1[[0,4,8]] = 1      # xi_1 indices
delta_heuristic1[[1,5,9]] = 0      # xi_2 indices

stacked_delta_heuristic1 = []
for i in range(0,N_control):
    stacked_delta_heuristic1.append(delta_heuristic1)
stacked_delta_heuristic1 = np.array(stacked_delta_heuristic1)

delta_heuristic2 = np.zeros((12,))
delta_heuristic2[[2,6,10]] = 1     # 1st bit train composition
delta_heuristic2[[3,7,11]] = 0     # 2nd bit train composition
delta_heuristic2[[0,4,8]] = 1      # xi_1 indices (1 is to keep the original order)
delta_heuristic2[[1,5,9]] = 0      # xi_2 indices

stacked_delta_heuristic2 = []
for i in range(0,N_control):
    stacked_delta_heuristic2.append(delta_heuristic2)
stacked_delta_heuristic2 = np.array(stacked_delta_heuristic2)

# add by Xiaoyu
current_delta = stacked_delta_heuristic2

###########################################################

def store_data(Env, list_data, mdl, warm_start, cost_reference, total_inf_time=0, number_model=0):    
    
    """
    store the closed-loop data in a list for the optimization-based or learning-based approaches
    
    arguments:
        Env: environment of the approach
        list_data: list of the approach
        mdl: output of the optimization (provided by Env.step())
        warm_start: (bool)
        cost_reference: step cost for the reference approach
        reference: (bool) whether the approach is the reference or not
        
    output:

        cost: one-step cost for the approach
    
    """
    
    state_n = np.expand_dims(Env.state_n, axis=1)
    state_n_after = np.expand_dims(Env.state_n_after, axis=1)
    state_d = np.expand_dims(Env.state_d[:,0,:], axis=1)
    state_l = np.expand_dims(Env.state_l[:,0,:], axis=1)
    cost = cost_per_step(state_n, state_n_after, Env.d_pre_cut_old, state_d, state_l, 1)
    # state_a = np.expand_dims(Env.state_a[:, 0, :], axis=1)
    
    if warm_start==1:
        runtime = mdl._Runtime
    else:
        runtime = mdl.Runtime
        
    if Env.name == 'minlp_net_ws_10m': # this is the reference
        perc_minlp_cost = 0
    else:
        perc_minlp_cost = (cost-cost_reference)/cost_reference*100
        
    if Env.name == 'learning_nlp' or Env.name == 'learning_lp':
        list_data.append([runtime, cost, perc_minlp_cost, mdl.status])
    else:
        list_data.append([runtime, cost, perc_minlp_cost, mdl.status, total_inf_time, number_model])        
    
    print(Env.name + '\t\t %.2f \t\t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(time.time()-start_time, runtime, cost, perc_minlp_cost, mdl.status))
        
    return cost

def get_gap(cost, cost_reference):
    
    """
    This function outputs the relative percentage difference between two values.    
    """
    
    opt_gap = (cost-cost_reference)/cost_reference*100
    return opt_gap

cntr_infeas_total_learning_nlp = 0
cntr_infeas_total_learning_lp = 0
cntr_infeas_episode_learning_nlp = 0
cntr_infeas_episode_learning_lp = 0
cntr_infeas_total_heuristic_nlp = 0
cntr_infeas_total_heuristic_lp = 0

d_stack = np.zeros((5,30,3,40,38)) # method number, n_episode_steps, shape of d
a_stack = np.zeros((5,30,3,40,38))
l_stack = np.zeros((5,30,3,40,38))
n_stack = np.zeros((5,30,3,40,38))
n_after_stack = np.zeros((5,30,3,40,38))
n_before_stack = np.zeros((5,30,3,40,38))
idx_cntr_stack = np.zeros((5,30,1))
step_cost_stack = np.zeros((5,30,1))
opt_gap_stack = np.zeros((5,30,1))

d_list_stack = []
a_list_stack = []
l_list_stack = []
n_list_stack = []
n_after_list_stack = []
n_before_list_stack = []
idx_cntr_list_stack = []
step_cost_list_stack = []
opt_gap_list_stack = []
runtime_list_stack = []
env_status_list_stack = []

with torch.no_grad():
    
    k = 1 # counts the number of data points

    while time.time() < start_loop + timeout :

        j = 0 # counts the time step in the episode
            
        Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)
        idx_cntr_init = Env.idx_cntr
        idx_cntr = Env.idx_cntr
        
        Env_minlp_net_ws_10m.copyEnv(Env)
        Env_minlp_net.copyEnv(Env)
        Env_minlp_net_ws.copyEnv(Env)
        Env_minlp.copyEnv(Env)
        Env_minlp_ws.copyEnv(Env)
        Env_milp.copyEnv(Env)
        Env_learning_nlp.copyEnv(Env)
        Env_learning_lp.copyEnv(Env)
        
        cntr_infeas_episode_learning_nlp = 0
        cntr_infeas_episode_learning_lp = 0
        
        cost_episode_minlp_net_ws_10m = 0
        cost_episode_minlp_net = 0
        cost_episode_minlp_net_ws = 0
        cost_episode_minlp = 0
        cost_episode_minlp_ws = 0
        cost_episode_milp = 0
        cost_episode_learning_nlp = 0
        cost_episode_learning_lp = 0
        
        runtime_episode_minlp_net_ws_10m = 0
        runtime_episode_minlp_net = 0
        runtime_episode_minlp_net_ws = 0
        runtime_episode_minlp = 0
        runtime_episode_minlp_ws = 0
        runtime_episode_milp = 0
        runtime_episode_learning_nlp = 0
        runtime_episode_learning_lp = 0
        
        d_list_stack.append(np.zeros((5,3,30,38)))
        a_list_stack.append(np.zeros((5,3,30,38)))
        l_list_stack.append(np.zeros((5,3,30,38)))
        n_list_stack.append(np.zeros((5,3,30,38)))
        n_after_list_stack.append(np.zeros((5,3,30,38)))
        n_before_list_stack.append(np.zeros((5,3,30,38)))
        idx_cntr_list_stack.append(np.zeros((3,1)))
        step_cost_list_stack.append(np.zeros((5,30,1)))
        opt_gap_list_stack.append(np.zeros((5,30,1)))
        runtime_list_stack.append(np.zeros((5,30,1)))
        env_status_list_stack.append(np.zeros((4,1)))
        
        flag_end_episode = False
        
        while not (Env_learning_nlp.terminated or Env_learning_nlp.truncated) and not (Env_learning_lp.terminated or Env_learning_lp.truncated) and j < 30:
            
            print('\nnumber_points=%d, time_step_episode=%d' %(k,j))
            
            # minlp (without early termination) - minlp_net_ws_10m
            # this is the reference
            start_time = time.time()
            early_term = 0
            warm_start = 1
            info = Env_minlp_net_ws_10m.step(0, d_pre, rho_whole, mipgap, log, timelimit_reference, early_term, warm_start, n_threads, opt='minlp')[-1]
            mdl_reference = info['mdl']
            
            # minlp (with early termination) - minlp_ws
            start_time = time.time()
            early_term = 1
            warm_start = 1
            info = Env_minlp_ws.step(0, d_pre, rho_whole, mipgap, log, timelimit, early_term, warm_start, n_threads, opt='minlp')[-1]
            mdl_minlp_ws = info['mdl']
            
            # milp
            start_time = time.time()
            early_term = 0
            warm_start = 0
            info = Env_milp.step(0, d_pre, rho_whole, mipgap, log, timelimit, early_term, warm_start, n_threads, opt='milp')[-1]
            mdl_milp = info['mdl']
            
            # learning + nlp
            state_learning = build_stacked_state(Env_learning_nlp.state_n, Env_learning_nlp.state_rho, Env_learning_nlp.state_depot, Env_learning_nlp.state_l, input_size, N_control, state_min_reduced, state_max_reduced)
            start_time=time.time()
            total_inf_time=0
            early_term = 1
            warm_start = 0
            opt_time = timelimit
            for i in range(len(model_list)):
                start_inf_time = time.time()
                # output_net = model_tscript_list[i](state_learning, h0[i], c0[i])*mask
                output_net = model_list[i](state_learning, h0[i], c0[i])
                action_idx = total_action_set[torch.max(output_net, dim=2)[1].squeeze().numpy()]
                delta_SL = build_delta_vector(action_idx, N_control, action_dict_reduced)
                inf_time = time.time()-start_inf_time
            
                total_inf_time += inf_time
                opt_time = opt_time - inf_time
            
                a_values, d_values, r_values, l_values, y_values, n_values, n_after_values,  mdl_learning_nlp = gurobi_nlp_presolve(N, Env_learning_nlp.d_pre_cut, Env_learning_nlp.state_rho, Env_learning_nlp.state_a, Env_learning_nlp.state_d, Env_learning_nlp.state_r, Env_learning_nlp.state_l, Env_learning_nlp.state_y, Env_learning_nlp.state_n, Env_learning_nlp.state_depot, delta_SL, mipgap, log, opt_time, early_term, warm_start, n_threads)
                       
                number_model = i+1
            
                if mdl_feasible(mdl_learning_nlp):
                    break
            
            if mdl_feasible(mdl_learning_nlp):
                info = Env_learning_nlp.step(delta_SL, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt='nlp')[-1]
                current_delta = delta_SL            
            
            else:
                print('learning + nlp: not feasible, model_number: %d' %(number_model))
                cntr_infeas_total_learning_nlp +=1
                cntr_infeas_episode_learning_nlp +=1

                # add by Xiaoyu_20241120
                for line in range(3):
                    if current_delta[N_control - 2, 1 + 4 * line] == 0:
                        match_nlp[line] = match_constant[line]
                    else:
                        match_nlp[line] = match_constant[line] + 1
            
                # add by Xiaoyu
                delta_recursive = np.zeros([N_control, 12], dtype=int)
                delta_recursive[:-1, :] = current_delta[1:, :]
                for line in range(3):
                    delta_recursive[N_control - 1, 0 + 4 * line] = current_delta[N_control - 2, 0 + 4 * line]
                    delta_recursive[N_control - 1, 1 + 4 * line] = current_delta[N_control - 2, 1 + 4 * line]
                    delta_recursive[N_control - 1, 2 + 4 * line] = current_delta[N_control - 1 - match_nlp[line], 2 + 4 * line]
                    delta_recursive[N_control - 1, 3 + 4 * line] = current_delta[N_control - 1 - match_nlp[line], 3 + 4 * line]
                a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl_learning_nlp = gurobi_nlp_presolve(N, Env_learning_nlp.d_pre_cut, Env_learning_nlp.state_rho, Env_learning_nlp.state_a,Env_learning_nlp.state_d, Env_learning_nlp.state_r, Env_learning_nlp.state_l,Env_learning_nlp.state_y, Env_learning_nlp.state_n, Env_learning_nlp.state_depot, delta_recursive, mipgap,log, opt_time, early_term, warm_start, n_threads)

                # add by Xiaoyu_20241120
                if mdl_feasible(mdl_learning_nlp):
                    info = Env_learning_nlp.step(delta_recursive, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt='nlp')[-1]
                    current_delta = delta_recursive
                else:
                    print('recursive solution (nlp) is infeasible')
                    a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl_learning_nlp = gurobi_nlp_presolve(N, Env_learning_nlp.d_pre_cut, Env_learning_nlp.state_rho, Env_learning_nlp.state_a,Env_learning_nlp.state_d, Env_learning_nlp.state_r, Env_learning_nlp.state_l,Env_learning_nlp.state_y, Env_learning_nlp.state_n, Env_learning_nlp.state_depot,stacked_delta_heuristic1, mipgap, log, opt_time, early_term, warm_start, n_threads)

                    if mdl_feasible(mdl_learning_nlp):
                        # cost_learning_nlp = store_data(Env_learning_nlp, list_learning_nlp, mdl, warm_start, cost_reference, total_inf_time, number_model)
                        info = Env_learning_nlp.step(stacked_delta_heuristic1, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt='nlp')[-1]
                        current_delta = stacked_delta_heuristic1
                    else:
                        cntr_infeas_total_heuristic_nlp += 1
                        flag_end_episode = True
                        print('heuristic solution is infeasible')
            
                # if mdl_feasible(mdl_learning_nlp)==False:
                #     cntr_infeas_total_heuristic_nlp +=1
                #     flag_end_episode = True
                #     print('heuristic solution (nlp) is infeasible')
                    
            # learning + lp
            state_learning = build_stacked_state(Env_learning_lp.state_n, Env_learning_lp.state_rho, Env_learning_lp.state_depot, Env_learning_lp.state_l, input_size, N_control, state_min_reduced, state_max_reduced)
            start_time=time.time()
            total_inf_time=0
            early_term = 1
            warm_start = 0
            opt_time = timelimit
            for i in range(len(model_list)):
                start_inf_time = time.time()
                # output_net = model_tscript_list[i](state_learning, h0[i], c0[i])*mask
                output_net = model_list[i](state_learning, h0[i], c0[i])
                action_idx = total_action_set[torch.max(output_net, dim=2)[1].squeeze().numpy()]
                delta_SL = build_delta_vector(action_idx, N_control, action_dict_reduced)
                inf_time = time.time()-start_inf_time
            
                total_inf_time += inf_time
                opt_time = opt_time - inf_time
            
                a_values, d_values, r_values, l_values, y_values, n_values, n_after_values,  mdl_learning_lp = gurobi_lp_presolve(N, Env_learning_lp.d_pre_cut, Env_learning_lp.state_rho, Env_learning_lp.state_a, Env_learning_lp.state_d, Env_learning_lp.state_r, Env_learning_lp.state_l, Env_learning_lp.state_y, Env_learning_lp.state_n, Env_learning_lp.state_depot, delta_SL, mipgap, log, opt_time, n_threads)
                       
                number_model = i+1
            
                if mdl_feasible(mdl_learning_lp):
                    break
            
            if mdl_feasible(mdl_learning_lp):
                info = Env_learning_lp.step(delta_SL, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt='lp')[-1]
                current_delta = delta_SL            
            
            else:
                print('learning + lp: not feasible, model_number: %d' %(number_model))
                cntr_infeas_total_learning_lp +=1
                cntr_infeas_episode_learning_lp +=1

                # add by Xiaoyu_20241120
                for line in range(3):
                    if current_delta[N_control - 2, 1 + 4 * line] == 0:
                        match_lp[line] = match_constant[line]
                    else:
                        match_lp[line] = match_constant[line] + 1
            
                # add by Xiaoyu
                delta_recursive = np.zeros([N_control, 12], dtype=int)
                delta_recursive[:-1, :] = current_delta[1:, :]
                for line in range(3):
                    delta_recursive[N_control - 1, 0 + 4 * line] = current_delta[N_control - 2, 0 + 4 * line]
                    delta_recursive[N_control - 1, 1 + 4 * line] = current_delta[N_control - 2, 1 + 4 * line]
                    delta_recursive[N_control - 1, 2 + 4 * line] = current_delta[N_control - 1 - match_lp[line], 2 + 4 * line]
                    delta_recursive[N_control - 1, 3 + 4 * line] = current_delta[N_control - 1 - match_lp[line], 3 + 4 * line]
                a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl_learning_lp = gurobi_lp_presolve(N, Env_learning_lp.d_pre_cut, Env_learning_lp.state_rho, Env_learning_lp.state_a,Env_learning_lp.state_d, Env_learning_lp.state_r, Env_learning_lp.state_l, Env_learning_lp.state_y,Env_learning_lp.state_n, Env_learning_lp.state_depot, delta_recursive, mipgap, log, opt_time, n_threads)

                # add by Xiaoyu_20241120
                if mdl_feasible(mdl_learning_lp):
                    info = Env_learning_lp.step(delta_recursive, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt='lp')[-1]
                    current_delta = delta_recursive
                else:
                    print('recursive solution (lp) is infeasible')
                    # add by Xiaoyu_20241120
                    a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl_learning_lp = gurobi_lp_presolve(N, Env_learning_lp.d_pre_cut, Env_learning_lp.state_rho, Env_learning_lp.state_a,Env_learning_lp.state_d, Env_learning_lp.state_r, Env_learning_lp.state_l, Env_learning_lp.state_y, Env_learning_lp.state_n, Env_learning_lp.state_depot, stacked_delta_heuristic1, mipgap, log, opt_time, n_threads)

                    if mdl_feasible(mdl_learning_lp):
                        # cost_learning_nlp = store_data(Env_learning_nlp, list_learning_nlp, mdl, warm_start, cost_reference, total_inf_time, number_model)
                        info = Env_learning_lp.step(stacked_delta_heuristic1, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt='lp')[-1]
                        current_delta = stacked_delta_heuristic1
                    else:
                        cntr_infeas_total_heuristic_lp += 1
                        flag_end_episode = True
                        print('heuristic solution (lp) is infeasible')
            
                # if mdl_feasible(mdl_learning_lp)==False:
                #     cntr_infeas_total_heuristic_lp +=1
                #     flag_end_episode = True
                #     print('heuristic solution (lp) is infeasible')
            
            # # learning + lp
            # state_learning = build_stacked_state(Env_learning_lp.state_n, Env_learning_lp.state_rho, Env_learning_lp.state_depot, Env_learning_lp.state_l, input_size, N_control, state_min_reduced, state_max_reduced)            
            # start_time=time.time()
            # total_inf_time=0
            # early_term = 0
            # warm_start = 0
            # opt_time = timelimit
            # for i in range(len(model_list)):
            #     start_inf_time = time.time()
            #     # output_net = model_tscript_list[i](state_learning, h0[i], c0[i])*mask
            #     output_net = model_list[i](state_learning, h0[i], c0[i])
            #     action_idx = total_action_set[torch.max(output_net, dim=2)[1].squeeze().numpy()]
            #     delta_SL = build_delta_vector(action_idx, N_control, action_dict_reduced)
            #     inf_time = time.time()-start_inf_time
            #     total_inf_time += inf_time

            #     opt_time = opt_time - inf_time
                
            #     a_values, d_values, r_values, l_values, y_values, n_values, n_after_values,  mdl_learning_lp = gurobi_lp_presolve(N, Env_learning_lp.d_pre_cut, Env_learning_lp.state_rho, Env_learning_lp.state_a, Env_learning_lp.state_d, Env_learning_lp.state_r, Env_learning_lp.state_l, Env_learning_lp.state_y, Env_learning_lp.state_n, Env_learning_lp.state_depot, delta_SL, mipgap, log, opt_time, n_threads)
                
            #     number_model = i+1
                
            #     if mdl_feasible(mdl_learning_lp)==True:
            #         break
            
            # if mdl_feasible(mdl_learning_lp)==True:
            #     info = Env_learning_lp.step(delta_SL, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt='lp')[-1]
            #     current_delta = delta_SL
                
            # else:
            #     print('learning + lp: not feasible, model_number: %d' %(number_model))
            #     cntr_infeas_total_learning_lp +=1
            #     cntr_infeas_episode_learning_lp +=1
                
            #     # add by Xiaoyu
            #     delta_recursive = np.zeros([N_control, 12], dtype=int)
            #     delta_recursive[:-1, :] = current_delta[1:, :]
            #     for line in range(3):
            #         delta_recursive[N_control - 1, 0 + 4 * line] = current_delta[N_control - 2, 0 + 4 * line]
            #         delta_recursive[N_control - 1, 1 + 4 * line] = current_delta[N_control - 2, 1 + 4 * line]
            #         delta_recursive[N_control - 1, 2 + 4 * line] = current_delta[N_control - 1 - match[line], 2 + 4 * line]
            #         delta_recursive[N_control - 1, 3 + 4 * line] = current_delta[N_control - 1 - match[line], 3 + 4 * line]
            #     info = Env_learning_lp.step(delta_recursive, d_pre, rho_whole, mipgap, log, opt_time, early_term, warm_start, n_threads, opt='lp')[-1]
            #     current_delta = delta_recursive

            #     mdl_learning_lp = info['mdl']
                
            #     if mdl_feasible(mdl_learning_lp)==False:
            #         cntr_infeas_total_heuristic_lp += 1
            #         flag_end_episode = True
            #         print('heuristic solution is infeasible')
                    
            if (Env_learning_nlp.terminated or Env_learning_nlp.truncated) or (Env_learning_lp.terminated or Env_learning_lp.truncated) or j >= 30:
                flag_end_episode = True  
                    
            # store data          
            # if (mdl_feasible(mdl_learning_nlp) and mdl_feasible(mdl_learning_lp))==True:
            if not flag_end_episode:
                
                # minlp_net_ws_10m (reference)
                cost_reference = store_data(Env_minlp_net_ws_10m, list_minlp_net_ws_10m, mdl_reference, 1, 0)
                cost_episode_minlp_net_ws_10m += cost_reference
                runtime_episode_minlp_net_ws_10m += mdl_reference._Runtime
                
                # minlp_ws
                cost_minlp_ws = store_data(Env_minlp_ws, list_minlp_ws, mdl_minlp_ws, 1, cost_reference)
                cost_episode_minlp_ws += cost_minlp_ws
                runtime_episode_minlp_ws += mdl_minlp_ws._Runtime
                
                # milp
                cost_milp = store_data(Env_milp, list_milp, mdl_milp, 0, cost_reference)
                cost_episode_milp += cost_milp
                runtime_episode_milp += mdl_milp.Runtime
                
                # learning approaches
                cost_learning_nlp = store_data(Env_learning_nlp, list_learning_nlp, mdl_learning_nlp, warm_start, cost_reference, total_inf_time, number_model)
                print('infeasibility (nlp) (episode): %d out of %d, \t infeasibility (total): %d out of %d'%(cntr_infeas_episode_learning_nlp, j+1, cntr_infeas_total_learning_nlp, k))
                cost_learning_lp = store_data(Env_learning_lp, list_learning_lp, mdl_learning_lp, warm_start, cost_reference, total_inf_time, number_model)
                print('infeasibility (lp) (episode): %d out of %d, \t infeasibility (total): %d out of %d'%(cntr_infeas_episode_learning_lp, j+1, cntr_infeas_total_learning_lp, k))
                
                cost_episode_learning_nlp += cost_learning_nlp
                runtime_episode_learning_nlp += mdl_learning_nlp.Runtime + total_inf_time
                
                cost_episode_learning_lp += cost_learning_lp
                runtime_episode_learning_lp += mdl_learning_lp.Runtime + total_inf_time         

                step_cost_list_stack[-1][0,j,0] = cost_episode_minlp_net_ws_10m
                step_cost_list_stack[-1][1,j,0] = cost_episode_minlp_ws
                step_cost_list_stack[-1][2,j,0] = cost_episode_milp
                step_cost_list_stack[-1][3,j,0] = cost_episode_learning_nlp
                step_cost_list_stack[-1][4,j,0] = cost_episode_learning_lp         
                
                opt_gap_list_stack[-1][0,j,0] = get_gap(cost_episode_minlp_net_ws_10m, cost_episode_minlp_net_ws_10m)
                opt_gap_list_stack[-1][1,j,0] = get_gap(cost_episode_minlp_ws, cost_episode_minlp_net_ws_10m)
                opt_gap_list_stack[-1][2,j,0] = get_gap(cost_episode_milp, cost_episode_minlp_net_ws_10m)
                opt_gap_list_stack[-1][3,j,0] = get_gap(cost_episode_learning_nlp, cost_episode_minlp_net_ws_10m)
                opt_gap_list_stack[-1][4,j,0] = get_gap(cost_episode_learning_lp, cost_episode_minlp_net_ws_10m)
                
                runtime_list_stack[-1][0,j,0] = runtime_episode_minlp_net_ws_10m
                runtime_list_stack[-1][1,j,0] = runtime_episode_minlp_ws
                runtime_list_stack[-1][2,j,0] = runtime_episode_milp
                runtime_list_stack[-1][3,j,0] = runtime_episode_learning_nlp
                runtime_list_stack[-1][4,j,0] = runtime_episode_learning_lp
                
                print('episode costs: \t net_ws_10m: %.2f net %.2f net_ws %.2f minlp %.2f minlp_ws %.2f milp %.2f learn_nlp %.2f learn_lp %.2f' %(cost_episode_minlp_net_ws_10m, cost_episode_minlp_net, cost_episode_minlp_net_ws, cost_episode_minlp, cost_episode_minlp_ws, cost_episode_milp, cost_episode_learning_nlp, cost_episode_learning_lp))
                
                print('episode opt_gaps: \t net_ws_10m: %.2f net %.2f net_ws %.2f minlp %.2f minlp_ws %.2f milp %.2f learn_nlp %.2f learn_lp %.2f' %(get_gap(cost_episode_minlp_net_ws_10m, cost_episode_minlp_net_ws_10m), get_gap(cost_episode_minlp_net, cost_episode_minlp_net_ws_10m), get_gap(cost_episode_minlp_net_ws, cost_episode_minlp_net_ws_10m), get_gap(cost_episode_minlp, cost_episode_minlp_net_ws_10m), get_gap(cost_episode_minlp_ws, cost_episode_minlp_net_ws_10m), get_gap(cost_episode_milp, cost_episode_minlp_net_ws_10m), get_gap(cost_episode_learning_nlp, cost_episode_minlp_net_ws_10m), get_gap(cost_episode_learning_lp, cost_episode_minlp_net_ws_10m)))
                
                print('episode runtimes: \t net_ws_10m: %.2f net %.2f net_ws %.2f minlp %.2f minlp_ws %.2f milp %.2f learn_nlp %.2f learn_lp %.2f' %(runtime_episode_minlp_net_ws_10m, runtime_episode_minlp_net, runtime_episode_minlp_net_ws, runtime_episode_minlp, runtime_episode_minlp_ws, runtime_episode_milp, runtime_episode_learning_nlp, runtime_episode_learning_lp))
                
                print('elapsed_time = %.2f' %(time.time()-start_loop))
                
                j += 1
                k +=1
                idx_cntr +=1              
            
            #store data           
            if flag_end_episode:
                
                d_list_stack[-1][0,:,:,:] = Env_minlp_net_ws_10m.d_real[:,idx_cntr_init:idx_cntr_init+30,:]
                d_list_stack[-1][1,:,:,:] = Env_minlp_ws.d_real[:,idx_cntr_init:idx_cntr_init+30,:]
                d_list_stack[-1][2,:,:,:] = Env_milp.d_real[:,idx_cntr_init:idx_cntr_init+30,:]
                d_list_stack[-1][3,:,:,:] = Env_learning_nlp.d_real[:,idx_cntr_init:idx_cntr_init+30,:]
                d_list_stack[-1][4,:,:,:] = Env_learning_lp.d_real[:,idx_cntr_init:idx_cntr_init+30,:]
                
                a_list_stack[-1][0,:,:,:] = Env_minlp_net_ws_10m.a_real[:,idx_cntr_init:idx_cntr_init+30,:]
                a_list_stack[-1][1,:,:,:] = Env_minlp_ws.a_real[:,idx_cntr_init:idx_cntr_init+30,:]
                a_list_stack[-1][2,:,:,:] = Env_milp.a_real[:,idx_cntr_init:idx_cntr_init+30,:]
                a_list_stack[-1][3,:,:,:] = Env_learning_nlp.a_real[:,idx_cntr_init:idx_cntr_init+30,:]
                a_list_stack[-1][4,:,:,:] = Env_learning_lp.a_real[:,idx_cntr_init:idx_cntr_init+30,:]
                
                l_list_stack[-1][0,:,:,:] = Env_minlp_net_ws_10m.l_real[:,idx_cntr_init:idx_cntr_init+30,:]
                l_list_stack[-1][1,:,:,:] = Env_minlp_ws.l_real[:,idx_cntr_init:idx_cntr_init+30,:]
                l_list_stack[-1][2,:,:,:] = Env_milp.l_real[:,idx_cntr_init:idx_cntr_init+30,:]
                l_list_stack[-1][3,:,:,:] = Env_learning_nlp.l_real[:,idx_cntr_init:idx_cntr_init+30,:]
                l_list_stack[-1][4,:,:,:] = Env_learning_lp.l_real[:,idx_cntr_init:idx_cntr_init+30,:]
                
                n_list_stack[-1][0,:,:,:] = Env_minlp_net_ws_10m.n_real[:,idx_cntr_init:idx_cntr_init+30,:]
                n_list_stack[-1][1,:,:,:] = Env_minlp_ws.n_real[:,idx_cntr_init:idx_cntr_init+30,:]
                n_list_stack[-1][2,:,:,:] = Env_milp.n_real[:,idx_cntr_init:idx_cntr_init+30,:]
                n_list_stack[-1][3,:,:,:] = Env_learning_nlp.n_real[:,idx_cntr_init:idx_cntr_init+30,:]
                n_list_stack[-1][4,:,:,:] = Env_learning_lp.n_real[:,idx_cntr_init:idx_cntr_init+30,:]
                
                n_after_list_stack[-1][0,:,:,:] = Env_minlp_net_ws_10m.n_after_real[:,idx_cntr_init:idx_cntr_init+30,:]
                n_after_list_stack[-1][1,:,:,:] = Env_minlp_ws.n_after_real[:,idx_cntr_init:idx_cntr_init+30,:]
                n_after_list_stack[-1][2,:,:,:] = Env_milp.n_after_real[:,idx_cntr_init:idx_cntr_init+30,:]
                n_after_list_stack[-1][3,:,:,:] = Env_learning_nlp.n_after_real[:,idx_cntr_init:idx_cntr_init+30,:]
                n_after_list_stack[-1][4,:,:,:] = Env_learning_lp.n_after_real[:,idx_cntr_init:idx_cntr_init+30,:]
                
                n_before_list_stack[-1][0,:,:,:] = Env_minlp_net_ws_10m.n_before_real[:,idx_cntr_init:idx_cntr_init+30,:]
                n_before_list_stack[-1][1,:,:,:] = Env_minlp_ws.n_before_real[:,idx_cntr_init:idx_cntr_init+30,:]
                n_before_list_stack[-1][2,:,:,:] = Env_milp.n_before_real[:,idx_cntr_init:idx_cntr_init+30,:]
                n_before_list_stack[-1][3,:,:,:] = Env_learning_nlp.n_before_real[:,idx_cntr_init:idx_cntr_init+30,:]
                n_before_list_stack[-1][4,:,:,:] = Env_learning_lp.n_before_real[:,idx_cntr_init:idx_cntr_init+30,:]
                
                idx_cntr_list_stack[-1][0] = idx_cntr_init
                idx_cntr_list_stack[-1][1] = Env_milp.idx_cntr
                idx_cntr_list_stack[-1][2] = Env_milp.idx_cntr-idx_cntr_init
                
                env_status_list_stack[-1][0] = Env_learning_nlp.terminated
                env_status_list_stack[-1][1] = Env_learning_nlp.truncated
                env_status_list_stack[-1][2] = Env_learning_lp.terminated
                env_status_list_stack[-1][3] = Env_learning_lp.truncated   
                
                print(idx_cntr_init, Env_milp.idx_cntr, Env_milp.idx_cntr-idx_cntr_init)
                
                print('end of episode')
                
                break
            
            if time.time() > start_loop + timeout:
                break      
                        
        print('\n')        
    
array_minlp_net_ws_10m = np.array(list_minlp_net_ws_10m)
array_minlp_net = np.array(list_minlp_net)
array_minlp_net_ws = np.array(list_minlp_net_ws)
array_minlp = np.array(list_minlp)
array_minlp_ws = np.array(list_minlp_ws)
array_milp = np.array(list_milp)
array_learning_nlp = np.array(list_learning_nlp)
array_learning_lp = np.array(list_learning_lp)

array_d_real = np.array(d_list_stack)
array_a_real = np.array(a_list_stack)
array_l_real = np.array(l_list_stack)
array_n_real = np.array(n_list_stack)
array_n_after_real = np.array(n_after_list_stack)
array_n_before_real = np.array(n_before_list_stack)
array_idx_cntr = np.array(idx_cntr_list_stack)
array_step_cost = np.array(step_cost_list_stack)
array_opt_gap = np.array(opt_gap_list_stack)
array_runtime = np.array(runtime_list_stack)

dict_arrays = {
    'array_minlp_net_ws_10m': array_minlp_net_ws_10m,
    'array_minlp_net': array_minlp_net,
    'array_minlp_net_ws': array_minlp_net_ws,
    'array_minlp': array_minlp,
    'array_minlp_ws': array_minlp_ws,
    'array_milp': array_milp,
    'array_learning_nlp': array_learning_nlp,
    'array_learning_lp': array_learning_lp,
    'cntr_infeas_total_learning_nlp': cntr_infeas_total_learning_nlp,
    'cntr_infeas_total_learning_lp': cntr_infeas_total_learning_lp,
    'cntr_infeas_total_heuristic_nlp': cntr_infeas_total_heuristic_nlp,
    'cntr_infeas_total_heuristic_lp': cntr_infeas_total_heuristic_lp,
    'array_d_real': array_d_real,
    'array_a_real': array_a_real,
    'array_l_real': array_l_real,
    'array_n_real': array_n_real,
    'array_n_after_real': array_n_after_real,
    'array_n_before_real': array_n_before_real,
    'array_idx_cntr': array_idx_cntr,
    'array_step_cost': array_step_cost,
    'array_opt_gap': array_opt_gap,
    'array_runtime': array_runtime,
    'info': 'Each array has a list containing [mdl.Runtime, cost_minlp, perc_minlp_cost, mdl.status]. The arrays learning_nlp, learning_nlp have extra elements [inference_time, number_model]. \n Arrays for the episodes have the total [cost, opt_gap, runtime] for each episode.',
    'info_episode': 'Each array has a list containing [d_real, a_real, l_real, n_real, n_after_real, n_before_real, idx_cntr, step cost, step optimality gap]. \nd_real = [episode, method, data, data, data]. \nidx_cntr = [episode, 0:2]; 0: idx_cntr_init, 1: idx_cntr_end, 2: episode length \n step_cost/opt_gap/runtime = [episode, method, episode number] ',
    'info_env_status': '0: nlp.terminated, 1: nlp.truncated, 2: lp.terminated, 3: lp.truncated'
    
}

if testing:
    np.save('tests//tests_learning_cl_' + opt_data + '%.2d_%.3d_%.3d.npy' %(N,timelimit,job_idx), dict_arrays)
else:
    np.save('/scratch/cfoliveiradasi/railway_largescale/tests/' + 'tests_learning_cl_' + opt_data + '_N%.2d_%.3d_%.3d.npy' %(N,timelimit,job_idx), dict_arrays)
    
x = datetime.datetime.now()
date_str = '%.2d%.2d_%.2d%.2d' %(x.month, x.day, x.hour, x.minute)
print('date_time: ' + date_str)

print('total time: %f' %(time.time()-start_loop))
print('test completed!')