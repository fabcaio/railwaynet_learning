import numpy as np
from rail_rl_env import un, ul, uy, ua, ud, ur, num_station, E_regular, action_dict, eta1, eta2, num_line, un_after
import torch

inv_action_dict = {tuple(v[0]): int(k) for k, v in action_dict.items()}

# def build_list_action(delta: np.array, N_control: np.int32) -> list:
#     list_action = [inv_action_dict[tuple(delta[0])]]
#     for i in range(1, N_control):
#         list_action.append(inv_action_dict[tuple(delta[i])])
        
#     return list_action

def build_list_action(delta: np.array, N_control: np.int32) -> list:
    delta = delta.astype(np.int32)
    list_action = [inv_action_dict[tuple(delta[0])]]
    for i in range(1, N_control):
        list_action.append(inv_action_dict[tuple(delta[i])])
        
    return list_action

def build_delta_vector(list_action: list, N_control: np.int32, action_dict: dict) -> np.array:
        # from list of actions builds a np.array with the stacked deltas for each time step of the prediction horizon
        delta = action_dict[str(int(list_action[0]))]
        for i in range(1, N_control):
            delta = np.concatenate((delta, action_dict[str(int(list_action[i]))]))

        return delta

def norm_state(state, state_min, state_max):
    # simple state normalization when the constraints are of the form state_min <= state <= state_max
        
    state_span = state_max - state_min
    
    state_norm = (state-state_min)/(state_span+1e-8)
    
    state_norm = (state_norm-0.5)*2
    
    return state_norm

def remove_station_zeros(in_vector, num_station=num_station):
    
    #input: state_n.shape (3,38)
    
    out_vector = np.zeros(2*np.sum(num_station)-3,)    
    out_vector[0 : num_station[0]*2 - 1] = in_vector[0, 0:num_station[0]*2 - 1]
    out_vector[num_station[0]*2 - 1 : num_station[0]*2 - 1 + num_station[1]*2 - 1] = in_vector[1, 0:num_station[1]*2 - 1]
    out_vector[num_station[0]*2 - 1 + num_station[1]*2 - 1 : num_station[0]*2 - 1 + num_station[1]*2 - 1 + num_station[2]*2 - 1] = in_vector[2, 0:num_station[2]*2 - 1]
    
    return out_vector

def preprocess_state(state_n, state_rho, state_depot, state_l):
    # receives the normalized state and then removes all the excess zeros, flattens and concatenates the states
    state_n_reduced = remove_station_zeros(state_n)

    tmp_vector = state_rho.transpose(1,0,2)
    tmp_list = []
    for i in range(tmp_vector.shape[0]):
        tmp_list.append(remove_station_zeros(tmp_vector[i]))

    state_rho_reduced = np.array(tmp_list).flatten()

    tmp_vector = state_l.transpose(1,0,2)
    state_l_reduced = remove_station_zeros(tmp_vector[0]) # this removes the train composition from the other time steps
    # tmp_list = [] # this keeps the composition of other time steps (alternative to the previous lines)
    # for i in range(tmp_vector.shape[0]):
    #     tmp_list.append(remove_station_zeros(tmp_vector[i]))
    # state_l_reduced = np.array(tmp_list).flatten()

    state_learning = np.concatenate((state_n_reduced, state_rho_reduced, state_depot, state_l_reduced))
    
    return state_learning

def build_stacked_state(state_n, state_rho, state_depot, state_l, input_size, N_control, state_min, state_max):
    
    # use reduced (without zeros) version of state_min/state_max

    state_reduced = preprocess_state(state_n, state_rho, state_depot, state_l)
    state_reduced_norm = norm_state(state_reduced, state_min, state_max)    
    state_reduced_norm = torch.tensor(state_reduced_norm, dtype=torch.float32).reshape(1, 1, -1)
    state_padding = torch.zeros(1, N_control-1, input_size)
    state_padded = torch.cat((state_reduced_norm, state_padding), dim=1)
    
    return state_padded

# state_n = Env.state_n, state_rho = Env.state_rho, state_depot = Env.state_depot, state_l = Env.state_l
# build_stacked_state(state_n, state_rho, state_depot, state_l, input_size, N_control)

def get_original_schedule(N, idx_group, start_index):
    
    #get_original_schedule(N, Env.idx_cntr, Env.start_index)
    
    max_station_len = np.max(num_station)
    
    a = np.zeros((num_line, N, 2* max_station_len))
    d = np.zeros((num_line, N, 2* max_station_len))
    r = np.zeros((num_line, N, 2* max_station_len))
    l = np.zeros((num_line, N, 2* max_station_len))
    y = np.zeros((num_line, N, 2* max_station_len))
    n = np.zeros((num_line, N, 2* max_station_len))
    n_after = np.zeros((num_line, N, 2* max_station_len))
    # sign_o = np.zeros((num_line, N, 2* max_station_len))
    
    for m in range(3):
        for s in range(2* max_station_len):
            a[m, :, s] = ua[m, start_index[m,s]:start_index[m,s]+N, s, idx_group]
            d[m, :, s] = ud[m, start_index[m,s]:start_index[m,s]+N, s, idx_group]
            r[m, :, s] = ur[m, start_index[m,s]:start_index[m,s]+N, s, idx_group]
            l[m, :, s] = ul[m, start_index[m,s]:start_index[m,s]+N, s, idx_group]
            y[m, :, s] = uy[m, start_index[m,s]:start_index[m,s]+N, s, idx_group]
            n[m, :, s] = un[m, start_index[m,s]:start_index[m,s]+N, s, idx_group]
            n_after[m, :, s] = un_after[m, start_index[m,s]:start_index[m,s]+N, s, idx_group]
        
    return a, d, r, l, y, n, n_after

def cost_per_step(n, n_after, d_pre_cut, d, l, N):
    # n: waiting passengers
    # n_after: waiting passengers after departure
    # d_pre_cut: previous departure times
    # sign_o: changes in the composition (see paper)
    
    cost = (sum(eta1*n[0,k,s]*(d[0,k,s]-d_pre_cut[0,k,s])+eta1*n_after[0,k,s]*(d_pre_cut[0,k+1,s]-d[0,k,s]) + eta2*l[0,k,s]*E_regular[0,s]
            for k in range(0,N) for s in range(2 * num_station[0])) +
    sum(eta1 * n[1, k, s] * (d[1, k, s] - d_pre_cut[1, k, s]) + eta1 * n_after[1, k, s] * (d_pre_cut[1, k + 1, s] - d[1, k, s]) + eta2*l[1, k, s] * E_regular[1, s]
            for k in range(0,N) for s in range(2 * num_station[1])) +
    sum(eta1 * n[2, k, s] * (d[2, k, s] - d_pre_cut[2, k, s]) + eta1 * n_after[2, k, s] * (d_pre_cut[2, k + 1, s] - d[2, k, s]) + eta2*l[2, k, s] * E_regular[2, s]
            for k in range(0,N) for s in range(2 * num_station[2])))
    
    return cost

def downsample_average_state_rho(arr, n):
    
    """
    inputs:
        arr: is expected to be Env.state_rho with shape (num_line, pred_horizon+1, 2*num_max_stations)=(3,41,38)
        n: is the downsampling step
    
    output: is the downsized passenger flow (3,m,38) where m is floor(41/n)
    """
    
    end =  n * int(arr.shape[1]/n)
    return np.mean(arr[:,:end,:].reshape(arr.shape[0], -1, n, arr.shape[2]), 2)

# a, d, r, l, y, n, n_after, sign_o = get_original_schedule(N, Env.idx_cntr, Env.start_index)
# cost_per_step(n, n_after, Env.d_pre_cut_old, d, l, sign_o, N)

# def get_optimality_gap_cl(network, Env, Env2, N_control, N_iter, dict_min_max_state, idx_cntr_min=15, idx_cntr_max=200, n_threads=1):
#     """computes the closed loop optimality gap (agent vs MILP) for the microgrid system
#     input: trained agent, environment, number of random iterations
#     output: averaged cost for agent, average optimal cost, and optimality gap"""

#     cost_lstm_mem = []
#     cost_optimal_mem = []
#     cost_original_mem = []

#     h0 = torch.zeros(network.num_layers, 1, network.hidden_size)
#     c0 = torch.zeros(network.num_layers, 1, network.hidden_size)
    
#     cntr_infeasible = 0
    
#     n_points = 0
    
#     script_model = torch.jit.optimize_for_inference(torch.jit.script(network.eval()))
    
#     # saves trajectory
#     list_a = []
#     list_d = []
#     list_l = []
#     list_n = []
#     list_n_after = []
#     list_stepcost = []

#     for i in range(N_iter):
        
#         #saves trajectory
#         tmp_a = []
#         tmp_d = []
#         tmp_l = []
#         tmp_n = []
#         tmp_n_after = []
#         tmp_stepcost = []   
        
#         Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot, idx_cntr_min, idx_cntr_max)
#         Env2.copyEnv(Env)
        
#         while (Env.terminated or Env.truncated)==False and (Env2.terminated or Env2.truncated)==False:
            
#             state = build_stacked_state(Env.state_n, Env.state_rho, Env.state_depot, Env.state_l, network.input_size, N_control, dict_min_max_state)
#             state_learning = torch.tensor(state, dtype=torch.float32)    
#             output_net = script_model(state_learning, h0, c0)
#             action_idx = torch.max(output_net, dim=2)[1].squeeze().numpy()
            
#             info = Env.step(action_idx, d_pre, rho_whole, r_max, r_min, differ, Cmax, sigma, same, num_station, num_train, E_regular, n_threads, "qp")[-1]
#             info_minlp = Env2.step(action_idx, d_pre, rho_whole, r_max, r_min, differ, Cmax, sigma, same, num_station, num_train, E_regular, n_threads, "minlp")[-1]     
            
#             if info_minlp['feasible']==True and info['feasible']==True:                
#                 cost_lstm = cost_per_step(Env.n, Env.n_after, Env.d_pre_cut_old, Env.d, Env.l, Env.sign_o, 1)
#                 cost_optimal = cost_per_step(Env2.n, Env2.n_after, Env2.d_pre_cut_old, Env2.d, Env2.l, Env2.sign_o, 1)
                
#                 a, d, r, l, y, n, n_after, sign_o = get_original_schedule(N_control+1, Env.idx_group, Env.start_index-1)
#                 cost_original=cost_per_step(n, n_after, Env.d_pre_cut_old, d, l, sign_o, 1)
#                 cost_original_mem.append(cost_original)               
                
#                 cost_optimal_mem.append(cost_optimal)
#                 cost_lstm_mem.append(cost_lstm)
                
#                 #saves trajectory
#                 tmp_a.append([Env.a[0,:], Env2.a[0,:], a[0,:]])
#                 tmp_d.append([Env.d[0,:], Env2.d[0,:], d[0,:]])
#                 tmp_l.append([Env.l[0,:], Env2.l[0,:], l[0,:]])
#                 tmp_n.append([Env.n[0,:], Env2.n[0,:], n[0,:]])
#                 tmp_n_after.append([Env.n_after[0,:], Env2.n_after[0,:], n_after[0,:]])
#                 tmp_stepcost.append([cost_lstm, cost_optimal, cost_original])               

#             elif info_minlp['feasible']==True and info['feasible']==False:
#                 cntr_infeasible += 1
                
#             n_points += 1
            
#         # saves trajectory
#         list_a.append(tmp_a)
#         list_d.append(tmp_d)
#         list_l.append(tmp_l)
#         list_n.append(tmp_n)
#         list_n_after.append(tmp_n_after)
#         list_stepcost.append(tmp_stepcost)
        
#     if len(cost_lstm_mem) == 0:
#         #in case there are no feasible actions
#         return np.inf, np.inf, np.inf, np.inf
#     else:         
#         avg_cost_lstm = sum(cost_lstm_mem)/len(cost_lstm_mem)
#         avg_cost_optimal = sum(cost_optimal_mem)/len(cost_optimal_mem)
#         avg_cost_original = sum(cost_original_mem)/len(cost_original_mem)
#         optimality_gap = (avg_cost_lstm-avg_cost_optimal)/avg_cost_optimal
#         optimality_gap_original = (avg_cost_original-avg_cost_optimal)/avg_cost_optimal
        
#     print('\nn_infeasible: %.2f out of %d points, infeasibility rate = %.2f per cent' %(cntr_infeasible, n_points, cntr_infeasible/n_points*100))
#     print('\nmean optimality gap of learning-based approach (closed-loop) is %.2f per cent' %(optimality_gap*100))
#     print('mean optimality gap of original approach (closed-loop) is %.2f per cent' %(optimality_gap_original*100))
#     print('average cost of optimal/learning/original approaches are %.2f, %.2f, %.2f' %(avg_cost_optimal, avg_cost_lstm, avg_cost_original))
        
#     return list_a, list_d, list_l, list_n, list_n_after, list_stepcost