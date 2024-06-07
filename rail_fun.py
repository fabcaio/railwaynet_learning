import numpy as np
from rail_rl_env import gurobi_minlp, qp_feasible, gurobi_qp_presolve, get_original_schedule
from rail_rl_env import d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot, r_max, r_min, differ, Cmax, sigma, same, num_station, num_train, E_regular, t_constant, h_min, tau_min, l_min, l_max, eta, action_dict
import torch
import time

inv_action_dict = {tuple(v[0]): int(k) for k, v in action_dict.items()}

def build_list_action(delta: np.array, N_control: np.int32) -> list:
    list_action = [inv_action_dict[tuple(delta[0])]]
    for i in range(1, N_control):
        list_action.append(inv_action_dict[tuple(delta[i])])
        
    return list_action

def build_delta_vector(list_action: list, N_control: np.int32) -> np.array:
        # from list of actions builds a np.array with the stacked deltas for each time step of the prediction horizon
        delta = action_dict[str(int(list_action[0]))]
        for i in range(1, N_control):
            delta = np.concatenate((delta, action_dict[str(int(list_action[i]))]))

        return delta

def get_dict_min_max_state(N):
    # dataset information for normalization and standardization
    num_passengers_max = un.transpose((1,0,2)).reshape(28,-1).max(axis=1)
    num_passengers_min = un.transpose((1,0,2)).reshape(28,-1).min(axis=1)

    rho_max = rho_whole.transpose((1,0,2)).reshape(28,-1).max(axis=1)
    rho_min = rho_whole.transpose((1,0,2)).reshape(28,-1).min(axis=1)

    rho_min = np.tile(rho_min, N+1)
    rho_max = np.tile(rho_max, N+1)

    depot_max = depot.transpose((1,0,2)).reshape(14,-1).max(axis=1)[[0,-1]]
    depot_min = depot.transpose((1,0,2)).reshape(14,-1).min(axis=1)[[0,-1]]

    train_composition_max = ul.transpose((1,0,2)).reshape(28,-1).max()
    train_composition_min = ul.transpose((1,0,2)).reshape(28,-1).min()
    
    dict_min_max_state = {'num_passengers_min': num_passengers_min, 'num_passengers_max': num_passengers_max,
                    'rho_min': rho_min, 'rho_max': rho_max,
                    'depot_min': depot_min, 'depot_max': depot_max,
                    'train_composition_min': train_composition_min, 'train_composition_max': train_composition_max
                    }
    
    return dict_min_max_state

def norm_state(state, state_min, state_max):
    # simple state normalization when the constraints are of the form state_min <= state <= state_max
        
    state_span = state_max - state_min
    
    state_norm = (state-state_min)/(state_span+1e-8)
    
    state_norm = (state_norm-0.5)*2
    
    return state_norm

def preprocess_state(state_n, state_rho, state_depot, state_l, dict_min_max):
    
    #concatenates and normalizes the state
    
    num_passengers = state_n
    rho_flat = state_rho.flatten()
    depot_reduced = state_depot[[0,-1]] # removes 0 padding

    #keeps the current train composition and also the past compositions at platforms 0 and 14
    train_composition_reduced = np.zeros(32,)
    train_composition_reduced[0:2] = np.flip(state_l[1:,0])
    train_composition_reduced[2:16] = state_l[0,:14] # only current train composition
    train_composition_reduced[16:18] = np.flip(state_l[1:,14])
    train_composition_reduced[18:] = state_l[0,14:]

    num_passengers_norm = norm_state(num_passengers, dict_min_max['num_passengers_min'], dict_min_max['num_passengers_max'])
    rho_flat_norm = norm_state(rho_flat, dict_min_max['rho_min'], dict_min_max['rho_max'])
    depot_reduced_norm = norm_state(depot_reduced, dict_min_max['depot_min'], dict_min_max['depot_max'])
    train_composition_reduced_norm = norm_state(train_composition_reduced, dict_min_max['train_composition_min'], dict_min_max['train_composition_max'])

    state_learning = np.concatenate((num_passengers_norm, rho_flat_norm, depot_reduced_norm, train_composition_reduced_norm))
    
    return state_learning

def temp_scheduler(mem_cntr, n_samples=80000, final_temp = 80, d1=5000):
    """
    temperature scheduler (linear) for softmax exploration
    n_samples: when temperature achieves if final value
    final_temp: final temperature
    d1: when temperature is equal to 0.5 (approximately equivalent to the period of random exploration)
    """
        
    d1 = 5000

    if mem_cntr <= d1:
        temp = mem_cntr*0.5/d1
    else:
        temp = 0.5 + (mem_cntr-d1)*(final_temp-0.5)/(n_samples-d1)
        
    # return max(temp, final_temp)
    return temp

def build_stacked_state(state_n, state_rho, state_depot, state_l, input_size, N_control, dict_min_max_state):
    state = np.zeros((1,N_control, input_size))
    state[0,0, :] = preprocess_state(state_n, state_rho, state_depot, state_l, dict_min_max_state)
    state[0,1:,:] = np.zeros((N_control-1, input_size))
    
    return state

def cost_per_step(n, n_after, d_pre_cut, d, l, sign_o, N):
    # n: waiting passengers
    # n_after: waiting passengers after departure
    # d_pre_cut: previous departure times
    # sign_o: changes in the composition (see paper)
    
    cost = sum(eta*n[k,s]*(d[k,s]-d_pre_cut[k,s])+eta*n_after[k,s]*(d_pre_cut[k+1,s]-d[k,s]) + l[k,s]*E_regular[s] + sign_o[k,num_station]*50 for k in range(N) for s in range(2*num_station))
    # cost = sum(eta * n[k, s] * (d[k, s] - d_pre_cut[k, s]) + eta * n_after[k, s] * (d_pre_cut[k + 1, s] - d[k, s]) + l[k, s] * E_regular[s] + sign_o[k,num_station]*50 for k in range(N) for s in range(2 * num_station))
    return cost

def get_optimality_gap_cl(network, Env, Env2, N_control, N_iter, dict_min_max_state, idx_cntr_min=15, idx_cntr_max=200, n_threads=1):
    """computes the closed loop optimality gap (agent vs MILP) for the microgrid system
    input: trained agent, environment, number of random iterations
    output: averaged cost for agent, average optimal cost, and optimality gap"""

    cost_lstm_mem = []
    cost_optimal_mem = []
    cost_original_mem = []

    h0 = torch.zeros(network.num_layers, 1, network.hidden_size)
    c0 = torch.zeros(network.num_layers, 1, network.hidden_size)
    
    cntr_infeasible = 0
    
    n_points = 0
    
    script_model = torch.jit.optimize_for_inference(torch.jit.script(network.eval()))
    
    # saves trajectory
    list_a = []
    list_d = []
    list_l = []
    list_n = []
    list_n_after = []
    list_stepcost = []

    for i in range(N_iter):
        
        #saves trajectory
        tmp_a = []
        tmp_d = []
        tmp_l = []
        tmp_n = []
        tmp_n_after = []
        tmp_stepcost = []   
        
        Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)
        Env.idx_cntr = np.random.randint(idx_cntr_min, idx_cntr_max)
        Env2.copyEnv(Env)
        
        while (Env.terminated or Env.truncated)==False and (Env2.terminated or Env2.truncated)==False:
            
            state = build_stacked_state(Env.state_n, Env.state_rho, Env.state_depot, Env.state_l, network.input_size, N_control, dict_min_max_state)
            state_learning = torch.tensor(state, dtype=torch.float32)    
            output_net = script_model(state_learning, h0, c0)
            action_idx = torch.max(output_net, dim=2)[1].squeeze().numpy()
            
            info = Env.step(action_idx, d_pre, rho_whole, r_max, r_min, differ, Cmax, sigma, same, num_station, num_train, E_regular, n_threads, "qp")[-1]
            info_minlp = Env2.step(action_idx, d_pre, rho_whole, r_max, r_min, differ, Cmax, sigma, same, num_station, num_train, E_regular, n_threads, "minlp")[-1]     
            
            if info_minlp['feasible']==True and info['feasible']==True:                
                cost_lstm = cost_per_step(Env.n, Env.n_after, Env.d_pre_cut_old, Env.d, Env.l, Env.sign_o, 1)
                cost_optimal = cost_per_step(Env2.n, Env2.n_after, Env2.d_pre_cut_old, Env2.d, Env2.l, Env2.sign_o, 1)
                
                a, d, r, l, y, n, n_after, sign_o = get_original_schedule(N_control+1, Env.idx_group, Env.start_index-1)
                cost_original=cost_per_step(n, n_after, Env.d_pre_cut_old, d, l, sign_o, 1)
                cost_original_mem.append(cost_original)               
                
                cost_optimal_mem.append(cost_optimal)
                cost_lstm_mem.append(cost_lstm)
                
                #saves trajectory
                tmp_a.append([Env.a[0,:], Env2.a[0,:], a[0,:]])
                tmp_d.append([Env.d[0,:], Env2.d[0,:], d[0,:]])
                tmp_l.append([Env.l[0,:], Env2.l[0,:], l[0,:]])
                tmp_n.append([Env.n[0,:], Env2.n[0,:], n[0,:]])
                tmp_n_after.append([Env.n_after[0,:], Env2.n_after[0,:], n_after[0,:]])
                tmp_stepcost.append([cost_lstm, cost_optimal, cost_original])               

            elif info_minlp['feasible']==True and info['feasible']==False:
                cntr_infeasible += 1
                
            n_points += 1
            
        # saves trajectory
        list_a.append(tmp_a)
        list_d.append(tmp_d)
        list_l.append(tmp_l)
        list_n.append(tmp_n)
        list_n_after.append(tmp_n_after)
        list_stepcost.append(tmp_stepcost)
        
    if len(cost_lstm_mem) == 0:
        #in case there are no feasible actions
        return np.inf, np.inf, np.inf, np.inf
    else:         
        avg_cost_lstm = sum(cost_lstm_mem)/len(cost_lstm_mem)
        avg_cost_optimal = sum(cost_optimal_mem)/len(cost_optimal_mem)
        avg_cost_original = sum(cost_original_mem)/len(cost_original_mem)
        optimality_gap = (avg_cost_lstm-avg_cost_optimal)/avg_cost_optimal
        optimality_gap_original = (avg_cost_original-avg_cost_optimal)/avg_cost_optimal
        
    print('\nn_infeasible: %.2f out of %d points, infeasibility rate = %.2f per cent' %(cntr_infeasible, n_points, cntr_infeasible/n_points*100))
    print('\nmean optimality gap of learning-based approach (closed-loop) is %.2f per cent' %(optimality_gap*100))
    print('mean optimality gap of original approach (closed-loop) is %.2f per cent' %(optimality_gap_original*100))
    print('average cost of optimal/learning/original approaches are %.2f, %.2f, %.2f' %(avg_cost_optimal, avg_cost_lstm, avg_cost_original))
        
    return list_a, list_d, list_l, list_n, list_n_after, list_stepcost

def get_comparison(network, Env, N_control, N_iter, dict_min_max_state, idx_cntr_min=15, idx_cntr_max=200, n_threads=1):
    
    #compares the learning-based approach with the optimal solution (open-loop)
    #outputs: time, optimality gap, feasibility

    time_lstm = []
    time_minlp = []

    cost_lstm = []
    cost_minlp = []
    cost_original = []

    h0 = torch.zeros(network.num_layers, 1, network.hidden_size)
    c0 = torch.zeros(network.num_layers, 1, network.hidden_size)

    script_model = torch.jit.optimize_for_inference(torch.jit.script(network.eval()))
    
    cntr_infeasible = 0
    
    for i in range(N_iter):
        Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)
        Env.idx_cntr = np.random.randint(idx_cntr_min, idx_cntr_max)

        start_time = time.time()
        state = build_stacked_state(Env.state_n, Env.state_rho, Env.state_depot, Env.state_l, network.input_size, N_control, dict_min_max_state)
        state_learning = torch.tensor(state, dtype=torch.float32)    
        output_net = script_model(state_learning, h0, c0)
        action_idx = torch.max(output_net, dim=2)[1].squeeze().numpy()
        delta_SL = build_delta_vector(action_idx, N_control)
        mdl = gurobi_qp_presolve(Env.control_trains,Env.d_pre_cut,Env.state_rho, Env.state_a, Env.state_d,
                            Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, delta_SL,
                            num_station,differ,sigma,same,t_constant,h_min,l_min,l_max,r_min,
                            r_max,tau_min,E_regular,Cmax,eta, n_threads)[-1]
        end_time = time.time()
        time_lstm.append(end_time - start_time)

        start_time = time.time()
        mdl_minlp = gurobi_minlp(Env.control_trains,Env.d_pre_cut,Env.state_rho, Env.state_a, Env.state_d,
                        Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot,
                        num_station,differ,sigma,same,t_constant,h_min,l_min,l_max,r_min,
                        r_max,tau_min,E_regular,Cmax,eta, n_threads)[-1]
        end_time = time.time()
        time_minlp.append(end_time - start_time)
        
        a, d, r, l, y, n, n_after, sign_o = get_original_schedule(N_control+1, Env.idx_group, Env.start_index)
        
        
        if (mdl_minlp.status == 2) and (mdl.status == 2):
            cost_lstm.append(mdl.ObjVal)
            cost_minlp.append(mdl_minlp.ObjVal)
            
            cost_original.append(cost_per_step(n, n_after, Env.d_pre_cut_old, d, l, sign_o, N_control+1))
            
        if (mdl_minlp.status == 2) and (mdl.status != 2):
            cntr_infeasible += 1
            
        avg_cost_lstm = sum(cost_lstm)/len(cost_lstm)
        avg_cost_optimal = sum(cost_minlp)/len(cost_minlp)
        avg_cost_original = sum(cost_original)/len(cost_original)
        optimality_gap = (avg_cost_lstm-avg_cost_optimal)/avg_cost_optimal
        optimality_gap_original = (avg_cost_original-avg_cost_optimal)/avg_cost_optimal
        
    print('min/mean/max times (learning): %.3f/%.3f/%.3f sec' %(np.min(time_lstm), np.mean(time_lstm),np.max(time_lstm)))
    print('min/mean/max times (optimal): %.3f/%.3f/%.3f sec' %(np.min(time_minlp), np.mean(time_minlp),np.max(time_minlp)))
    print('mean speed-up (learning is %.2f times faster than optimal)' %(np.mean(time_minlp)/np.mean(time_lstm)) )
    print('\nn_infeasible: %.2f out of %d points, infeasibility rate = %.2f per cent' %(cntr_infeasible, N_iter, cntr_infeasible/N_iter*100))
    print('\nmean optimality gap of learning-based approach (open-loop) is %.2f per cent' %(optimality_gap*100))
    print('mean optimality gap of original approach (open-loop) is %.2f per cent' %(optimality_gap_original*100))
    print('average cost of optimal/learning/original approaches are %.2f, %.2f, %.2f' %(avg_cost_optimal, avg_cost_lstm, avg_cost_original))
            
    return time_lstm, time_minlp, cntr_infeasible, avg_cost_lstm, avg_cost_optimal, avg_cost_original