from rail_fun import cost_per_step, build_stacked_state, build_delta_vector
import numpy as np
from rail_rl_env import RailNet, d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot, mdl_feasible, action_dict, gurobi_minlp, gurobi_nlp_presolve, gurobi_lp_presolve, gurobi_milp
import time
from rail_data_preprocess_original import get_preprocessed_data
from rail_training import Network, Network_CE, Network_mask1, Network_mask2
import torch
import sys
import datetime

x = datetime.datetime.now()
date_str = '%.2d%.2d_%.2d%.2d' %(x.month, x.day, x.hour, x.minute)
print('date_time: ' + date_str)

start_loop = time.time()
testing = True

mipgap = 1e-3
log = 0
early_term = 1

if testing:
    opt_data = 'minlp_ol'
    N=40
    job_idx = 999
    timelimit = 25
    timeout = 3*60 #minutes
    n_threads = 4
    timelimit = 25
    timelimit_reference = 30 # sets the time for the longest minlp (with warm-start)
else:
    opt_data=sys.argv[1]
    N=int(sys.argv[2])
    job_idx = int(sys.argv[3])
    timelimit_job = int(sys.argv[4])
    n_threads = int(sys.argv[5])
    timeout = (timelimit_job-1.5)*60*60
    # timelimit = 240
    timelimit = int(sys.argv[6])
    timelimit_reference = 600 # sets the time for the longest minlp (with warm-start)

batch_size=1
device = "cpu"

Env = RailNet(N)

print('N: %d \t mipgap: %.4f \t timelimit: %.2f \t n_threads: %d \t early_term: %d' %(N, mipgap, timelimit, n_threads, early_term))


#######################################################################################################################################

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
            # print('number of parameters =', model_list[0].count_parameters())
            # test_accuracy(model_list[i], 32*100, stacked_states_val_tensor, stacked_actions_reduced_val, N_control, device)
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
            network.load_state_dict(torch.load('training_data//' + opt_data + '_N%d_%.3d_' %(N,i) + 'weight'))
            
            h0.append(torch.zeros(num_layers, 1, hidden_size))
            c0.append(torch.zeros(num_layers, 1, hidden_size))
            
            model_list.append(network)
            
        num_networks_minlp_ol = len(idx_best_networks)
        
        # first batch of networks
        # num_layers=1
        # model1_str = 'milp_ol_hs512_N40_tc25_s04_1908_2250_weight'
        # model2_str = 'milp_ol_hs256_N40_tc25_s05_2008_0005_weight'
        # model3_str = 'milp_ol_hs256_N40_tc25_s07_2008_0123_weight'
        # model4_str = 'milp_ol_hs256_N40_tc25_s11_2008_1140_weight'
        # model5_str = 'milp_ol_hs512_N40_tc25_s42_2008_1300_weight'
        # model_str_list = [model1_str, model2_str, model3_str, model4_str, model5_str]
        # model_list = []
        # h0 = []
        # c0 = []

        # seed_list = [4,5,7,11,42]
        # lr_list = [1e-4, 1e-3, 1e-4, 1e-5, 1e-5]
        # hidden_size_list = [512, 256, 256, 256, 512]
        # for i in range(len(model_str_list)):
        #     model_list.append(Network(input_size, hidden_size_list[i], num_layers, lr_list[i], num_actions, batch_size))
        #     tmp_str = model_str_list[i]
        #     model_list[i].load_state_dict(torch.load('training_data//' + tmp_str, map_location=torch.device('cpu')))
        #     h0.append(torch.zeros(num_layers, 1, hidden_size_list[i]))
        #     c0.append(torch.zeros(num_layers, 1, hidden_size_list[i]))

# torch script is not working for Network_mask1
# model_tscript_list = []
# for i in range(len(model_list)):
#     model_tscript_list.append(torch.jit.optimize_for_inference(torch.jit.script(model_list[i].eval())))

cntr_infeas_milp_nlp = 0
cntr_infeas_milp_lp = 0

list_minlp_nostop_ws_10m = []
list_minlp_nostop = []
list_minlp_nostop_ws = []

list_minlp = []
list_minlp_ws = []

list_milp = []
list_milp_nlp = []
list_milp_lp = []

with torch.no_grad():
    
    #initializes the inference
    for i in range(20):
        Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)
        state_learning = build_stacked_state(Env.state_n, Env.state_rho, Env.state_depot, Env.state_l, input_size, N_control, state_min_reduced, state_max_reduced)
        for i in range(len(model_list)):
            output_net = model_list[i](state_learning, h0[i], c0[i])
            action_idx = total_action_set[torch.max(output_net, dim=2)[1].squeeze().numpy()]
            delta_SL = build_delta_vector(action_idx, N_control, action_dict_reduced)
            
    j = 1

    while time.time() < start_loop + timeout :
        
        print('\nj=%d' %j)
            
        Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)
        state_learning = build_stacked_state(Env.state_n, Env.state_rho, Env.state_depot, Env.state_l, input_size, N_control, state_min_reduced, state_max_reduced)
        
        # minlp (warm-start, with early termination, 10 minutes)
        start_time=time.time()
        early_term = 0
        warm_start = 1
        a_values, d_values, r_values, l_values, y_values, _, n_values, n_after_values,  mdl = gurobi_minlp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit_reference, early_term, warm_start, n_threads)
        cost_minlp = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values,  N)
        list_minlp_nostop_ws_10m.append([mdl._Runtime, cost_minlp, 0, mdl.status])
        print('minlp_net_ws_10m \t %.2f \t\t %.2f \t\t %.2f \t\t 0%% \t %d' %(time.time()-start_time, mdl._Runtime, cost_minlp, mdl.status))
        
        # minlp (without early termination)
        start_time=time.time()
        early_term = 0
        warm_start = 0
        a_values, d_values, r_values, l_values, y_values, _, n_values, n_after_values,  mdl = gurobi_minlp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit, early_term, warm_start, n_threads)
        cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values,  N)
        perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
        list_minlp_nostop.append([mdl.Runtime, cost, perc_minlp_cost, mdl.status])
        print('minlp_net \t %.2f \t\t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(time.time()-start_time, mdl.Runtime, cost, perc_minlp_cost, mdl.status))
        
        # minlp (without early termination + warm start)
        start_time=time.time()
        early_term = 0
        warm_start = 1
        a_values, d_values, r_values, l_values, y_values, _, n_values, n_after_values,  mdl = gurobi_minlp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit, early_term, warm_start, n_threads)
        cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values,  N)
        perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
        list_minlp_nostop_ws.append([mdl._Runtime, cost, perc_minlp_cost, mdl.status])
        print('minlp_net_ws \t %.2f \t\t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(time.time()-start_time, mdl._Runtime, cost, perc_minlp_cost, mdl.status))
        
        # minlp (early termination, no warm start)
        start_time=time.time()
        early_term = 1
        warm_start = 0
        a_values, d_values, r_values, l_values, y_values, delta_minlp, n_values, n_after_values,  mdl = gurobi_minlp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit, early_term, warm_start, n_threads)
        cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values,  N)
        perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
        list_minlp.append([mdl.Runtime, cost, perc_minlp_cost, mdl.status])
        print('minlp \t \t %.2f \t\t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(time.time()-start_time, mdl.Runtime, cost, perc_minlp_cost, mdl.status))
        
        # minlp (early termination, warm start)
        start_time=time.time()
        early_term = 1
        warm_start = 1
        a_values, d_values, r_values, l_values, y_values, delta_minlp, n_values, n_after_values,  mdl = gurobi_minlp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit, early_term, warm_start, n_threads)
        cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values,  N)
        perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
        list_minlp_ws.append([mdl._Runtime, cost, perc_minlp_cost, mdl.status])
        print('minlp_ws \t \t %.2f \t\t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(time.time()-start_time, mdl._Runtime, cost, perc_minlp_cost, mdl.status))
        
        # milp
        start_time=time.time()
        a_values, d_values, r_values, l_values, y_values, delta_milp, n_values, n_after_values,  mdl = gurobi_milp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit, n_threads)
        cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values,  N)
        perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
        list_milp.append([mdl.Runtime, cost, perc_minlp_cost, mdl.status])
        print('milp \t \t %.2f \t\t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(time.time()-start_time, mdl.Runtime, cost, perc_minlp_cost, mdl.status))
        
        # learning + nlp
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
        
            a_values, d_values, r_values, l_values, y_values, n_values, n_after_values,  mdl = gurobi_nlp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, delta_SL, mipgap, log, opt_time, early_term, warm_start, n_threads)
            
            opt_time = opt_time - mdl.Runtime
            
            number_model = i+1
            
            if mdl_feasible(mdl):
                break
        
        if mdl_feasible(mdl):
            cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values,  N)
            perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
            list_milp_nlp.append([mdl.Runtime+total_inf_time, cost, perc_minlp_cost, mdl.status, total_inf_time , number_model])
            print('%s + nlp \t %.2f \t\t %.2f \t\t %.2f \t\t %.2f%% \t %d \t model_number: %d \t inf_time: %.4f' %(opt_data,time.time()-start_time, mdl.Runtime, cost, perc_minlp_cost, mdl.status, number_model, total_inf_time))
        else:
            list_milp_nlp.append([False, False, False, mdl.status, total_inf_time, number_model])
            cntr_infeas_milp_nlp +=1    
            print('%s + nlp: not feasible, model_number: %d' %(opt_data,number_model))
        print('infeasibility: %d out of %d'%(cntr_infeas_milp_nlp, j)) 
        
        # learning + lp        
        start_time=time.time()
        total_inf_time=0
        opt_time = timelimit
        for i in range(len(model_list)):
            start_inf_time = time.time()
            output_net = model_list[i](state_learning, h0[i], c0[i])
            action_idx = total_action_set[torch.max(output_net, dim=2)[1].squeeze().numpy()]
            delta_SL = build_delta_vector(action_idx, N_control, action_dict_reduced)
            inf_time = time.time()-start_inf_time
            total_inf_time += inf_time
            
            opt_time = opt_time - inf_time

            a_values, d_values, r_values, l_values, y_values, n_values, n_after_values,  mdl = gurobi_lp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, delta_SL, mipgap, log, opt_time, n_threads)
            
            opt_time = opt_time - mdl.Runtime
            
            number_model = i+1
            
            if mdl_feasible(mdl):
                break
        
        if mdl_feasible(mdl):
            cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values,  N)
            perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
            list_milp_lp.append([mdl.Runtime, cost, perc_minlp_cost, mdl.status, total_inf_time , number_model])
            print('%s + lp \t %.2f \t\t %.2f \t\t %.2f \t\t %.2f%% \t %d \t model_number: %d \t inf_time: %.4f' %(opt_data,time.time()-start_time, mdl.Runtime, cost, perc_minlp_cost, mdl.status, number_model, total_inf_time))
        else:
            list_milp_lp.append([False, False, False, mdl.status, total_inf_time, number_model])
            cntr_infeas_milp_lp +=1
            print('%s + lp: not feasible, model_number: %d \n' %(opt_data,number_model))
        print('infeasibility: %d out of %d'%(cntr_infeas_milp_lp, j))
        
        j+=1
        
    
array_minlp_nostop_ws_10m = np.array(list_minlp_nostop_ws_10m)
array_minlp_nostop = np.array(list_minlp_nostop)
array_minlp_nostop_ws = np.array(list_minlp_nostop_ws)

array_minlp = np.array(list_minlp)
array_minlp_ws = np.array(list_minlp_ws)

array_milp = np.array(list_milp)
array_milp_nlp = np.array(list_milp_nlp)
array_milp_lp = np.array(list_milp_lp)

dict_arrays = {
    'array_minlp_nostop_ws_10m': array_minlp_nostop_ws_10m,
    'array_minlp_nostop': array_minlp_nostop,
    'array_minlp_nostop_ws': array_minlp_nostop_ws,
    'array_minlp': array_minlp,
    'array_minlp_ws': array_minlp_ws,
    'array_milp': array_milp,
    'array_milp_nlp': array_milp_nlp,
    'array_milp_lp': array_milp_lp,
    'cntr_infeas_milp_nlp': cntr_infeas_milp_nlp,
    'cntr_infeas_milp_lp': cntr_infeas_milp_lp,
    'info': 'Each array has a list containing [mdl.Runtime, cost_minlp, perc_minlp_cost, mdl.status]. The arrays milp_nlp, milp_lp have extra elements [inference_time, number_model].'
}

if testing:
    np.save('tests//tests_learning_' + opt_data + '%.2d_%.3d_%.3d.npy' %(N,timelimit,job_idx), dict_arrays)
else:
    np.save('/scratch/cfoliveiradasi/railway_largescale/tests/' + 'tests_learning_' + opt_data + '_N%.2d_%.3d_%.3d.npy' %(N,timelimit,job_idx), dict_arrays)
    
x = datetime.datetime.now()
date_str = '%.2d%.2d_%.2d%.2d' %(x.month, x.day, x.hour, x.minute)
print('date_time: ' + date_str)

print('total time: %f' %(time.time()-start_loop))
print('test completed!')