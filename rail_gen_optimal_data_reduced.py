'''
this file generates the data for the training of the supervised learning approach
'''

import numpy as np
from rail_rl_env import RailNet, gurobi_milp, gurobi_minlp, mdl_feasible, d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot, num_station
from rail_fun import downsample_average_state_rho, build_list_action
import time
import datetime
import sys
import os, psutil

start_time = time.time()

testing = False

if testing==True:
    opt = 'milp_ol'
    N = 40
    job_idx = 999
    n_threads = 1
    timeout = 10*60 # for testing
    timelimit_gurobi = 10
    print_param = 1
else:
    opt = sys.argv[1] # available options: 'milp_ol', 'milp_cl', 'minlp_ol', 'minlp_cl'
    N = int(sys.argv[2])
    job_idx = int(sys.argv[3])
    time_limit = int(sys.argv[4])
    n_threads = int(sys.argv[5])
    timeout = (time_limit-1)*60*60 # in hours
    timelimit_gurobi = 240
    print_param = 10

if opt == 'milp_ol' or opt=='milp_cl':
    mipgap = 1e-3
    timelimit_gurobi = 60
else:
    mipgap = 1e-3
    
early_term = 0
warm_start = 1
N_control = N-2

dict_state_list = {
    'state_n' : [],
    'state_depot' : [],
    'idx_cntr' : [],
    'state_l_0' : [],
    'state_l_1' : [],
    'state_l_2' : [],
    'state_rho_down' : [],
    'state_rho_mean' : []
}

dict_output_list = {
    'delta': [],
    'list_actions': [],
    'mdl_Obj': [],
    'mdl_mipgap': [],
    'mdl_runtime': [],
    'mdl_status': []
}

def get_State(Env):
    
    state_n = np.concatenate((Env.state_n[0,0:2*num_station[0]-1], Env.state_n[1,0:2*num_station[1]-1], Env.state_n[2,0:2*num_station[2]-1]))
    state_depot = Env.state_depot

    d = {}
    for i in range(3):
        d['state_l_' + str(i)] = np.concatenate((Env.state_l[0,i,0:2*num_station[0]-1], Env.state_l[1,i,0:2*num_station[1]-1], Env.state_l[2,i,0:2*num_station[2]-1]))
    state_l_0 = d['state_l_0']
    state_l_1 = d['state_l_1']
    state_l_2 = d['state_l_2']

    state_rho_down = downsample_average_state_rho(Env.state_rho, 9) # 4 samples of the passenger flow

    d = {}
    for i in range(4):
        d[str(i)] = np.concatenate((state_rho_down[0,i,0:2*num_station[0]-1], state_rho_down[1,i,0:2*num_station[1]-1], state_rho_down[2,i,0:2*num_station[2]-1]))
    state_rho_down = np.concatenate((d['0'], d['1'], d['2'], d['3']))

    state_rho_mean = np.mean(Env.state_rho, 1)
    state_rho_mean = np.concatenate((state_rho_mean[0,0:2*num_station[0]-1], state_rho_mean[1,0:2*num_station[1]-1], state_rho_mean[2,0:2*num_station[2]-1]))

    idx_cntr = np.array([Env.idx_cntr])
    
    dict_state = {
    'state_n' : state_n,
    'state_depot' : state_depot,
    'idx_cntr' : idx_cntr,
    'state_l_0' : state_l_0,
    'state_l_1' : state_l_1,
    'state_l_2' : state_l_2,
    'state_rho_down' : state_rho_down,
    'state_rho_mean' : state_rho_mean
    }
    return dict_state

def store_State(dict, dict_state):
    for i in dict_state.keys():
        dict[i].append(dict_state[i])
        
def store_Output(dict, delta, mdl):
    dict['delta'].append(delta)
    dict['list_actions'].append(build_list_action(np.round(delta,2), N_control))
    dict['mdl_Obj'].append(np.array(mdl.ObjVal).reshape(1,))
    dict['mdl_mipgap'].append(np.array(mdl.mipgap).reshape(1,))
    dict['mdl_runtime'].append(np.array(mdl.runtime).reshape(1,))
    dict['mdl_status'].append(np.array(mdl.status).reshape(1,))    
    
milp_info_compressed = []
list_time_idx = []

cntr_feasible = 0
cntr_infeasible = 0

i = 1
Env = RailNet(N)

mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
print('\nram_usage(MB)=%.2f\n' %mem_usage)

while time.time() < start_time + timeout:
    
    del Env
    Env = RailNet(N)    
    Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)
    
    #open_loop
    if opt=='milp_ol' or opt == 'minlp_ol':
        
        if i==1:
            log=1
        else:
            log=0
        
        if opt=='milp_ol':
            _, _, _, _, _, delta, _, _, mdl = gurobi_milp(N,Env.d_pre_cut,Env.state_rho,Env.state_a,Env.state_d,Env.state_r,Env.state_l,Env.state_y,Env.state_n,Env.state_depot,mipgap,log,timelimit_gurobi,n_threads)
        elif opt=='minlp_ol':
            _, _, _, _, _, delta, _, _, mdl = gurobi_minlp(N,Env.d_pre_cut,Env.state_rho,Env.state_a,Env.state_d,Env.state_r,Env.state_l,Env.state_y,Env.state_n,Env.state_depot,mipgap,log,timelimit_gurobi,early_term,warm_start,n_threads)
        
        if mdl_feasible(mdl)==True:
            
            dict_state = get_State(Env)
            store_State(dict_state_list, dict_state)
            store_Output(dict_output_list, delta, mdl)
                       
            cntr_feasible += 1
            
            elapsed_time = time.time()-start_time
            
            mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
            
            if i % print_param == 0:
                print('i=%d' %i, 'elapsed_time=%.2f' %elapsed_time, 'avg_solution_time=%.2f' %(elapsed_time/(i)), 'mipgap=%.5f' %mdl.mipgap, 'runtime=%.2f' %mdl._Runtime, 'status=%d' %mdl.status, 'mem_usage=%.2f' %mem_usage)
                
            del dict_state, delta, mdl, _
        else:
            cntr_infeasible +=1
            print('not feasible.')
            del delta, mdl, _       
              
        i+=1

    #closed-loop
    elif opt=='milp_cl' or opt == 'minlp_cl':
        while (Env.terminated or Env.truncated)==False:
        
            if i==1:
                log=1
            else:
                log=0
                
            dict_state = get_State(Env)
            
            if opt=='milp_cl':
                _, _, _, _, _, _, _, _, _, _, terminated, truncated, delta, info = Env.step(0, d_pre, rho_whole, mipgap, log, timelimit_gurobi, early_term, warm_start, n_threads, 'milp')
            elif opt=='minlp_cl':
                _, _, _, _, _, _, _, _, _, _, terminated, truncated, delta, info = Env.step(0, d_pre, rho_whole, mipgap, log, timelimit_gurobi, early_term, warm_start, n_threads, 'minlp')
            
            mdl = info['mdl']
        
            if mdl_feasible(mdl)==True:
                
                store_State(dict_state_list, dict_state)
                store_Output(dict_output_list, delta, mdl)
                
                cntr_feasible += 1
                
                elapsed_time = time.time()-start_time
                
                mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
                
                if i % print_param == 0:
                        print('i=%d' %i, 'elapsed_time=%.2f' %elapsed_time, 'avg_solution_time=%.2f' %(elapsed_time/(i)), 'mipgap=%.5f' %mdl.mipgap, 'runtime=%.2f' %mdl._Runtime, 'status=%d' %mdl.status, 'ram_usage(MB)=%.2f' %mem_usage )
            
            else:
                cntr_infeasible +=1
                print('not feasible.')
                
            del delta, mdl, info, _
        
            i+=1
            
            if time.time() > start_time + timeout:
                break
            
for i in dict_state_list.keys():
    dict_state_list[i]=np.array(dict_state_list[i])
    
for i in dict_output_list.keys():
    dict_output_list[i]=np.array(dict_output_list[i])
      
x = datetime.datetime.now()

if testing==True:    
    np.save('data_optimal//data_reduced_%s_N%.2d_%.3d.npy' % (opt, N, job_idx), (dict_state_list, dict_output_list), allow_pickle=True)
else:
    np.save('/scratch/cfoliveiradasi/railway_largescale/data_optimal/data_reduced_%s_N%.2d_%.3d.npy' % (opt, N, job_idx), (dict_state_list, dict_output_list), allow_pickle=True)
    
elapsed_time = time.time() - start_time

print('cntr_feasible = %d' % cntr_feasible, 'cntr_infeasible = %d' %cntr_infeasible)
print('elapsed time = %.2f' % elapsed_time)
print('date and time : ' + '%.2d%.2d_%.2d%.2d%.2d' %(x.month, x.day, x.hour, x.minute, x.second))
print('completed')

# minlp_info_compressed = np.load('data_minlp//data_minlp_N%.2d.npy' %N, allow_pickle=True)
# N_datapoints = minlp_info_compressed.shape[0]
# state_n, state_rho, state_depot, state_l, delta_minlp = decompress_minlp_info(minlp_info_compressed[j,:])