'''
this file generates the data for the training of the supervised learning approach
'''

import numpy as np
from rail_rl_env import RailNet, gurobi_milp, gurobi_minlp, mdl_feasible
from rail_rl_env import d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot

import time
import datetime
import sys
import os, psutil

start_time = time.time()

testing = False

if testing==True:
    opt = 'milp_cl'
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
    timeout = (time_limit-1.5)*60*60 # in hours
    timelimit_gurobi = 240
    print_param = 10

mipgap = 1e-3
early_term = 0
warm_start = 1

def compress_milp_info(state_n, state_rho, state_depot, state_l, delta_milp, mdl_Obj, mdl_mipgap, mdl_runtime, mdl_status):
    
    milp_info_compressed = np.concatenate((state_n.flatten(), state_rho.flatten(), state_depot, state_l.flatten(), delta_milp.flatten(), mdl_Obj, mdl_mipgap, mdl_runtime, mdl_status))
    
    return milp_info_compressed

milp_info_compressed = []
list_time_idx = []

cntr_feasible = 0
cntr_infeasible = 0

def append_datapoint(state_n, state_rho, state_depot, state_l, delta, mdl):
    mdl_Obj = np.array(mdl.ObjVal).reshape(1,)
    mdl_mipgap = np.array(mdl.MIPGap).reshape(1,)
    mdl_runtime = np.array(mdl.Runtime).reshape(1,)
    mdl_status = np.array(mdl.Status).reshape(1,)
        
    milp_info_compressed.append(compress_milp_info(state_n, state_rho, state_depot, state_l, delta, mdl_Obj, mdl_mipgap, mdl_runtime, mdl_status))
    
    return

i = 1
Env = RailNet(N)
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
            _, _, _, _, _, delta_milp, _, _, mdl = gurobi_milp(N,Env.d_pre_cut,Env.state_rho,Env.state_a,Env.state_d,Env.state_r,Env.state_l,Env.state_y,Env.state_n,Env.state_depot,mipgap,log,timelimit_gurobi,n_threads)
        elif opt=='minlp_ol':
            _, _, _, _, _, delta_milp, _, _, mdl = gurobi_minlp(N,Env.d_pre_cut,Env.state_rho,Env.state_a,Env.state_d,Env.state_r,Env.state_l,Env.state_y,Env.state_n,Env.state_depot,mipgap,log,timelimit_gurobi,early_term,warm_start,n_threads)
        
        if mdl_feasible(mdl)==True:
            state_n = Env.state_n, 
            state_rho = Env.state_rho
            state_depot = Env.state_depot
            state_l = Env.state_l
            
            append_datapoint(state_n[0], state_rho, state_depot, state_l, delta_milp, mdl)
            list_time_idx.append(Env.idx_cntr)
            
            cntr_feasible += 1
            
            elapsed_time = time.time()-start_time  
            
            if i % print_param == 0:
                print('i=%d' %i, 'elapsed_time=%.2f' %elapsed_time, 'avg_solution_time=%.2f' %(elapsed_time/(i)), 'n_points=%d' %(len(milp_info_compressed)), 'mipgap=%.5f' %mdl.mipgap, 'runtime=%.2f' %mdl._Runtime, 'status=%d' %mdl.status )
                
            del state_n, state_rho, state_depot, state_l, delta_milp, mdl, _
        else:
            cntr_infeasible +=1
            print('not feasible.')
            del state_n, state_rho, state_depot, state_l, delta_milp, mdl, _       
              
        i+=1

    #closed-loop
    elif opt=='milp_cl' or opt == 'minlp_cl':
        while (Env.terminated or Env.truncated)==False:
        
            if i==1:
                log=1
            else:
                log=0
                
            state_n = Env.state_n
            state_rho = Env.state_rho
            state_depot = Env.state_depot
            state_l = Env.state_l
            
            if opt=='milp_cl':
                _, _, _, _, _, _, _, _, _, _, terminated, truncated, delta_milp, info = Env.step(0, d_pre, rho_whole, mipgap, log, timelimit_gurobi, early_term, warm_start, n_threads, 'milp')
            elif opt=='minlp_cl':
                _, _, _, _, _, _, _, _, _, _, terminated, truncated, delta_milp, info = Env.step(0, d_pre, rho_whole, mipgap, log, timelimit_gurobi, early_term, warm_start, n_threads, 'minlp')
            
            mdl = info['mdl']
        
            if mdl_feasible(mdl)==True:
                
                append_datapoint(state_n, state_rho, state_depot, state_l, delta_milp, mdl)
                list_time_idx.append(Env.idx_cntr)
                
                cntr_feasible += 1
                
                elapsed_time = time.time()-start_time
                
                mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
                
                if i % print_param == 0:
                        print('i=%d' %i, 'elapsed_time=%.2f' %elapsed_time, 'avg_solution_time=%.2f' %(elapsed_time/(i)), 'n_points=%d' %(len(milp_info_compressed)), 'mipgap=%.5f' %mdl.mipgap, 'runtime=%.2f' %mdl._Runtime, 'status=%d' %mdl.status, 'ram_usage(MB)=%.2f' %mem_usage )
            
            else:
                cntr_infeasible +=1
                print('not feasible.')
                
            del state_n, state_rho, state_depot, state_l, delta_milp, mdl, info, _
        
            i+=1
            
            if time.time() > start_time + timeout:
                break
    
milp_info_compressed = np.array(milp_info_compressed)
list_time_idx = np.array(list_time_idx)

dict_array={
    'milp_info_compressed': milp_info_compressed,
    'list_time_idx': list_time_idx
}
    
x = datetime.datetime.now()

if testing==True:    
    np.save('data_optimal//data_%s_N%.2d_%.3d.npy' % (opt, N, job_idx), dict_array, allow_pickle=True)
else:
    np.save('/scratch/cfoliveiradasi/railway_largescale/data_optimal/data_%s_N%.2d_%.3d.npy' % (opt, N, job_idx), dict_array, allow_pickle=True)
    
elapsed_time = time.time() - start_time

print('cntr_feasible = %d' % cntr_feasible, 'cntr_infeasible = %d' %cntr_infeasible, 'number of iterations = %d' %(milp_info_compressed.shape[0]))
print('elapsed time = %.2f' % elapsed_time)
print('date and time : ' + '%.2d%.2d_%.2d%.2d%.2d' %(x.month, x.day, x.hour, x.minute, x.second))
print('completed')

# minlp_info_compressed = np.load('data_minlp//data_minlp_N%.2d.npy' %N, allow_pickle=True)
# N_datapoints = minlp_info_compressed.shape[0]
# state_n, state_rho, state_depot, state_l, delta_minlp = decompress_minlp_info(minlp_info_compressed[j,:])