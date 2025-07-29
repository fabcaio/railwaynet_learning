from rail_fun import cost_per_step
import numpy as np
from rail_rl_env import RailNet
from rail_rl_env import gurobi_minlp, gurobi_milp, gurobi_nlp_presolve, gurobi_lp_presolve
from rail_rl_env import time_from_best, epsilon_to_compare_gap
import time
from rail_rl_env import d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot
import sys

# N=40
# job_idx = 999
# timeout = 5*60 #minutes
# n_threads = 1
# timelimit=20

N = int(sys.argv[1])
job_idx = int(sys.argv[2])
timelimit_job = int(sys.argv[3])
n_threads = int(sys.argv[4])
timeout = (timelimit_job-1.5)*60*60
timelimit = 240

Env = RailNet(N)

mipgap = 1e-3
log = 0

print('mipgap: %.4f \t timelimit (gurobi): %.2f \t n_threads: %d' %(mipgap, timelimit, n_threads))
print('callback settings. \t time %.4f \t epsilon %.4f' %(time_from_best, epsilon_to_compare_gap))

list_minlp_nostop = []
list_minlp_nostop_ws = []

list_minlp = []
list_minlp_nlp = []
list_minlp_nlp_ws = []
list_minlp_lp = []

list_minlp_ws = []
list_minlp_ws_nlp = []
list_minlp_ws_nlp_ws = []
list_minlp_ws_lp = []

list_milp = []
list_milp_nlp = []
list_milp_nlp_ws = []
list_milp_lp = []

start_loop = time.time()

while time.time() < start_loop + timeout :
        
    Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)
    
    # minlp (no early_term, no warm_start)
    start_time=time.time()
    early_term=0
    warm_start=0
    a_values, d_values, r_values, l_values, y_values, _, n_values, n_after_values, mdl = gurobi_minlp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit, early_term, warm_start, n_threads)
    cost_minlp = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    list_minlp_nostop.append([mdl.Runtime, cost_minlp, 0, mdl.status])
    print('minlp_net \t %.2f \t\t %.2f \t\t 0%% \t %d' %(mdl.Runtime, cost_minlp, mdl.status))
    
    # minlp (no early_term, warm_start)
    start_time=time.time()
    early_term=0
    warm_start=1
    a_values, d_values, r_values, l_values, y_values, _, n_values, n_after_values, mdl = gurobi_minlp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit, early_term, warm_start, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_minlp_nostop_ws.append([mdl._Runtime, cost, perc_minlp_cost, mdl.status])
    print('minlp_net_ws \t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(mdl._Runtime, cost, perc_minlp_cost, mdl.status))
    
    # minlp (no warm_start)
    start_time=time.time()
    early_term=1
    warm_start=0
    a_values, d_values, r_values, l_values, y_values, delta_minlp, n_values, n_after_values, mdl = gurobi_minlp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit, early_term, warm_start, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_minlp.append([mdl.Runtime, cost, perc_minlp_cost, mdl.status])
    print('minlp \t\t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(mdl.Runtime, cost, perc_minlp_cost, mdl.status))
    
    #minlp(no warm_start) + nlp (no warm_start)
    start_time=time.time()
    early_term=1
    warm_start=0
    a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl = gurobi_nlp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot,delta_minlp, mipgap, log, timelimit, early_term, warm_start, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_minlp_nlp.append([mdl.Runtime, cost, perc_minlp_cost, mdl.status])
    print('minlp_nlp \t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(mdl.Runtime, cost, perc_minlp_cost, mdl.status))
    
    #minlp(no warm_start) + nlp
    start_time=time.time()
    early_term=1
    warm_start=1
    a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl = gurobi_nlp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot,delta_minlp, mipgap, log, timelimit, early_term, warm_start, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_minlp_nlp_ws.append([mdl._Runtime, cost, perc_minlp_cost, mdl.status])
    print('minlp_nlp_ws \t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(mdl._Runtime, cost, perc_minlp_cost, mdl.status))
    
    #minlp(early_term, no warm_start) + lp
    start_time=time.time()
    a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl = gurobi_lp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot,delta_minlp, mipgap, log, timelimit, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_minlp_lp.append([mdl.Runtime, cost, perc_minlp_cost, mdl.status])
    print('minlp_lp \t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(mdl.Runtime, cost, perc_minlp_cost, mdl.status))
    
    # minlp
    start_time=time.time()
    early_term=1
    warm_start=1
    a_values, d_values, r_values, l_values, y_values, delta_minlp_ws, n_values, n_after_values, mdl = gurobi_minlp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit, early_term, warm_start, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_minlp_ws.append([mdl._Runtime, cost, perc_minlp_cost, mdl.status])
    print('minlp_ws \t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(mdl._Runtime, cost, perc_minlp_cost, mdl.status))
    
    #minlp + nlp (no warm_start)
    start_time=time.time()
    early_term=1
    warm_start=0
    a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl = gurobi_nlp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot,delta_minlp_ws, mipgap, log, timelimit, early_term, warm_start, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_minlp_ws_nlp.append([mdl.Runtime, cost, perc_minlp_cost, mdl.status])
    print('minlp_ws_nlp \t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(mdl.Runtime, cost, perc_minlp_cost, mdl.status))
    
    #minlp + nlp
    start_time=time.time()
    early_term=1
    warm_start=1
    a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl = gurobi_nlp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot,delta_minlp_ws, mipgap, log, timelimit, early_term, warm_start, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_minlp_ws_nlp_ws.append([mdl._Runtime, cost, perc_minlp_cost, mdl.status])
    print('minlp_ws_nlp_ws \t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(mdl._Runtime, cost, perc_minlp_cost, mdl.status))
    
    #minlp(early_term, no warm_start) + lp
    start_time=time.time()
    a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl = gurobi_lp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot,delta_minlp_ws, mipgap, log, timelimit, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_minlp_ws_lp.append([mdl.Runtime, cost, perc_minlp_cost, mdl.status])
    print('minlp_ws_lp \t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(mdl.Runtime, cost, perc_minlp_cost, mdl.status))
    
    #milp
    start_time=time.time()
    a_values, d_values, r_values, l_values, y_values, delta_milp, n_values, n_after_values, mdl = gurobi_milp(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot, mipgap, log, timelimit, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_milp.append([mdl.Runtime, cost, perc_minlp_cost, mdl.status])
    print('milp \t \t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(mdl.Runtime, cost, perc_minlp_cost, mdl.status))
    
    #milp + nlp(no warm_start)
    start_time=time.time()
    early_term=1
    warm_start=0
    a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl = gurobi_nlp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot,delta_milp, mipgap, log, timelimit, early_term, warm_start, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_milp_nlp.append([mdl.Runtime, cost, perc_minlp_cost, mdl.status])
    print('milp_nlp \t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(mdl.Runtime, cost, perc_minlp_cost, mdl.status))
    
    #milp + nlp
    start_time=time.time()
    early_term=1
    warm_start=1
    a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl = gurobi_nlp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot,delta_milp, mipgap, log, timelimit, early_term, warm_start, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_milp_nlp_ws.append([mdl._Runtime, cost, perc_minlp_cost, mdl.status])
    print('milp_nlp_ws \t %.2f \t\t %.2f \t\t %.2f%% \t %d' %(mdl.Runtime, cost, perc_minlp_cost, mdl.status))
    
    #milp + lp
    start_time=time.time()    
    a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl = gurobi_lp_presolve(N, Env.d_pre_cut, Env.state_rho, Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot,delta_milp, mipgap, log, timelimit, n_threads)
    cost = cost_per_step(n_values, n_after_values, Env.d_pre_cut_old, d_values, l_values, N)
    perc_minlp_cost = (cost-cost_minlp)/cost_minlp*100
    list_milp_lp.append([mdl.Runtime, cost, perc_minlp_cost, mdl.status])
    print('milp_lp \t %.2f \t\t %.2f \t\t %.2f%% \t %d \n' %(mdl.Runtime, cost, perc_minlp_cost, mdl.status))
    
array_minlp_nostop = np.array(list_minlp_nostop)
array_minlp_nostop_ws = np.array(list_minlp_nostop_ws)

array_minlp = np.array(list_minlp)
array_minlp_nlp = np.array(list_minlp_nlp)
array_minlp_nlp_ws = np.array(list_minlp_nlp_ws)
array_minlp_lp = np.array(list_minlp_lp)

array_minlp_ws = np.array(list_minlp_ws)
array_minlp_ws_nlp = np.array(list_minlp_ws_nlp)
array_minlp_ws_nlp_ws = np.array(list_minlp_ws_nlp_ws)
array_minlp_ws_lp = np.array(list_minlp_ws_lp)

array_milp = np.array(list_milp)
array_milp_nlp = np.array(list_milp_nlp)
array_milp_nlp_ws = np.array(list_milp_nlp_ws)
array_milp_lp = np.array(list_milp_lp)

dict_arrays = {
    'array_minlp_nostop': array_minlp_nostop,
    'array_minlp_nostop_ws': array_minlp_nostop_ws,
    'array_minlp': array_minlp,
    'array_minlp_nlp': array_minlp_nlp,
    'array_minlp_nlp_ws': array_minlp_nlp_ws,
    'array_minlp_lp': array_minlp_lp,
    'array_minlp_ws': array_minlp_ws,
    'array_minlp_ws_nlp': array_minlp_ws_nlp,
    'array_minlp_ws_nlp_ws': array_minlp_ws_nlp_ws,
    'array_minlp_ws_lp': array_minlp_ws_lp,
    'array_milp': array_milp,
    'array_milp_nlp': array_milp_nlp,
    'array_milp_nlp_ws': array_milp_nlp_ws,
    'array_milp_lp': array_milp_lp,
    'info': 'Each array has a list containing [mdl.Runtime, cost_minlp, perc_minlp_cost, mdl.status]'
}

np.save('tests//tests_minlp_milp_N%d_%.3d.npy' %(N, job_idx), dict_arrays)
