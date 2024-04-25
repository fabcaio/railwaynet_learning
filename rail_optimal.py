'''
this file generates the data for the training of the supervised learning approach

it solves the MINLP and saves the states (parameters) and the solution in a .npy
'''

import numpy as np

from rail_rl_env import RailNet, gurobi_minlp, qp_feasible

import time
import datetime

n_threads = 8
N_datapoints= 10000

data_sets = np.load('training_sets.npy', allow_pickle=True).item()
d_pre = data_sets['d_pre']
rho_whole = data_sets['rho_whole']
un = data_sets['un']
ul = data_sets['ul']
uy = data_sets['uy']
ua = data_sets['ua']
ud = data_sets['ud']
utau = data_sets['utau']
ur = data_sets['ur']
depot = data_sets['depot']

r_max = data_sets['r_max']
r_min = data_sets['r_min']
differ = data_sets['differ']
Cmax = data_sets['Cmax']
sigma = data_sets['sigma']
same = data_sets['same']
num_station = data_sets['num_station']
num_train = data_sets['num_train']
E_regular = data_sets['E_regular']

epsilon = 10 ** (-10)
Mt = 1000000
mt = -1000000
t_constant = 60
h_min = 120
tau_min = 30
l_min = 1
l_max = 4
eta = 10**(-3)

g = 3

control_trains = 8
N = control_trains
N_control = N-2
Env = RailNet(N)

solution_minlp = []

def compress_minlp_info(state_n, state_rho, state_depot, state_l, delta_minlp, mdl_Obj):
    
    minlp_info_compressed = np.concatenate((state_n, state_rho.flatten(), state_depot, state_l.flatten(), delta_minlp.flatten(), mdl_Obj.reshape(1,)))
    
    return minlp_info_compressed

minlp_info_compressed = np.zeros((N_datapoints, 427))

start_time = time.time()

cntr_feasible = 0

for i in range(N_datapoints):

    Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)

    state_n = Env.state_n
    state_rho = Env.state_rho
    state_depot = Env.state_depot
    state_l = Env.state_l

    a_minlp, d_minlp, r_minlp, l_minlp, y_minlp, delta_minlp, mdl_minlp = gurobi_minlp(control_trains, Env.d_pre_cut, Env.state_rho,
                            Env.state_a, Env.state_d, Env.state_r, Env.state_l, Env.state_y, Env.state_n, Env.state_depot,
                            num_station, differ, sigma, same, t_constant, h_min, l_min, l_max, r_min, r_max, tau_min,
                            E_regular, Cmax, eta, n_threads)
    
    if qp_feasible(mdl_minlp)==True:
        minlp_info_compressed[i,:] = compress_minlp_info(state_n, state_rho, state_depot, state_l, delta_minlp, np.array(mdl_minlp.ObjVal))
        cntr_feasible += 1
    
    elapsed_time = time.time()-start_time
    
    if i == 0:
        mdl_minlp.Params.LogToConsole = 1
        mdl_minlp.optimize()
    
    if i % 100 == 0:
        print('i=%d' %i, 'elapsed_time=%.2f' %elapsed_time, 'avg_solution_time=%.2f' %(elapsed_time/(i+1)))
    
x = datetime.datetime.now()
    
np.save('data_minlp//data_minlp_N%.2d_%.2d%.2d.npy' % (N, x.month, x.day), minlp_info_compressed, allow_pickle=True)

print('cntr_feasible = %d' % cntr_feasible)
print('elapsed time = %.2f' % elapsed_time)
print('completed')

# minlp_info_compressed = np.load('data_minlp//data_minlp_N%.2d.npy' %N, allow_pickle=True)
# N_datapoints = minlp_info_compressed.shape[0]
# state_n, state_rho, state_depot, state_l, delta_minlp = decompress_minlp_info(minlp_info_compressed[j,:])