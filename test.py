from gurobipy import Model, GRB, quicksum
import numpy as np
import time
# import basic_setting as base
import pandas as pd
import scipy.io
from rail_rl_env_backup import RailNet

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
print(E_regular)
control_trains = 20
N = int(control_trains)
epsilon = 10 ** (-10)
Mt = 1000000
mt = -1000000
t_constant = 60
h_min = 120
tau_min = 30
l_min = 1
l_max = 4
eta = 10**(-3)


Env = RailNet(N)
list_action = np.zeros(control_trains - 1)
for i in range(control_trains - 1):
    list_action[i] = 62
delta = Env.build_delta_vector(list_action)
Env.set_randState(d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot)
state_rho, d_pre_cut, state_a, state_d, state_r, state_l, state_y,state_n, state_depot, reward, terminated, truncated, info = Env.step(list_action, d_pre, rho_whole, r_max, r_min, differ, Cmax, sigma, same, num_station,num_train, E_regular)


'''
the maximum value and the minimum value of states
'''

state_rho_max = 8
state_rho_min = 0
d_pre_cut_max = 70000
d_pre_cut_min = 2000
state_a_max = 70000
state_a_min = 2000
state_d_max = 70000
state_d_min = 2000
state_r_max = 150
state_r_min = 50
state_l_max = 3
state_l_min = 1
state_y_max = 2
state_y_min = -2
state_n_max = 15000
state_n_min = 0
state_depot_max = 60
state_depot_min = 0