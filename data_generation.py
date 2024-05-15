import numpy as np
import pandas as pd
import random

"""
Basic setting and parameters for passenger demands in railway
"""
file_path = 'sectional_demands.csv'
demands_raw = pd.read_csv(file_path, sep=';')
num_station = int((len(demands_raw.columns)+1)/2)
time_slots = len(demands_raw)
section_demands = demands_raw.values
t = [0]
t_start = 6 # start from 5:00, the t_start rows before 5:00 (from 2:00 - 5:00) are ignored
group = 1000
hours = 20
demands_use = np.zeros([2*hours,2*num_station])
for s in range(num_station-1):
    demands_use[0,s] = section_demands[t_start,s+1]/1800
    demands_use[0,s+num_station] = section_demands[t_start,s+num_station]/1800
for i in range(1,2*hours):
    t = t + [i*30*60]
    for s in range(num_station - 1):
        demands_use[i, s] = section_demands[t_start+i, s + 1]/1800
        demands_use[i, s + num_station] = section_demands[t_start+i, s + num_station]/1800
num_train = 220
cut_time = 5400 # 4320 # time to start control
Cmax = 400
# Parameters
tau_regular = 60
h_regular = 240
t_roll = 300
sigma = np.zeros([2*num_station])
sigma[2*num_station-1] = 1
sigma[num_station] = 1
same = np.zeros([2*num_station, 2*num_station])
same[0, 2*num_station-1] = 1
same[2*num_station-1, 0] = 1
same[num_station-1, num_station] = 1
same[num_station, num_station-1] = 1

def int_pwc(break_px, break_py, low, up):
    """
    function for calculating total number of passengers
    """
    data_len = len(break_px)
    x = [-1000] + break_px + [10e8]
    y = [break_py[0]] + break_py + [break_py[data_len - 1]]
    length_x = len(x)
    for i in range(0, length_x):
        if x[i] - low <= 0:
            start = i
    x[start] = low
    for i in range(0, length_x):
        if x[i] - up < 0:
            finish = i + 1
    x[finish] = up
    output = 0
    for i in range(start, finish):
        output = output + y[i] * (x[i + 1] - x[i])
    return output

def int_pwc_t(break_px, break_py, low, up):
    """
    function for calculating total waiting time of passengers
    """
    data_len = len(break_px)
    x = [-1000] + break_px + [10e8]
    y = [break_py[0]] + break_py + [break_py[data_len - 1]]
    length_x = len(x)
    for i in range(0, length_x):
        if x[i] - low <= 0:
            start = i
    x[start] = low
    for i in range(0, length_x):
        if x[i] - up < 0:
            finish = i + 1
    x[finish] = up
    output = 0
    for i in range(start, finish):
        output = output + 0.5 * y[i] * (x[i + 1] - x[i]) ** 2 + y[i] * (x[i + 1] - x[i]) * (x[finish] - x[i + 1])
    return output
"""
calculating parameters (running distance, running time, speed, energy consumption ect.) for train running along the line 
"""
sec_len = np.array([
    [1334, 1281, 2055, 2301, 2337, 1355, 1090, 1728, 993, 1982, 2366, 1275, 2631, 1000],
    [2631, 1275, 2366, 1982, 993, 1728, 1090, 1355, 2337, 2301, 2055, 1281, 1334, 1000]
])
v_cru = 68/3.6
a_acc = 0.75
a_coa = -0.12
a_bra = -0.7
gra = 0.1
c1 = 0.0078
c2 = 0.00085
c3 = 0.000076
r_regular = np.zeros([2*num_station])
r_min = np.zeros([2*num_station])
r_max = np.zeros([2*num_station])
s_acc = np.zeros([2*num_station])
t_acc = np.zeros([2*num_station])
s_cru = np.zeros([2*num_station])
t_cru = np.zeros([2*num_station])
s_bra = np.zeros([2*num_station])
t_bra = np.zeros([2*num_station])
E_acc = np.zeros([2*num_station])
E_cru = np.zeros([2*num_station])
E_regular = np.zeros([2*num_station])
for i in range(2):
    for s in range(num_station):
        s_acc[s + i * num_station] = v_cru**2/(2*a_acc)
        t_acc[s + i * num_station] = v_cru/a_acc
        s_bra[s + i * num_station] = -v_cru**2/(2*a_bra)
        t_bra[s + i * num_station] = -v_cru/a_bra
        s_cru[s + i * num_station] = sec_len[i, s] - s_acc[s + i * num_station] - s_bra[s + i * num_station]
        t_cru[s + i * num_station] = s_cru[s + i * num_station]/v_cru
        r_regular[s + i * num_station] = s_cru[s + i * num_station]/v_cru + v_cru/(2*a_acc) - v_cru/(2*a_bra)
        E_acc[s + i * num_station] = 0.5*(a_acc*a_acc+c1*a_acc+gra*a_acc)*t_acc[s + i * num_station]**2+1/3*c2*a_acc**2*t_acc[s + i * num_station]**3+1/4*c3*a_acc**3*t_acc[s + i * num_station]**4;
        E_cru[s + i * num_station] = (c1+c2*v_cru+c3*(v_cru**2)+gra)*v_cru*t_cru[s + i * num_station]
        E_regular[s + i * num_station] = 0.1 * (E_acc[s + i * num_station] + E_cru[s + i * num_station])
r_max = 1.2*r_regular
r_min = 0.8*r_regular
E_regular[num_station-1] = 0
E_regular[2*num_station-1] = 0
"""
calculating the original timetable
"""
d_pre = np.zeros([num_train,2*num_station]) # d_pre[k,s]
a_pre = np.zeros([num_train,2*num_station])
tau_pre = np.zeros([num_train,2*num_station])
r_pre = np.zeros([num_train,2*num_station])
h_pre = np.zeros([num_train,2*num_station])
for k in range(num_train):
    for s in range(2*num_station):
        tau_pre[k, s] = tau_regular
        r_pre[k,s] = r_regular[s]
        h_pre[k,s] = h_regular
a_pre[0, 0]=0
# train: train 0 to train num_train-2
for k in range(num_train-1):
# station
    for s in range(2*num_station-1):
        d_pre[k, s] = a_pre[k, s] + tau_pre[k,s]
        a_pre[k, s+1] = d_pre[k, s] + r_pre[k,s]
    d_pre[k, 2*num_station-1] = a_pre[k, 2*num_station-1] + tau_pre[k, 2*num_station-1]
    a_pre[k+1, 0] = d_pre[k, 0] + h_pre[k,0]
# the last train: num_train-1
for s in range(2*num_station-1):
    d_pre[num_train-1, s] = a_pre[num_train-1, s] + tau_pre[num_train-1, s]
    a_pre[num_train-1, s + 1] = d_pre[num_train-1, s] + r_pre[num_train-1, s]
d_pre[num_train-1, 2 * num_station - 1] = a_pre[num_train-1, 2 * num_station - 1] + tau_pre[num_train-1, 2 * num_station - 1]

start_train = np.zeros([2*num_station])
a_start = np.zeros([2*num_station])
d_start = np.zeros([2*num_station])
tau_start = np.zeros([2*num_station])
r_start = np.zeros([2*num_station])
for k in range(num_train):
    for s in range(2*num_station):
        if d_pre[k,s] <= cut_time:
            start_train[s] = k
            a_start[s] = a_pre[k,s]
            d_start[s] = d_pre[k, s]
            tau_start[s] = tau_pre[k, s]
            r_start[s] = r_pre[k, s]
differ = np.zeros([2*num_station])
for s in range(1,2*num_station):
    differ[s] = round((a_start[s-1]+tau_start[s-1]+r_start[s-1]-a_start[s])/(h_regular+tau_regular))

rho_expect = np.zeros([num_train,2*num_station])
for s in range(2*num_station-1):
    rho_expect[0,s] = int_pwc(t,demands_use[:,s],0, d_pre[0,s])
    for k in range(1,num_train):
        rho_expect[k,s]=int_pwc(t, demands_use[:,s], d_pre[k-1,s], d_pre[k,s])

rho_whole = np.zeros([num_train,2*num_station,group])
for g in range(group):
    for s in range(2*num_station-1):
        # Generate random demand based on Poisson distribution with mean as the expected value from demands_raw
        random_demand = np.random.poisson(rho_expect[0, s]) / (d_pre[0,s]-0)
        # Ensure demand is non-negative
        random_demand = max(0, random_demand)
        # Assign generated demand to section_demands
        rho_whole[0,s,g] = random_demand
        for k in range(1,num_train):
            random_demand = np.random.poisson(rho_expect[k, s]) / (d_pre[k,s]-d_pre[k-1,s])
            random_demand = max(0, random_demand)
            rho_whole[k,s,g] = random_demand
"""
calculating variables corresponding to the number of passengers and train availability for the original timetable 
"""
ua = np.zeros([num_train,2*num_station,group])
ud = np.zeros([num_train,2*num_station,group])
ur = np.zeros([num_train,2*num_station,group])
utau = np.zeros([num_train,2*num_station,group])
ul = np.zeros([num_train,2*num_station,group])
uy = np.zeros([num_train,2*num_station,group])

N_depot = np.zeros([num_station,group])
depot = np.zeros([num_train,num_station,group])
for g in range(group):
    N_depot[0, g] = random.randint(35, 45)
    depot[0, 0, g] = N_depot[0, g]
    N_depot[num_station - 1, g] = random.randint(5, 10)
    depot[0, num_station - 1, g] = N_depot[num_station - 1, g]
    for s in range(2*num_station):
        for k in range(num_train):
            ua[k, s, g] = a_pre[k, s]
            utau[k, s, g] = tau_pre[k, s]
            ud[k, s, g] = d_pre[k, s]
            ur[k, s, g] = r_pre[k, s]

for g in range(group):
    for k in range(num_train-1):
        if d_pre[k,0] < d_pre[1,2 * num_station - 1] + t_roll:
            # uy[k, 0, g] = random.choice([1, 2, 3, 4, 5, 6])
            uy[k, 0, g] = 2
            ul[k, 0, g] = uy[k, 0, g]
            depot[k+1,0,g] = depot[k,0,g] - uy[k, 0, g]
            # depot[k+1,0,g] = N_depot[0,g] - sum(uy[j, 0, g] j in range(k+1))
            for s in range(1, 2 * num_station - 1):
                uy[k, s, g] = 0
                ul[k, s, g] = round(ul[k, s - 1, g] + uy[k, s, g])
            uy[k, 2 * num_station - 1, g] = round(-ul[k, 2 * num_station - 2, g])
            ul[k, 2 * num_station - 1, g] = round(ul[k, 2 * num_station - 2, g] + uy[k, 2 * num_station - 1, g])
            depot[k + 1, num_station-1, g] = depot[k, num_station-1, g] - (ul[k, num_station, g] - ul[k, num_station-1, g])
        else:
            for i in range(1,num_train-1):
                if (d_pre[k,0] >= d_pre[i,2 * num_station - 1] + t_roll)&(d_pre[k,0] < d_pre[i+1,2 * num_station - 1] + t_roll):
                    temp1 = random.choice([1, 2, 3, 4])
                    temp2 = depot[k,0,g] - uy[i-1, 2 * num_station - 1, g]
                    uy[k, 0, g] = max(min(temp1,temp2),1)
                    ul[k, 0, g] = uy[k, 0, g]
                    depot[k + 1, 0, g] = depot[k, 0, g] - uy[k, 0, g] - uy[i-1, 2 * num_station - 1, g]
                    for s in range(1, 2 * num_station - 1):
                        uy[k, s, g] = 0
                        ul[k, s, g] = round(ul[k, s - 1, g] + uy[k, s, g])
                    uy[k, 2 * num_station - 1, g] = round(-ul[k, 2 * num_station - 2, g])
                    ul[k, 2 * num_station - 1, g] = round(ul[k, 2 * num_station - 2, g] + uy[k, 2 * num_station - 1, g])
                    depot[k + 1, num_station - 1, g] = depot[k, num_station - 1, g] - (ul[k, num_station, g] - ul[k, num_station-1, g])

un = np.zeros([num_train,2*num_station,group])
un_depart = np.zeros([num_train,2*num_station,group])
un_before = np.zeros([num_train,2*num_station,group])
uC = np.zeros([num_train,2*num_station,group])
un_after = np.zeros([num_train,2*num_station,group])
for g in range(group):
    for s in range(2*num_station):
        un[0,s,g] = rho_whole[0,s,g]*(d_pre[0,s] - 0)
    for s in range(2*num_station):
        for k in range(num_train - 1):
            un_before[k,s,g] = un[k,s,g] + rho_whole[k+1,s,g]*(ud[k,s,g] - d_pre[k,s])
            uC[k,s,g] = ul[k,s,g] * Cmax
            un_depart[k,s,g] = min(uC[k,s,g], un_before[k,s,g])
            un_after[k,s,g] = un_before[k,s,g] - un_depart[k,s,g]
            un[k+1,s,g] = un[k,s,g] + rho_whole[k+1,s,g]*(d_pre[k+1,s] - d_pre[k,s]) - un_depart[k,s,g]

training_sets = {'d_pre': d_pre, 'rho_whole': rho_whole, 'un': un, 'ul': ul, 'uy': uy, 'ua': ua, 'ud': ud, 'utau': utau, 'ur': ur,
                 'depot': depot, 'r_max': r_max, 'r_min': r_min, 'differ': differ, 'Cmax': Cmax, 'sigma': sigma, 'same': same,
                 'num_station': num_station, 'num_train': num_train, 'E_regular': E_regular}

np.save('training_sets.npy', training_sets)
print(depot[15:25,0,1:6])
print(d_pre)
print(E_regular)
