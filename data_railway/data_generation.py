import numpy as np
import pandas as pd
import random

"""
Basic setting and parameters for passenger demands in railway
"""
file_path_LC = 'data_railway//sectional_demands_LC.csv'
demands_raw_LC = pd.read_csv(file_path_LC, sep=';')
section_demands_LC = demands_raw_LC.values

file_path_L13 = 'data_railway//sectional_demands_L13.csv'
demands_raw_L13 = pd.read_csv(file_path_L13, sep=';')
section_demands_L13 = demands_raw_L13.values

file_path_L8 = 'data_railway//sectional_demands_L8.csv'
demands_raw_L8 = pd.read_csv(file_path_L8, sep=';')
section_demands_L8 = demands_raw_L8.values

num_line = 3
num_station = np.zeros(3, dtype=int)
num_station[0] = round((len(demands_raw_LC.columns)+1)/2)
num_station[1] = round((len(demands_raw_L13.columns)+1)/2)
num_station[2] = round((len(demands_raw_L8.columns)+1)/2)
max_station = round(max(num_station))
print(max_station)

time_slots = len(demands_raw_LC)
t = [0]
t_start = 4 # 6 start from 5:00, the t_start rows before 5:00 (from 2:00 - 5:00) are ignored
group = 1000
hours = 20
demands_use = np.zeros([num_line,2*hours,2*max_station])
# demands for Line Changping
for s in range(num_station[0]-1):
    demands_use[0,0,s] = section_demands_LC[t_start,s+1]/1800
    demands_use[0,0,s+num_station[0]] = section_demands_LC[t_start,s+num_station[0]]/1800
for i in range(1,2*hours):
    t = t + [i*30*60]
    for s in range(num_station[0] - 1):
        demands_use[0, i, s] = section_demands_LC[t_start+i, s + 1]/1800
        demands_use[0, i, s + num_station[0]] = section_demands_LC[t_start+i, s + num_station[0]]/1800
# demands for Line 13
for s in range(num_station[1]-1):
    demands_use[1,0,s] = section_demands_L13[t_start,s+1]/1800
    demands_use[1,0,s+num_station[1]] = section_demands_L13[t_start,s+num_station[1]]/1800
for i in range(1,2*hours):
    for s in range(num_station[1] - 1):
        demands_use[1, i, s] = section_demands_L13[t_start+i, s + 1]/1800
        demands_use[1, i, s + num_station[1]] = section_demands_L13[t_start+i, s + num_station[1]]/1800
# demands for Line 8
for s in range(num_station[2]-1):
    demands_use[2,0,s] = section_demands_L8[t_start,s+1]/1800
    demands_use[2,0,s+num_station[2]] = section_demands_L8[t_start,s+num_station[2]]/1800
for i in range(1,2*hours):
    for s in range(num_station[2] - 1):
        demands_use[2, i, s] = section_demands_L8[t_start+i, s + 1]/1800
        demands_use[2, i, s + num_station[2]] = section_demands_L8[t_start+i, s + num_station[2]]/1800
num_train = 260
cut_time = 7200 # 4320 # time to start control
Cmax = 400
# Parameters
tau_regular = 60
h_regular = 180
t_roll = 240
t_trans = 60
trans_rate = 10/100
sigma = np.zeros([num_line, 2*max_station])
# same = np.zeros([num_line, 2*max_station, 2*max_station])
for m in range(num_line):
    sigma[m, 2*num_station[m]-1] = 1
    # sigma[m, num_station[m]] = 1
    # same[m, 0, 2*num_station[m]-1] = 1
    # same[m, 2*num_station[m]-1, 0] = 1
    # same[m, num_station[m]-1, num_station[m]] = 1
    # same[m, num_station[m], num_station[m]-1] = 1
# sigma[1, 9] = 1
# sigma[1, 14] = 1
# same[1, 9, 14] = 1
# same[1, 14, 9] = 1
# sigma[2, 10] = 1
# sigma[2, 23] = 1
# same[2, 10, 23] = 1
# same[2, 23, 10] = 1
# sigma[3, num_station[3]] = 1
olin = np.zeros([num_line, 2*max_station], dtype=int)
opla = np.zeros([num_line, 2*max_station,2], dtype=int)
olin[0,9] = 2
opla[0,9,0] = 0
opla[0,9,1] = 37
olin[0,14] = 2
opla[0,14,0] = 0
opla[0,14,1] = 37
olin[0,12] = 1
opla[0,12,0] = 6
opla[0,12,1] = 27
olin[1,6] = 0
opla[1,6,0] = 11
olin[1,27] = 0
opla[1,27,0] = 11
olin[1,9] = 2
opla[1,9,0] = 4
opla[1,9,1] = 33
olin[1,24] = 2
opla[1,24,0] = 4
opla[1,24,1] = 33
olin[2,0] = 0
opla[2,0,0] = 9
opla[2,0,1] = 14
olin[2,4] = 1
opla[2,4,0] = 9
opla[2,4,1] = 24
olin[2,33] = 1
opla[2,33,0] = 9
opla[2,33,1] = 24


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
    [[1213,3508,2433,1683,1958,5357,1964,2025,3799,2367,5440, 1000, 0, 0, 0, 0, 0, 0, 0],
    [5440,2367,3799,2025,1964,5357,1958,1683,2433,3508,1213, 1000, 0, 0, 0, 0, 0, 0, 0]],
    [[2839,1206,1829,4866,1036,1545,3623,1423,2110,4785,2272,6720,2152,1110,1135,1769,1000, 0, 0],
     [1769,1135,1110,2152,6720,2272,4785,2110,1423,3623,1545,1036,4866,1829,1206,2839,1000, 0, 0]],
    [[2319,1986,2017,1115,1895,1543,1042,2553,2558,1017,1667,900,1009,1275,1084,1188,902,1437,1000],
     [1437,902,1188,1084,1275,1009,900,1667,1017,2558,2553,1042,1543,1895,1115,2017,1986,2319,1000]]
])
v_cru = 70/3.6
a_acc = 0.75
a_coa = -0.12
a_bra = -0.7
gra = 0.1
c1 = 0.0078
c2 = 0.00085
c3 = 0.000076
r_regular = np.zeros([num_line,2*max_station])
r_min = np.zeros([num_line,2*max_station])
r_max = np.zeros([num_line,2*max_station])
s_acc = np.zeros([num_line,2*max_station])
t_acc = np.zeros([num_line,2*max_station])
s_cru = np.zeros([num_line,2*max_station])
t_cru = np.zeros([num_line,2*max_station])
s_bra = np.zeros([num_line,2*max_station])
t_bra = np.zeros([num_line,2*max_station])
E_acc = np.zeros([num_line,2*max_station])
E_cru = np.zeros([num_line,2*max_station])
E_regular = np.zeros([num_line,2*max_station])
for m in range(num_line):
    for i in range(2):
        for s in range(num_station[m]):
            s_acc[m, s + i * num_station[m]] = v_cru**2/(2*a_acc)
            t_acc[m, s + i * num_station[m]] = v_cru/a_acc
            s_bra[m, s + i * num_station[m]] = -v_cru**2/(2*a_bra)
            t_bra[m, s + i * num_station[m]] = -v_cru/a_bra
            s_cru[m, s + i * num_station[m]] = sec_len[m, i, s] - s_acc[m, s + i * num_station[m]] - s_bra[m, s + i * num_station[m]]
            t_cru[m, s + i * num_station[m]] = s_cru[m, s + i * num_station[m]]/v_cru
            r_regular[m, s + i * num_station[m]] = s_cru[m, s + i * num_station[m]]/v_cru + v_cru/(2*a_acc) - v_cru/(2*a_bra)
            E_acc[m, s + i * num_station[m]] = 0.5*(a_acc*a_acc+c1*a_acc+gra*a_acc)*t_acc[m, s + i * num_station[m]]**2+1/3*c2*a_acc**2*t_acc[m, s + i * num_station[m]]**3+1/4*c3*a_acc**3*t_acc[m, s + i * num_station[m]]**4;
            E_cru[m, s + i * num_station[m]] = (c1+c2*v_cru+c3*(v_cru**2)+gra)*v_cru*t_cru[m, s + i * num_station[m]]
            E_regular[m, s + i * num_station[m]] = 0.1 * (E_acc[m, s + i * num_station[m]] + E_cru[m, s + i * num_station[m]])
r_max = 1.2*r_regular
r_min = 0.8*r_regular
for m in range(num_line):
    E_regular[m, num_station[m]-1] = 0
    E_regular[m, 2*num_station[m]-1] = 0
"""
calculating the original timetable
"""
d_pre = np.zeros([num_line,num_train,2*max_station]) # d_pre[k,s]
a_pre = np.zeros([num_line,num_train,2*max_station])
tau_pre = np.zeros([num_line,num_train,2*max_station])
r_pre = np.zeros([num_line,num_train,2*max_station])
h_pre = np.zeros([num_line,num_train,2*max_station])
for m in range(num_line):
    for k in range(num_train):
        for s in range(2*num_station[m]):
            tau_pre[m, k, s] = tau_regular
            r_pre[m,k,s] = r_regular[m,s]
            h_pre[m,k,s] = h_regular
    a_pre[m, 0, 0]= h_regular
# train: train 0 to train num_train-2
for m in range(num_line):
    for k in range(num_train-1):
    # station
        for s in range(2*num_station[m]-1):
            d_pre[m, k, s] = a_pre[m, k, s] + tau_pre[m, k, s]
            a_pre[m, k, s+1] = d_pre[m, k, s] + r_pre[m, k, s]
        d_pre[m, k, 2*num_station[m]-1] = a_pre[m, k, 2*num_station[m]-1] + tau_pre[m, k, 2*num_station[m]-1]
        a_pre[m, k+1, 0] = d_pre[m, k, 0] + h_pre[m, k, 0]
# the last train: num_train-1
    for s in range(2*num_station[m]-1):
        d_pre[m,num_train-1, s] = a_pre[m,num_train-1, s] + tau_pre[m,num_train-1, s]
        a_pre[m,num_train-1, s + 1] = d_pre[m,num_train-1, s] + r_pre[m,num_train-1, s]
    d_pre[m,num_train-1, 2 * num_station[m] - 1] = a_pre[m,num_train-1, 2 * num_station[m] - 1] + tau_pre[m,num_train-1, 2 * num_station[m] - 1]

otra = np.zeros([num_line,num_train, 2*max_station,2], dtype=int)
for m in range(num_line):
    for s in range(2*num_station[m]):
        for k in range(1,num_train):
            for j in range(2):
                for i in range(num_train-2):#d_pre = a_pre + t_trans if we set t_trans = tau
                    if (d_pre[olin[m,s], i+1, opla[m,s,j]] <= d_pre[m,k,s]) and (d_pre[olin[m,s], i+1, opla[m,s,j]] > d_pre[m,k-1,s]):
                        otra[m, k, s, j] = i

start_train = np.zeros([num_line,2*max_station])
a_start = np.zeros([num_line,2*max_station])
d_start = np.zeros([num_line,2*max_station])
tau_start = np.zeros([num_line,2*max_station])
r_start = np.zeros([num_line,2*max_station])
for m in range(num_line):
    for k in range(num_train):
        for s in range(2*num_station[m]):
            if d_pre[m, k, s] <= cut_time:
                start_train[m, s] = k
                a_start[m, s] = a_pre[m, k, s]
                d_start[m, s] = d_pre[m, k, s]
                tau_start[m, s] = tau_pre[m, k, s]
                r_start[m, s] = r_pre[m, k, s]
differ = np.zeros([num_line,2*max_station], dtype=int)
for m in range(num_line):
    for s in range(1,2*num_station[m]):
        differ[m,s] = round((d_start[m,s-1]+r_start[m,s-1]+tau_start[m,s]-d_start[m,s])/(h_regular+tau_regular))
print(differ)

rho_expect = np.zeros([num_line,num_train,2*max_station])
for m in range(num_line):
    for s in range(2*num_station[m]-1):
        rho_expect[m,0,s] = int_pwc(t,demands_use[m,:,s],0, d_pre[m,0,s])
        for k in range(1,num_train):
            rho_expect[m,k,s]=int_pwc(t, demands_use[m,:,s], d_pre[m,k-1,s], d_pre[m,k,s])

rho_whole = np.zeros([num_line,num_train,2*max_station,group])
for g in range(group):
    for m in range(num_line):
        for s in range(2*num_station[m]-1):
            # Generate random demand based on Poisson distribution with mean as the expected value from demands_raw
            random_demand = np.random.poisson(rho_expect[m, 0, s]) / (d_pre[m,0,s]-0)
            # Ensure demand is non-negative
            random_demand = max(0, random_demand)
            # Assign generated demand to section_demands
            rho_whole[m,0,s,g] = random_demand
            for k in range(1,num_train):
                random_demand = np.random.poisson(rho_expect[m, k, s]) / (d_pre[m,k,s]-d_pre[m,k-1,s])
                random_demand = max(0, random_demand)
                rho_whole[m,k,s,g] = random_demand
"""
calculating variables corresponding to the number of passengers and train availability for the original timetable 
"""
ua = np.zeros([num_line,num_train,2*max_station,group])
ud = np.zeros([num_line,num_train,2*max_station,group])
ur = np.zeros([num_line,num_train,2*max_station,group])
utau = np.zeros([num_line,num_train,2*max_station,group])
ul = np.zeros([num_line,num_train,2*max_station,group])
uy = np.zeros([num_line,num_train,2*max_station,group])

N_depot = np.zeros([num_line,group])
depot = np.zeros([num_line,num_train,group])
for g in range(group):
    N_depot[0, g] = random.randint(65, 85) # lower bound 46
    N_depot[1, g] = random.randint(80, 100) # lower bound 58
    N_depot[2, g] = random.randint(70, 90) # lower bound 50
    # N_depot[0, num_station[m] - 1, g] = random.randint(10, 15)
    # N_depot[1, num_station[m] - 1, g] = random.randint(15, 20)
    # N_depot[2, num_station[m] - 1, g] = random.randint(10, 15)
    for m in range(num_line):
        depot[m, 0, g] = N_depot[m, g]
        # depot[m, 0, num_station[m] - 1, g] = N_depot[m, num_station[m] - 1, g]
        for s in range(2*num_station[m]):
            for k in range(num_train):
                ua[m, k, s, g] = a_pre[m, k, s]
                utau[m, k, s, g] = tau_pre[m, k, s]
                ud[m, k, s, g] = d_pre[m, k, s]
                ur[m, k, s, g] = r_pre[m, k, s]
on_loop_trains = np.zeros(num_line)
for g in range(group):
    for m in range(num_line):
        for k in range(num_train-1):
            if d_pre[m,k,0] < d_pre[m,2,2 * num_station[m] - 1] + t_roll:
                # uy[k, 0, g] = random.choice([1, 2, 3, 4, 5, 6])
                on_loop_trains[m] = k
                uy[m, k, 0, g] = 2
                ul[m, k, 0, g] = uy[m, k, 0, g]
                depot[m,k+1,g] = depot[m,k,g] - uy[m,k, 0, g]
                # depot[k+1,0,g] = N_depot[0,g] - sum(uy[j, 0, g] j in range(k+1))
                for s in range(1, 2 * num_station[m] - 1):
                    uy[m, k, s, g] = 0
                    ul[m, k, s, g] = round(ul[m, k, s - 1, g] + uy[m, k, s, g])
                uy[m, k, 2 * num_station[m] - 1, g] = round(-ul[m, k, 2 * num_station[m] - 2, g])
                ul[m, k, 2 * num_station[m] - 1, g] = round(ul[m, k, 2 * num_station[m] - 2, g] + uy[m, k, 2 * num_station[m] - 1, g])
                # depot[m, k + 1, num_station[m]-1, g] = depot[m, k, num_station[m]-1, g] - (ul[m, k, num_station[m], g] - ul[m, k, num_station[m]-1, g])
            else:
                for i in range(2,num_train-1):
                    if (d_pre[m,k,0] >= d_pre[m,i,2 * num_station[m] - 1] + t_roll)&(d_pre[m,k,0] < d_pre[m,i+1,2 * num_station[m] - 1] + t_roll):
                        temp1 = 2 #random.choice([1, 2, 3, 4])
                        temp2 = depot[m,k,g] - uy[m,i-1, 2 * num_station[m] - 1, g]
                        uy[m, k, 0, g] = max(min(temp1,temp2),1)
                        ul[m, k, 0, g] = uy[m, k, 0, g]
                        depot[m, k + 1, g] = depot[m, k, g] - uy[m, k, 0, g] - uy[m, i-2, 2 * num_station[m] - 1, g]
                        for s in range(1, 2 * num_station[m] - 1):
                            uy[m, k, s, g] = 0
                            ul[m, k, s, g] = round(ul[m, k, s - 1, g] + uy[m, k, s, g])
                        uy[m, k, 2 * num_station[m] - 1, g] = round(-ul[m, k, 2 * num_station[m] - 2, g])
                        ul[m, k, 2 * num_station[m] - 1, g] = round(ul[m, k, 2 * num_station[m] - 2, g] + uy[m, k, 2 * num_station[m] - 1, g])
                        # depot[m, k + 1, num_station[m] - 1, g] = depot[m, k, num_station[m] - 1, g] - (ul[m, k, num_station[m], g] - ul[m, k, num_station[m]-1, g])

un = np.zeros([num_line,num_train,2*max_station,group])
un_depart = np.zeros([num_line,num_train,2*max_station,group])
un_before = np.zeros([num_line,num_train,2*max_station,group])
uC = np.zeros([num_line,num_train,2*max_station,group])
un_after = np.zeros([num_line,num_train,2*max_station,group])
un_arrive = np.zeros([num_line,num_train,2*max_station,group])
for g in range(group):
    for m in range(num_line):
        for s in range(2*num_station[m]):
            un[m,0,s,g] = rho_whole[m,0,s,g]*(d_pre[m,0,s] - 0)
        for s in range(2*num_station[m]):
            for k in range(num_train - 1):
                un_before[m,k,s,g] = un[m,k,s,g] + rho_whole[m,k+1,s,g]*(ud[m,k,s,g] - d_pre[m,k,s])
                uC[m,k,s,g] = ul[m,k,s,g] * Cmax
                un_depart[m,k,s,g] = min(uC[m,k,s,g], un_before[m,k,s,g])
                un_after[m,k,s,g] = un_before[m,k,s,g] - un_depart[m,k,s,g]
                un[m,k+1,s,g] = un[m,k,s,g] + rho_whole[m,k+1,s,g]*(d_pre[m,k+1,s] - d_pre[m,k,s]) - un_depart[m,k,s,g]
        for s in range(1,2*num_station[m]):
            for k in range(num_train - 1):
                un_arrive[m, k, s, g] = un_depart[m, k, s-1, g]

un = np.zeros([num_line,num_train,2*max_station,group])
un_depart = np.zeros([num_line,num_train,2*max_station,group])
un_before = np.zeros([num_line,num_train,2*max_station,group])
uC = np.zeros([num_line,num_train,2*max_station,group])
un_after = np.zeros([num_line,num_train,2*max_station,group])
un_trans = np.zeros([num_line,num_train,2*max_station,group])
for g in range(group):
    for m in range(num_line):
        for s in range(2*num_station[m]):
            un[m,0,s,g] = rho_whole[m,0,s,g]*(d_pre[m,0,s] - 0)
        for s in range(2*num_station[m]):
            for k in range(num_train - 1):
                un_trans[m,k,s,g] = trans_rate*un_arrive[olin[m,s],otra[m,k,s,0],opla[m,s,0],g] + trans_rate*un_arrive[olin[m,s],otra[m,k,s,1],opla[m,s,1],g]
                un_before[m,k,s,g] = un[m,k,s,g] + rho_whole[m,k+1,s,g]*(ud[m,k,s,g] - d_pre[m,k,s]) + un_trans[m,k,s,g]
                uC[m,k,s,g] = ul[m,k,s,g] * Cmax
                un_depart[m,k,s,g] = min(uC[m,k,s,g], un_before[m,k,s,g])
                un_after[m,k,s,g] = un_before[m,k,s,g] - un_depart[m,k,s,g]
                un[m,k+1,s,g] = un[m,k,s,g] + rho_whole[m,k+1,s,g]*(d_pre[m,k+1,s] - d_pre[m,k,s]) + un_trans[m,k,s,g] - un_depart[m,k,s,g]

training_sets = {'d_pre': d_pre, 'rho_whole': rho_whole, 'un': un, 'un_after': un_after, 'ul': ul, 'uy': uy, 'ua': ua, 'ud': ud, 'utau': utau, 'ur': ur,
                 'depot': depot, 'r_max': r_max, 'r_min': r_min, 'differ': differ, 'Cmax': Cmax, 'sigma': sigma,
                 'num_station': num_station, 'num_train': num_train, 'max_station':max_station, 'num_line': num_line, 'E_regular': E_regular,
                 'olin':olin,'opla':opla,'otra':otra,'trans_rate':trans_rate}

np.save('data_railway//training_sets.npy', training_sets)
print('completed')
# print(depot[0,35:65,0,1:6])
# print(depot[1,35:65,0,1:6])
# print(depot[2,35:65,0,1:6])
# print(on_loop_trains)
# print(d_pre)
# print(E_regular)
