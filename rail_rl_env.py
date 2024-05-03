from gurobipy import Model, GRB, quicksum
import numpy as np
import time

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
t_roll = 300

"""
build action dictionary
"""
position1 = [np.array([[0,0,0,0]]),np.array([[0,0,0,1]]),np.array([[0,0,1,0]]), np.array([[0,0,1,1]]), np.array([[1,0,0,0]]),np.array([[1,0,0,1]]),np.array([[1,0,1,0]]),np.array([[1,0,1,1]]), np.array([[1,1,0,0]]),np.array([[1,1,0,1]]),np.array([[1,1,1,0]]),np.array([[1,1,1,1]])]
position2 = [np.array([[0,0,0,0]]),np.array([[0,0,0,1]]),np.array([[0,0,1,0]]), np.array([[0,0,1,1]]), np.array([[1,0,0,0]]),np.array([[1,0,0,1]]),np.array([[1,0,1,0]]),np.array([[1,0,1,1]]), np.array([[1,1,0,0]]),np.array([[1,1,0,1]]),np.array([[1,1,1,0]]),np.array([[1,1,1,1]])]
combined_arrays = []
for arr1 in position1:
    for arr2 in position2:
        combined_arrays.append(np.hstack((arr1, arr2)))
action_dict = {str(i): combined_arrays[i] for i in range(len(combined_arrays))}
# print(action_dict)

def original(control_trains,d_pre_cut,rho,d_real,l_real,state_n,Cmax,eta,start_index):
    or_n = np.zeros([control_trains+1, 2 * num_station])
    or_n_depart = np.zeros([control_trains, 2 * num_station])
    or_n_before = np.zeros([control_trains, 2 * num_station])
    or_n_after = np.zeros([control_trains, 2 * num_station])
    or_C = np.zeros([control_trains, 2 * num_station])
    J_original = 0
    for s in range(2 * num_station):
        or_n[0, s] = state_n[s]
        for k in range(control_trains):
            or_n_before[k, s] = or_n[k, s] + rho[k + 1, s] * (d_real[round(start_index[s] + k), s] - d_pre_cut[k, s])
            or_C[k, s] = l_real[round(start_index[s] + k), s] * Cmax
            or_n_depart[k, s] = min(or_C[k, s], or_n_before[k, s])
            or_n_after[k, s] = or_n_before[k, s] - or_n_depart[k, s]
            or_n[k + 1, s] = or_n[k, s] + rho[k + 1, s] * (d_pre_cut[k + 1, s] - d_pre_cut[k, s]) - or_n_depart[k, s]
        J_original=J_original+sum(eta*or_n[k,s]*(d_real[round(start_index[s]+k),s] - d_pre_cut[k,s]) + eta*or_n_after[k,s]*(d_pre_cut[k+1,s]-d_real[round(start_index[s]+k), s]) + l_real[round(start_index[s]+k), s]*E_regular[s] for k in range(control_trains))
    return J_original

def gurobi_minlp(control_trains,d_pre_cut,rho,state_a,state_d,state_r,state_l,state_y,state_n,depot_real,
              num_station,differ,sigma,same,t_constant,h_min,l_min,l_max,r_min,r_max,tau_min,E_regular,Cmax,eta, n_threads):
    """
    function of gurobi optimization
    control_trains:         horizon
    """
    epsilon = 10 ** (-10)
    Mt = 1000000
    mt = -1000000
    mdl = Model('OPT')
    mdl.Params.LogToConsole = 0
    mdl.Params.Threads = n_threads
    # decision variables
    d = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='departuretime')
    a = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='arrivaltime')
    r = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='runningtime')
    tau = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='dwelltime')
    tau_add = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='dwelltime_add')
    l = mdl.addVars(control_trains,2*num_station, vtype=GRB.INTEGER,name='composition')
    y = mdl.addVars(control_trains,2*num_station, lb=-l_max, ub=l_max, vtype=GRB.INTEGER,name='delta_composition')
    o = mdl.addVars(control_trains,2*num_station, vtype=GRB.INTEGER,name='delta_composition')
    gamma = mdl.addVars(control_trains,2*num_station, vtype=GRB.BINARY,name='delta_composition')
    sign_o = mdl.addVars(control_trains,2*num_station, vtype=GRB.BINARY,name='sign_composition')
    xi = mdl.addVars(control_trains,control_trains+1,2*num_station,2*num_station, vtype=GRB.BINARY,name='depot_wagon')

    n = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='waiting passengers')
    n_depart = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='departing passengers')
    n_before = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='passengers before departure')
    n_after = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='passengers after departure')
    C = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='Capacity')
    # objective
    mdl.setObjective(quicksum(eta*n[k,s]*(d[k,s]-d_pre_cut[k,s])+eta*n_after[k,s]*(d_pre_cut[k+1,s]-d[k,s]) + l[k,s]*E_regular[s] + sign_o[k,num_station]*50 for k in range(control_trains) for s in range(2*num_station)))
    # constraints
    mdl.addConstrs(a[k,s] == state_a[round(1+k),s] for k in range(2) for s in range(2*num_station))
    mdl.addConstrs(d[k,s] == state_d[round(1+k),s] for k in range(2) for s in range(2*num_station))
    mdl.addConstrs(r[k,s] == state_r[round(1+k),s] for k in range(2) for s in range(2*num_station))
    mdl.addConstrs(l[k,s] == state_l[round(1+k),s] for k in range(2) for s in range(2*num_station))
    mdl.addConstrs(y[k,s] == state_y[round(1+k),s] for k in range(2) for s in range(2*num_station))
    for s in range(1,2*num_station):
        for k in range(round(differ[s])):
            mdl.addConstr(a[k,s]==state_d[round(1+k-differ[s]),s-1]+state_r[round(1+k-differ[s]),s-1])
            mdl.addConstr(l[k,s]==state_l[round(1+k-differ[s]),s-1] + sigma[s]*y[k,s])
        for k in range(round(differ[s]),control_trains):
            mdl.addConstr(a[k,s] == d[round(k-differ[s]),s-1] + r[round(k-differ[s]),s-1])
            mdl.addConstr(l[k,s] == l[round(k-differ[s]),s-1] + sigma[s]*y[k,s])
        for k in range(control_trains,round(control_trains+differ[s])):
            mdl.addConstr(d[round(k-differ[s]), s-1] + r[round(k-differ[s]), s-1] == state_a[round(1+k),s])
            # mdl.addConstr(l[round(k-differ[s]), s-1] + sigma[s]*uy[round(start_train[s]+k),s] == ul[round(start_train[s]+k),s])
    # mdl.addConstrs(a[k,s] == ud[round(start_train[s]+1+k-differ[s]),s-1] + ur[round(start_train[s]+1+k-differ[s]),s-1] for k in range(round(differ[s])) for s in range(1,2*num_station))
    # mdl.addConstrs(a[k,s] == d[round(k-differ[s]),s-1] + r[round(k-differ[s]),s-1] for k in range(round(differ[s]),control_trains) for s in range(1,2*num_station))

    mdl.addConstrs(o[k,s] - y[k,s] >= 0 for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(o[k,s] - y[k,s] <= 2*l_max*(1-gamma[k,s]) for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(o[k,s] + y[k,s] >= 0 for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(o[k,s] + y[k,s] <= 2*l_max*gamma[k,s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(o[k, s] <= sign_o[k,s]*l_max for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(o[k, s] >= epsilon + (1-sign_o[k,s])*(-l_max - epsilon) for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(tau_add[k, s] == sign_o[k,s]*t_constant for k in range(control_trains) for s in range(2*num_station))

    mdl.addConstrs(d[k,s]>=d_pre_cut[k,s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(d[k,s]<=d_pre_cut[k+1,s] - epsilon for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(a[k+1,s]>=d[k,s]+h_min for k in range(control_trains-1) for s in range(2*num_station))
    mdl.addConstrs(d[k,s]==a[k,s]+tau[k,s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(d[control_trains-1,s]+h_min <= state_a[round(1+control_trains),s] for s in range(2*num_station))
    mdl.addConstrs(r[k,s]>=r_min[s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(r[k,s]<=r_max[s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(l[k,0]==y[k,0] for k in range(control_trains))
    mdl.addConstrs(y[k, num_station - 1] == 0 for k in range(control_trains))
    mdl.addConstrs(y[k,2*num_station-1]==-l[k,2*num_station-2] for k in range(control_trains))
    mdl.addConstrs(l[k,s]>=l_min for k in range(control_trains) for s in range(2*num_station-1))
    mdl.addConstrs(l[k,s]<=l_max for k in range(control_trains) for s in range(2*num_station-1))
    mdl.addConstrs(tau[k,s]>=tau_min+sigma[s]*tau_add[k,s] for k in range(control_trains) for s in range(1,2*num_station-1))
    mdl.addConstrs(tau[k,s]>=tau_min for k in range(control_trains) for s in range(2*num_station))

    for s in range(2*num_station):
        for q in range(2*num_station):
            if same[s,q] == 1:
                if s == 0 | s == num_station:
                    mdl.addConstrs(d[i,q]+t_roll-d[k,s]<=(1-xi[k,i,s,q])*(Mt-d[k,s]) for i in range(control_trains) for k in range(control_trains))
                    mdl.addConstrs(d[i,q]+t_roll-d[k,s]>=epsilon+xi[k,i,s,q]*(mt-d[k,s]-epsilon) for i in range(control_trains) for k in range(control_trains))
                    mdl.addConstrs(xi[k,i,s,q]<=xi[k,i-1,s,q] for i in range(1,control_trains) for k in range(control_trains))
                    mdl.addConstrs(xi[k,i,s,q]>=xi[k-1,i,s,q] for i in range(control_trains) for k in range(1,control_trains))
                    mdl.addConstrs(quicksum(xi[k,i,s,q]*y[i,q] for i in range(control_trains)) + quicksum(y[i,s] for i in range(2,k+1)) <= depot_real[min(s,q)] for k in range(2,control_trains-1))
    mdl.addConstrs(n[0,s]==state_n[s] for s in range(2*num_station))
    mdl.addConstrs(n[k+1,s]==n[k,s]+rho[k+1,s]*(d_pre_cut[k+1,s]-d_pre_cut[k,s])-n_depart[k,s] for k in range(control_trains-1) for s in range(2*num_station))
    mdl.addConstrs(n_before[k,s]==n[k,s]+rho[k+1,s]*(d[k,s]-d_pre_cut[k,s]) for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(n_depart[k,s]<=C[k,s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(n_depart[k,s]<=n_before[k,s] for k in range(control_trains) for s in range(2*num_station))
    # mdl.addConstrs(n_depart[k,s]==min(C[k,s],n_before[k,s]) for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(C[k,s]==l[k,s]*Cmax for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(n_after[k,s]==n_before[k,s]-n_depart[k,s] for k in range(control_trains) for s in range(2*num_station))

    # print(depot_real[0])
    start_time = time.time()
    # tune parameter of gurobi solver
    mdl.Params.MIPGap = 10 ** (-10)
    mdl.Params.Cuts = 2
    mdl.Params.NonConvex = 2
    # solving
    mdl.optimize()
    # mdl.remove(C1)
    # mdl.update()
    end_time = time.time()
    gurobi_runtime = end_time - start_time
    # print('gurobi runing time = %f seconds' % gurobi_runtime)
    a_values = np.zeros([control_trains, 2 * num_station])
    d_values = np.zeros([control_trains, 2 * num_station])
    r_values = np.zeros([control_trains, 2 * num_station])
    l_values = np.zeros([control_trains, 2 * num_station])
    y_values = np.zeros([control_trains, 2 * num_station])
    delta_minlp = np.zeros([control_trains - 1, 8])
    if mdl.status == GRB.OPTIMAL:
        # Print the objective function value
        # print(f"Optimal Objective Value: {mdl.objVal}")
        # obtain delta from the minlp solution
        link = np.zeros([control_trains, 2 * num_station, 2])
        index = 0
        for s in range(2 * num_station):
            for q in range(2 * num_station):
                if same[s, q] == 1:
                    for k in range(1, control_trains):
                        for i in range(control_trains + 1):
                            if (d_pre_cut[k, s] < d_pre_cut[i + 1, q] + t_roll) & (
                                    d_pre_cut[k + 1, s] > d_pre_cut[i, q] + t_roll):
                                link[k, s, index] = i
                                index = round(index + 1)
                        index = 0
        for k in range(1, control_trains):
            delta_minlp[k - 1, 0] = xi[k, link[k, 0, 0], 0, 2 * num_station - 1].x
            if link[k, 0, 1] <= control_trains - 1:
                delta_minlp[k - 1, 1] = xi[k, link[k, 0, 1], 0, 2 * num_station - 1].x
            delta_minlp[k - 1, 4] = xi[k, link[k, num_station, 0], num_station, num_station - 1].x
            if link[k, num_station, 1] <= control_trains - 1:
                delta_minlp[k - 1, 5] = xi[k, link[k, num_station, 1], num_station, num_station - 1].x
        for k in range(1, control_trains):
            if l[k, 0].x == 1:
                delta_minlp[k - 1, 2] = 0
                delta_minlp[k - 1, 3] = 0
            elif l[k, 0].x == 2:
                delta_minlp[k - 1, 2] = 1
                delta_minlp[k - 1, 3] = 0
            elif l[k, 0].x == 3:
                delta_minlp[k - 1, 2] = 0
                delta_minlp[k - 1, 3] = 1
            elif l[k, 0].x == 4:
                delta_minlp[k - 1, 2] = 1
                delta_minlp[k - 1, 3] = 1
            if l[k, num_station].x == 1:
                delta_minlp[k - 1, 6] = 0
                delta_minlp[k - 1, 7] = 0
            elif l[k, num_station].x == 2:
                delta_minlp[k - 1, 6] = 1
                delta_minlp[k - 1, 7] = 0
            elif l[k, num_station].x == 3:
                delta_minlp[k - 1, 6] = 0
                delta_minlp[k - 1, 7] = 1
            elif l[k, num_station].x == 4:
                delta_minlp[k - 1, 6] = 1
                delta_minlp[k - 1, 7] = 1
            for s in range(2 * num_station):
                for k in range(control_trains):
                    a_values[k, s] = a[k, s].x
                    d_values[k, s] = d[k, s].x
                    r_values[k, s] = r[k, s].x
                    l_values[k, s] = l[k, s].x
                    y_values[k, s] = y[k, s].x
    # else:
    #     print("Optimization did not converge to an optimal solution.")

    return a_values, d_values, r_values, l_values, y_values, delta_minlp, mdl

def gurobi_qp(control_trains,d_pre_cut,rho,state_a,state_d,state_r,state_l,state_y,state_n,depot_real,delta,
              num_station,differ,sigma,same,t_constant,h_min,l_min,l_max,r_min,r_max,tau_min,E_regular,Cmax,eta):
    """
    function of gurobi optimization
    control_trains:         horizon
    """
    epsilon = 10 ** (-10)
    Mt = 1000000
    mt = -1000000
    mdl = Model('OPT')
    mdl.Params.LogToConsole = 0
    # decision variables
    d = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='departuretime')
    a = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='arrivaltime')
    r = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='runningtime')
    tau = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='dwelltime')
    tau_add = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='dwelltime_add')
    l = mdl.addVars(control_trains,2*num_station, vtype=GRB.INTEGER,name='composition')
    y = mdl.addVars(control_trains,2*num_station, lb=-l_max, ub=l_max, vtype=GRB.INTEGER,name='delta_composition')
    o = mdl.addVars(control_trains,2*num_station, vtype=GRB.INTEGER,name='delta_composition')
    gamma = mdl.addVars(control_trains,2*num_station, vtype=GRB.BINARY,name='delta_composition')
    sign_o = mdl.addVars(control_trains,2*num_station, vtype=GRB.BINARY,name='sign_composition')
    xi = mdl.addVars(control_trains,control_trains+1,2*num_station,2*num_station, vtype=GRB.BINARY,name='depot_wagon')

    n = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='waiting passengers')
    n_depart = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='departing passengers')
    n_before = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='passengers before departure')
    n_after = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='passengers after departure')
    C = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='Capacity')
    # objective
    mdl.setObjective(quicksum(eta*n[k,s]*(d[k,s]-d_pre_cut[k,s])+eta*n_after[k,s]*(d_pre_cut[k+1,s]-d[k,s]) + l[k,s]*E_regular[s] + sign_o[k,num_station]*50 for k in range(control_trains) for s in range(2*num_station)))
    # fixed integer from delta
    link = np.zeros([control_trains, 2 * num_station, 2])
    index = 0
    for s in range(2 * num_station):
        for q in range(2 * num_station):
            if same[s, q] == 1:
                for k in range(1, control_trains):
                    for i in range(control_trains + 1):
                        if (d_pre_cut[k, s] < d_pre_cut[i + 1, q] + t_roll) & (d_pre_cut[k + 1, s] > d_pre_cut[i, q] + t_roll):
                            link[k, s, index] = i
                            index = round(index + 1)
                    index = 0
    for k in range(1, control_trains):
        mdl.addConstr(xi[k, link[k, 0, 0], 0, 2 * num_station - 1] == delta[k - 1, 0])
        if link[k, 0, 1] <= control_trains - 1:
            mdl.addConstr(xi[k, link[k, 0, 1], 0, 2 * num_station - 1] == delta[k - 1, 1])
        mdl.addConstr(xi[k, link[k, num_station, 0], num_station, num_station - 1] == delta[k - 1, 4])
        if link[k, num_station, 1] <= control_trains - 1:
            mdl.addConstr(xi[k, link[k, num_station, 1], num_station, num_station - 1] == delta[k - 1, 5])
    for k in range(1, control_trains):
        mdl.addConstr(l[k, 0] == round(1 + delta[k - 1, 2] + 2 * delta[k - 1, 3]))
        mdl.addConstr(l[k, num_station] == round(1 + delta[k - 1, 6] + 2 * delta[k - 1, 7]))
    # constraints
    mdl.addConstrs(a[k,s] == state_a[round(1+k),s] for k in range(2) for s in range(2*num_station))
    mdl.addConstrs(d[k,s] == state_d[round(1+k),s] for k in range(2) for s in range(2*num_station))
    mdl.addConstrs(r[k,s] == state_r[round(1+k),s] for k in range(2) for s in range(2*num_station))
    mdl.addConstrs(l[k,s] == state_l[round(1+k),s] for k in range(2) for s in range(2*num_station))
    mdl.addConstrs(y[k,s] == state_y[round(1+k),s] for k in range(2) for s in range(2*num_station))
    for s in range(1,2*num_station):
        for k in range(round(differ[s])):
            mdl.addConstr(a[k,s]==state_d[round(1+k-differ[s]),s-1]+state_r[round(1+k-differ[s]),s-1])
            mdl.addConstr(l[k,s]==state_l[round(1+k-differ[s]),s-1] + sigma[s]*y[k,s])
        for k in range(round(differ[s]),control_trains):
            mdl.addConstr(a[k,s] == d[round(k-differ[s]),s-1] + r[round(k-differ[s]),s-1])
            mdl.addConstr(l[k,s] == l[round(k-differ[s]),s-1] + sigma[s]*y[k,s])
        for k in range(control_trains,round(control_trains+differ[s])):
            mdl.addConstr(d[round(k-differ[s]), s-1] + r[round(k-differ[s]), s-1] == state_a[round(1+k),s])
            # mdl.addConstr(l[round(k-differ[s]), s-1] + sigma[s]*uy[round(start_train[s]+k),s] == ul[round(start_train[s]+k),s])
    # mdl.addConstrs(a[k,s] == ud[round(start_train[s]+1+k-differ[s]),s-1] + ur[round(start_train[s]+1+k-differ[s]),s-1] for k in range(round(differ[s])) for s in range(1,2*num_station))
    # mdl.addConstrs(a[k,s] == d[round(k-differ[s]),s-1] + r[round(k-differ[s]),s-1] for k in range(round(differ[s]),control_trains) for s in range(1,2*num_station))

    mdl.addConstrs(o[k,s] - y[k,s] >= 0 for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(o[k,s] - y[k,s] <= 2*l_max*(1-gamma[k,s]) for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(o[k,s] + y[k,s] >= 0 for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(o[k,s] + y[k,s] <= 2*l_max*gamma[k,s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(o[k, s] <= sign_o[k,s]*l_max for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(o[k, s] >= epsilon + (1-sign_o[k,s])*(-l_max - epsilon) for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(tau_add[k, s] == sign_o[k,s]*t_constant for k in range(control_trains) for s in range(2*num_station))

    mdl.addConstrs(d[k,s]>=d_pre_cut[k,s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(d[k,s]<=d_pre_cut[k+1,s] - epsilon for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(a[k+1,s]>=d[k,s]+h_min for k in range(control_trains-1) for s in range(2*num_station))
    mdl.addConstrs(d[k,s]==a[k,s]+tau[k,s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(d[control_trains-1,s]+h_min <= state_a[round(1+control_trains),s] for s in range(2*num_station))
    mdl.addConstrs(r[k,s]>=r_min[s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(r[k,s]<=r_max[s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(l[k,0]==y[k,0] for k in range(control_trains))
    mdl.addConstrs(y[k, num_station - 1] == 0 for k in range(control_trains))
    mdl.addConstrs(y[k,2*num_station-1]==-l[k,2*num_station-2] for k in range(control_trains))
    mdl.addConstrs(l[k,s]>=l_min for k in range(control_trains) for s in range(2*num_station-1))
    mdl.addConstrs(l[k,s]<=l_max for k in range(control_trains) for s in range(2*num_station-1))
    mdl.addConstrs(tau[k,s]>=tau_min+sigma[s]*tau_add[k,s] for k in range(control_trains) for s in range(1,2*num_station-1))
    mdl.addConstrs(tau[k,s]>=tau_min for k in range(control_trains) for s in range(2*num_station))

    for s in range(2*num_station):
        for q in range(2*num_station):
            if same[s,q] == 1:
                if s == 0 | s == num_station:
                    mdl.addConstrs(d[i,q]+t_roll-d[k,s]<=(1-xi[k,i,s,q])*(Mt-d[k,s]) for i in range(control_trains) for k in range(control_trains))
                    mdl.addConstrs(d[i,q]+t_roll-d[k,s]>=epsilon+xi[k,i,s,q]*(mt-d[k,s]-epsilon) for i in range(control_trains) for k in range(control_trains))
                    mdl.addConstrs(xi[k,i,s,q]<=xi[k,i-1,s,q] for i in range(1,control_trains) for k in range(control_trains))
                    mdl.addConstrs(xi[k,i,s,q]>=xi[k-1,i,s,q] for i in range(control_trains) for k in range(1,control_trains))
                    mdl.addConstrs(quicksum(xi[k,i,s,q]*y[i,q] for i in range(control_trains)) + quicksum(y[i,s] for i in range(2,k+1)) <= depot_real[min(s,q)] for k in range(2,control_trains-1))
    mdl.addConstrs(n[0,s]==state_n[s] for s in range(2*num_station))
    mdl.addConstrs(n[k+1,s]==n[k,s]+rho[k+1,s]*(d_pre_cut[k+1,s]-d_pre_cut[k,s])-n_depart[k,s] for k in range(control_trains-1) for s in range(2*num_station))
    mdl.addConstrs(n_before[k,s]==n[k,s]+rho[k+1,s]*(d[k,s]-d_pre_cut[k,s]) for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(n_depart[k,s]<=C[k,s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(n_depart[k,s]<=n_before[k,s] for k in range(control_trains) for s in range(2*num_station))
    # mdl.addConstrs(n_depart[k,s]==min(C[k,s],n_before[k,s]) for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(C[k,s]==l[k,s]*Cmax for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(n_after[k,s]==n_before[k,s]-n_depart[k,s] for k in range(control_trains) for s in range(2*num_station))

    # print(depot_real[0])
    start_time = time.time()
    # tune parameter of gurobi solver
    mdl.Params.MIPGap = 10 ** (-10)
    mdl.Params.Cuts = 2
    mdl.Params.NonConvex = 2
    # solving
    mdl.optimize()
    # mdl.remove(C1)
    # mdl.update()
    end_time = time.time()
    gurobi_runtime = end_time - start_time
    # print('gurobi runing time = %f seconds' % gurobi_runtime)
    a_values = np.zeros([control_trains, 2 * num_station])
    d_values = np.zeros([control_trains, 2 * num_station])
    r_values = np.zeros([control_trains, 2 * num_station])
    l_values = np.zeros([control_trains, 2 * num_station])
    y_values = np.zeros([control_trains, 2 * num_station])
    if mdl.status == GRB.OPTIMAL:
        # Print the objective function value
        # print(f"Optimal Objective Value: {mdl.objVal}")
        for s in range(2 * num_station):
            for k in range(control_trains):
                a_values[k, s] = a[k, s].x
                d_values[k, s] = d[k, s].x
                r_values[k, s] = r[k, s].x
                l_values[k, s] = l[k, s].x
                y_values[k, s] = y[k, s].x
    # else:
    #     print("Optimization did not converge to an optimal solution.")

    return a_values, d_values, r_values, l_values, y_values, mdl

def gurobi_qp_presolve(control_trains,d_pre_cut,rho,state_a,state_d,state_r,state_l,state_y,state_n,depot_real,delta,
              num_station,differ,sigma,same,t_constant,h_min,l_min,l_max,r_min,r_max,tau_min,E_regular,Cmax,eta):
    """
    function of gurobi optimization
    control_trains:         horizon
    """
    epsilon = 10 ** (-10)
    Mt = 1000000
    mt = -1000000
    l = np.zeros([control_trains, 2 * num_station])
    y = np.zeros([control_trains, 2 * num_station])
    sign_o = np.zeros([control_trains, 2 * num_station])
    for s in range(2 * num_station):
        for k in range(2):
            l[k, s] = state_l[round(1 + k), s]
            y[k, s] = state_y[round(1 + k), s]
    for k in range(1, control_trains):
        l[k, 0] = round(1 + delta[k - 1, 2] + 2 * delta[k - 1, 3])
        y[k, 0] = l[k, 0]
    for s in range(1, num_station):
        for k in range(1, control_trains):
            l[k, s] = l[round(k - differ[s]), s - 1] + sigma[s] * y[k, s]
    for k in range(1, control_trains):
        l[k, num_station] = round(1 + delta[k - 1, 6] + 2 * delta[k - 1, 7])
        y[k, num_station] = l[k, num_station] - l[round(k - differ[num_station]), num_station - 1]
    for s in range(num_station + 1, 2 * num_station - 1):
        for k in range(1, control_trains):
            l[k, s] = l[round(k - differ[s]), s - 1] + sigma[s] * y[k, s]
    for k in range(1, control_trains):
        y[k, 2 * num_station - 1] = -l[round(k - differ[2 * num_station - 1]), 2 * num_station - 2]
    for s in range(2 * num_station):
        for k in range(control_trains):
            if y[k, s] == 0:
                sign_o[k, s] = 0
            else:
                sign_o[k, s] = 1

    mdl = Model('OPT')
    mdl.Params.LogToConsole = 0
    # decision variables
    d = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='departuretime')
    a = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='arrivaltime')
    r = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='runningtime')
    tau = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='dwelltime')
    tau_add = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='dwelltime_add')

    n = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='waiting passengers')
    n_depart = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='departing passengers')
    n_before = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='passengers before departure')
    n_after = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='passengers after departure')
    C = mdl.addVars(control_trains,2*num_station, vtype=GRB.CONTINUOUS,name='Capacity')
    # objective
    mdl.setObjective(quicksum(eta * n[k, s] * (d[k, s] - d_pre_cut[k, s]) + eta * n_after[k, s] * (d_pre_cut[k + 1, s] - d[k, s]) + l[k, s] * E_regular[s] + sign_o[k,num_station]*50 for k in range(control_trains) for s in range(2 * num_station)))
    # fixed integer from delta
    link = np.zeros([control_trains, 2 * num_station, 2])
    index = 0
    for s in range(2 * num_station):
        for q in range(2 * num_station):
            if same[s, q] == 1:
                for k in range(1, control_trains):
                    for i in range(control_trains + 1):
                        if (d_pre_cut[k, s] < d_pre_cut[i + 1, q] + t_roll) & (d_pre_cut[k + 1, s] > d_pre_cut[i, q] + t_roll):
                            link[k, s, index] = i
                            index = round(index + 1)
                    index = 0
    # xi = mdl.addVars(control_trains, control_trains+1, 2 * num_station, 2 * num_station, vtype=GRB.BINARY, name='depot_wagon')
    # for k in range(2, control_trains):
    #     mdl.addConstr(xi[k, link[k, 0, 0], 0, 2 * num_station - 1] == delta[k - 2, 0])
    #     if link[k, 0, 1] <= control_trains - 1:
    #         mdl.addConstr(xi[k, link[k, 0, 1], 0, 2 * num_station - 1] == delta[k - 2, 1])
    #     mdl.addConstr(xi[k, link[k, num_station, 0], num_station, num_station - 1] == delta[k - 2, 4])
    #     if link[k, num_station, 1] <= control_trains - 1:
    #         mdl.addConstr(xi[k, link[k, num_station, 1], num_station, num_station - 1] == delta[k - 2, 5])

    xi = np.zeros([control_trains, control_trains + 1, 2 * num_station, 2 * num_station])
    for k in range(1, control_trains):
        xi[k, round(link[k, 0, 0]), 0, 2 * num_station - 1] = delta[k - 1, 0]
        if link[k, 0, 1] <= control_trains - 1:
            xi[k, round(link[k, 0, 1]), 0, 2 * num_station - 1] = delta[k - 1, 1]
        xi[k, round(link[k, num_station, 0]), num_station, num_station - 1] = delta[k - 1, 4]
        if link[k, num_station, 1] <= control_trains - 1:
            xi[k, round(link[k, num_station, 1]), num_station, num_station - 1] = delta[k - 1, 5]
    for s in range(2 * num_station):
        for q in range(2 * num_station):
            if same[s, q] == 1:
                for k in range(control_trains):
                    for i in range(control_trains):
                        if d_pre_cut[k, s] >= d_pre_cut[i + 1, q] + t_roll:
                            xi[k, i, s, q] = round(1)
                        elif d_pre_cut[k + 1, s] <= d_pre_cut[i, q] + t_roll:
                            xi[k, i, s, q] = round(0)
                for k in range(2):
                    for i in range(control_trains):
                        if i <= 1:
                            if state_d[round(1 + k), s] >= state_d[round(1 + i), q] + t_roll:
                                xi[k, i, s, q] = round(1)
                            else:
                                xi[k, i, s, q] = round(0)
    # constraints
    mdl.addConstrs(a[k,s] == state_a[round(1+k),s] for k in range(2) for s in range(2*num_station))
    mdl.addConstrs(d[k,s] == state_d[round(1+k),s] for k in range(2) for s in range(2*num_station))
    mdl.addConstrs(r[k,s] == state_r[round(1+k),s] for k in range(2) for s in range(2*num_station))

    for s in range(1,2*num_station):
        for k in range(round(differ[s])):
            mdl.addConstr(a[k,s]==state_d[round(1+k-differ[s]),s-1]+state_r[round(1+k-differ[s]),s-1])
        for k in range(round(differ[s]),control_trains):
            mdl.addConstr(a[k,s] == d[round(k-differ[s]),s-1] + r[round(k-differ[s]),s-1])
        for k in range(control_trains,round(control_trains+differ[s])):
            mdl.addConstr(d[round(k-differ[s]), s-1] + r[round(k-differ[s]), s-1] == state_a[round(1+k),s])
            # mdl.addConstr(l[round(k-differ[s]), s-1] + sigma[s]*uy[round(start_train[s]+k),s] == ul[round(start_train[s]+k),s])
    # mdl.addConstrs(a[k,s] == ud[round(start_train[s]+1+k-differ[s]),s-1] + ur[round(start_train[s]+1+k-differ[s]),s-1] for k in range(round(differ[s])) for s in range(1,2*num_station))
    # mdl.addConstrs(a[k,s] == d[round(k-differ[s]),s-1] + r[round(k-differ[s]),s-1] for k in range(round(differ[s]),control_trains) for s in range(1,2*num_station))

    mdl.addConstrs(tau_add[k, s] == sign_o[k,s]*t_constant for k in range(control_trains) for s in range(2*num_station))

    mdl.addConstrs(d[k,s]>=d_pre_cut[k,s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(d[k,s]<=d_pre_cut[k+1,s] - epsilon for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(a[k+1,s]>=d[k,s]+h_min for k in range(control_trains-1) for s in range(2*num_station))
    mdl.addConstrs(d[k,s]==a[k,s]+tau[k,s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(d[control_trains-1,s]+h_min <= state_a[round(1+control_trains),s] for s in range(2*num_station))
    mdl.addConstrs(r[k,s]>=r_min[s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(r[k,s]<=r_max[s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(tau[k,s]>=tau_min+sigma[s]*tau_add[k,s] for k in range(control_trains) for s in range(1,2*num_station-1))
    mdl.addConstrs(tau[k,s]>=tau_min for k in range(control_trains) for s in range(2*num_station))

    for s in range(2*num_station):
        for q in range(2*num_station):
            if same[s,q] == 1:
                if s == 0 | s == num_station:
                    mdl.addConstrs(d[i,q]+t_roll-d[k,s]<=(1-xi[k,i,s,q])*(Mt-d[k,s]) for i in range(control_trains) for k in range(control_trains))
                    mdl.addConstrs(d[i,q]+t_roll-d[k,s]>=epsilon+xi[k,i,s,q]*(mt-d[k,s]-epsilon) for i in range(control_trains) for k in range(control_trains))
                    # mdl.addConstrs(xi[k,i,s,q]<=xi[k,i-1,s,q] for i in range(1,control_trains) for k in range(control_trains))
                    # mdl.addConstrs(xi[k,i,s,q]>=xi[k-1,i,s,q] for i in range(control_trains) for k in range(1,control_trains))
                    mdl.addConstrs(quicksum(xi[k,i,s,q]*y[i,q] for i in range(control_trains)) + quicksum(y[i,s] for i in range(2,k+1)) <= depot_real[min(s,q)] for k in range(2,control_trains-1))
    mdl.addConstrs(n[0,s]==state_n[s] for s in range(2*num_station))
    mdl.addConstrs(n[k+1,s]==n[k,s]+rho[k+1,s]*(d_pre_cut[k+1,s]-d_pre_cut[k,s])-n_depart[k,s] for k in range(control_trains-1) for s in range(2*num_station))
    mdl.addConstrs(n_before[k,s]==n[k,s]+rho[k+1,s]*(d[k,s]-d_pre_cut[k,s]) for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(n_depart[k,s]<=C[k,s] for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(n_depart[k,s]<=n_before[k,s] for k in range(control_trains) for s in range(2*num_station))
    # mdl.addConstrs(n_depart[k,s]==min(C[k,s],n_before[k,s]) for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(C[k,s]==l[k,s]*Cmax for k in range(control_trains) for s in range(2*num_station))
    mdl.addConstrs(n_after[k,s]==n_before[k,s]-n_depart[k,s] for k in range(control_trains) for s in range(2*num_station))

    # print(depot_real[0])
    start_time = time.time()
    # tune parameter of gurobi solver
    mdl.Params.MIPGap = 10 ** (-10)
    mdl.Params.Cuts = 2
    mdl.Params.NonConvex = 2
    # solving
    mdl.optimize()
    # mdl.remove(C1)
    # mdl.update()
    end_time = time.time()
    gurobi_runtime = end_time - start_time
    # print('gurobi runing time = %f seconds' % gurobi_runtime)
    a_values = np.zeros([control_trains, 2 * num_station])
    d_values = np.zeros([control_trains, 2 * num_station])
    r_values = np.zeros([control_trains, 2 * num_station])
    l_values = np.zeros([control_trains, 2 * num_station])
    y_values = np.zeros([control_trains, 2 * num_station])
    if mdl.status == GRB.OPTIMAL:
        # Print the objective function value
        # print(f"Optimal Objective Value: {mdl.objVal}")
        for s in range(2 * num_station):
            for k in range(control_trains):
                a_values[k, s] = a[k, s].x
                d_values[k, s] = d[k, s].x
                r_values[k, s] = r[k, s].x
        l_values = l
        y_values = y
    # else:
        # print("Optimization did not converge to an optimal solution.")

    return a_values, d_values, r_values, l_values, y_values, mdl

def qp_feasible(mdl):
    """
    checks whether the gurobi mdl is feasible
    """
    if mdl.status == 3 or mdl.status == 4 or mdl.status == 12:
        feas = False
        return feas
    else:
        feas = True
        return feas

def cost2rew2(cost, noctrl_cost):
    """
    computes the reward by scaling the objective value (cost) from the LP
    Returns:
        reward: float
    """

    if cost <= noctrl_cost:
        rew = 1 + (cost - noctrl_cost) / (0.8*noctrl_cost - noctrl_cost)
    else:
        rew = 1 - (cost - noctrl_cost) / (1.2*noctrl_cost - noctrl_cost)

    return rew

class RailNet():
    def __init__(self, control_trains, mode=np.array([[0, 0, 0, 0, 0, 0, 0, 0]])):

        self.cntr = 0  # termination counter
        self.control_trains = control_trains
        self.mode = mode
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.idx_cntr = 0
        self.idx_group = 0 # group index

    def setState(self, idx_cntr, idx_group, mode=np.array([[0, 0, 0, 0, 0, 0, 0, 0]])):
        self.cntr = 0
        self.mode = mode
        self.terminated = False
        self.truncated = False
        self.idx_cntr = idx_cntr
        self.idx_group = idx_group

    def copyEnv(self, env):
        self.cntr = env.cntr
        self.state_start = env.state_start
        self.mode = env.mode
        self.reward = env.reward
        self.terminated = env.terminated
        self.truncated = env.truncated
        self.idx_cntr = env.idx_cntr
        self.idx_group = env.idx_group

    def set_randState(self, d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot, mode=np.array([[0, 0, 0, 0, 0, 0, 0, 0]])):

        i = np.random.randint(15, 210 - self.control_trains)  # equivalent for 1 year of data Ts=30m
        g = np.random.randint(1000)

        self.mode = mode
        self.cntr = 0
        self.terminated = False
        self.truncated = False
        self.idx_cntr = i
        self.idx_group = g
        self.a_real = np.zeros([num_train, 2 * num_station])
        self.d_real = np.zeros([num_train, 2 * num_station])
        self.r_real = np.zeros([num_train, 2 * num_station])
        self.l_real = np.zeros([num_train, 2 * num_station])
        self.y_real = np.zeros([num_train, 2 * num_station])
        self.n_real = np.zeros([num_train, 2 * num_station])
        for s in range(2 * num_station):
            for k in range(num_train):
                self.a_real[k, s] = ua[k, s, self.idx_group]
                self.d_real[k, s] = ud[k, s, self.idx_group]
                self.r_real[k, s] = ur[k, s, self.idx_group]
                self.l_real[k, s] = ul[k, s, self.idx_group]
                self.y_real[k, s] = uy[k, s, self.idx_group]
                self.n_real[k, s] = un[k, s, self.idx_group]
        self.depot_real = np.zeros([num_train, num_station])
        for s in range(num_station):
            for k in range(num_train):
                self.depot_real[k, s] = depot[k, s, self.idx_group]

        '''cutting states for gurobi'''
        self.start_index = np.zeros([2 * num_station])
        self.start_index[0] = self.idx_cntr
        for s in range(1, 2 * num_station):
            self.start_index[s] = self.start_index[s - 1] - differ[s]
        self.state_rho = np.zeros([self.control_trains + 1, 2 * num_station])
        for s in range(2 * num_station):
            for k in range(self.control_trains + 1):
                self.state_rho[k, s] = rho_whole[round(self.start_index[s] + k), s, self.idx_group]
        self.state_a = np.zeros([self.control_trains + 2, 2 * num_station])
        self.d_pre_cut = np.zeros([self.control_trains + 2, 2 * num_station])
        for s in range(2 * num_station):
            for k in range(self.control_trains + 2):
                self.state_a[k, s] = self.a_real[round(self.start_index[s] - 1 + k), s]
                self.d_pre_cut[k, s] = d_pre[round(self.start_index[s] + k), s]
        self.state_d = np.zeros([3, 2 * num_station])
        self.state_r = np.zeros([3, 2 * num_station])
        self.state_l = np.zeros([3, 2 * num_station])
        self.state_y = np.zeros([3, 2 * num_station])
        for s in range(2 * num_station):
            for k in range(3):
                self.state_d[k, s] = self.d_real[round(self.start_index[s] - 1 + k), s]
                self.state_r[k, s] = self.r_real[round(self.start_index[s] - 1 + k), s]
                self.state_l[k, s] = self.l_real[round(self.start_index[s] - 1 + k), s]
                self.state_y[k, s] = self.y_real[round(self.start_index[s] - 1 + k), s]
        self.state_n = np.zeros([2 * num_station])
        for s in range(2 * num_station):
            self.state_n[s] = self.n_real[round(self.start_index[s]), s]
        self.state_depot = np.zeros([num_station])
        for s in range(num_station):
            self.state_depot[s] = self.depot_real[round(self.start_index[s] + 2), s]

    def build_delta_vector(self, list_action: list) -> np.array:
        # from list of actions builds a np.array with the stacked deltas for each time step of the prediction horizon
        delta = action_dict[str(round(list_action[0]))]
        for i in range(1, self.control_trains - 1):
            delta = np.concatenate((delta, action_dict[str(round(list_action[i]))]))

        return delta

    def step(self, list_action, d_pre, rho_whole, r_max, r_min, differ, Cmax, sigma, same, num_station,num_train, E_regular):

        t_constant = 60
        h_min = 120
        tau_min = 30
        l_min = 1
        l_max = 4
        eta = 10 ** (-3)

        '''objective value of the on_control cost'''
        J_original = original(self.control_trains,self.d_pre_cut,self.state_rho,self.d_real,self.l_real,self.state_n,Cmax,eta,self.start_index)
        # print(J_original)

        # a_minlp, d_minlp, r_minlp, l_minlp, y_minlp, delta_minlp, mdl_minlp = gurobi_minlp(self.control_trains, self.d_pre_cut, self.state_rho,
        #              self.state_a, self.state_d, self.state_r, self.state_l, self.state_y, self.state_n, self.state_depot,
        #              num_station, differ, sigma, same, t_constant, h_min, l_min, l_max, r_min, r_max, tau_min,
        #              E_regular, Cmax, eta)

        delta = self.build_delta_vector(list_action)
        '''use delta generated by minlp to guribi_qp'''
        # delta = delta_minlp
        a_qp, d_qp, r_qp, l_qp, y_qp, mdl = gurobi_qp_presolve(self.control_trains,self.d_pre_cut,self.state_rho,
                        self.state_a, self.state_d, self.state_r, self.state_l, self.state_y, self.state_n,self.state_depot,
                        delta, num_station,differ,sigma,same,t_constant,h_min,l_min,l_max,r_min,r_max,tau_min,E_regular,Cmax,eta)

        feas = qp_feasible(mdl)

        if feas == False:

            self.reward = np.array(-1)
            self.terminated = True

            info = {'feasible': feas}

        else:
            # rew = cost2rew(mdl.ObjVal, lower_bound=-450, alpha=0.0015)
            rew = cost2rew2(mdl.ObjVal, J_original)
            '''implement the first control input in MPC'''
            for s in range(2 * num_station):
                self.a_real[round(self.start_index[s] + 2), s] = a_qp[2, s]
                self.r_real[round(self.start_index[s] + 2), s] = r_qp[2, s]
                self.d_real[round(self.start_index[s] + 2), s] = d_qp[2, s]
                self.l_real[round(self.start_index[s] + 2), s] = l_qp[2, s]
                self.y_real[round(self.start_index[s] + 2), s] = y_qp[2, s]
            for k in range(num_train - 1):
                if d_pre[k, 0] < d_pre[0, 2 * num_station - 1] + t_roll:
                    self.depot_real[k + 1, 0] = self.depot_real[k, 0] - self.y_real[k, 0]
                    self.depot_real[k + 1, num_station - 1] = self.depot_real[k, num_station - 1] - (self.l_real[k, num_station] - self.l_real[k, num_station - 1])
                else:
                    for i in range(num_train - 1):
                        if (d_pre[k, 0] >= d_pre[i, 2 * num_station - 1] + t_roll) & (d_pre[k, 0] < d_pre[i + 1, 2 * num_station - 1] + t_roll):
                            self.depot_real[k + 1, 0] = self.depot_real[k, 0] - self.y_real[k, 0] - self.y_real[i, 2 * num_station - 1]
                            self.depot_real[k + 1, num_station - 1] = self.depot_real[k, num_station - 1] - (self.l_real[k, num_station] - self.l_real[k, num_station - 1])
            self.n_real = np.zeros([num_train, 2 * num_station])
            n_depart_real = np.zeros([num_train, 2 * num_station])
            n_before_real = np.zeros([num_train, 2 * num_station])
            C_real = np.zeros([num_train, 2 * num_station])
            n_after_real = np.zeros([num_train, 2 * num_station])
            for s in range(2 * num_station):
                self.n_real[0, s] = rho_whole[0, s, self.idx_group] * (d_pre[0, s] - 0)
            for s in range(2 * num_station):
                for k in range(num_train - 1):
                    n_before_real[k, s] = self.n_real[k, s] + rho_whole[k + 1, s, self.idx_group] * (self.d_real[k, s] - d_pre[k, s])
                    C_real[k, s] = self.l_real[k, s] * Cmax
                    n_depart_real[k, s] = min(C_real[k, s], n_before_real[k, s])
                    n_after_real[k, s] = n_before_real[k, s] - n_depart_real[k, s]
                    self.n_real[k + 1, s] = self.n_real[k, s] + rho_whole[k + 1, s, self.idx_group] * (d_pre[k + 1, s] - d_pre[k, s]) - n_depart_real[k, s]

            # print(np.amax(self.n_real))
            self.mode = action_dict[str(round(list_action[0]))]
            self.reward = rew
            self.cntr += 1
            self.idx_cntr += 1
            self.start_index += 1

            '''update states'''
            self.state_rho = np.zeros([self.control_trains + 1, 2 * num_station])
            self.d_pre_cut = np.zeros([self.control_trains + 2, 2 * num_station])
            self.state_a = np.zeros([self.control_trains + 2, 2 * num_station])
            self.state_d = np.zeros([3, 2 * num_station])
            self.state_r = np.zeros([3, 2 * num_station])
            self.state_l = np.zeros([3, 2 * num_station])
            self.state_y = np.zeros([3, 2 * num_station])
            self.state_n = np.zeros([2 * num_station])
            self.state_depot = np.zeros([num_station])
            for s in range(2 * num_station):
                for k in range(self.control_trains + 1):
                    self.state_rho[k, s] = rho_whole[round(self.start_index[s] + k), s, self.idx_group]
                for k in range(self.control_trains + 2):
                    self.state_a[k, s] = self.a_real[round(self.start_index[s] - 1 + k), s]
                    self.d_pre_cut[k, s] = d_pre[round(self.start_index[s] + k), s]
                for k in range(3):
                    self.state_d[k, s] = self.d_real[round(self.start_index[s] - 1 + k), s]
                    self.state_r[k, s] = self.r_real[round(self.start_index[s] - 1 + k), s]
                    self.state_l[k, s] = self.l_real[round(self.start_index[s] - 1 + k), s]
                    self.state_y[k, s] = self.y_real[round(self.start_index[s] - 1 + k), s]
                self.state_n[s] = self.n_real[round(self.start_index[s]), s]
            for s in range(num_station):
                self.state_depot[s] = self.depot_real[round(self.start_index[s] + 2), s]

            if (self.cntr >= 210 - self.control_trains - self.idx_cntr)|(self.cntr >= 15):
                self.truncated = True

            if self.idx_cntr >= 210 - self.control_trains:
                self.truncated = True

            info = {'feasible': feas,
                    'objval': mdl.ObjVal,
                    'mdl': mdl,
                    }

        return (self.state_rho, self.d_pre_cut, self.state_a, self.state_d, self.state_r, self.state_l, self.state_y,self.state_n, self.state_depot,
                self.reward, self.terminated, self.truncated, info)
