from gurobipy import Model, GRB, quicksum
import numpy as np
import copy

data_sets = np.load('data_railway//training_sets.npy', allow_pickle=True).item()
d_pre = data_sets['d_pre']
rho_whole = data_sets['rho_whole']
un = data_sets['un']
un_after = data_sets['un_after']
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
num_station = data_sets['num_station']
max_station = data_sets['max_station']
num_train = data_sets['num_train']
num_line = data_sets['num_line']
E_regular = data_sets['E_regular']
olin = data_sets['olin']
opla = data_sets['opla']
otra = data_sets['otra']
trans_rate = data_sets['trans_rate']

epsilon = 10 ** (-10)
Mt = 1000000
mt = -1000000
t_roll = 240
t_constant = 60
h_min = 120
tau_min = 30
l_min = 1
l_max = 4
eta0 = 10 ** (-5)
eta1 = 10 ** (-4)
eta2 = 10 ** (-1)

"""
build action dictionary
"""
position1 = [np.array([[0,0,0,0]]),np.array([[0,0,0,1]]),np.array([[0,0,1,0]]), np.array([[0,0,1,1]]), np.array([[1,0,0,0]]),np.array([[1,0,0,1]]),np.array([[1,0,1,0]]),np.array([[1,0,1,1]]), np.array([[1,1,0,0]]),np.array([[1,1,0,1]]),np.array([[1,1,1,0]]),np.array([[1,1,1,1]])]
position2 = [np.array([[0,0,0,0]]),np.array([[0,0,0,1]]),np.array([[0,0,1,0]]), np.array([[0,0,1,1]]), np.array([[1,0,0,0]]),np.array([[1,0,0,1]]),np.array([[1,0,1,0]]),np.array([[1,0,1,1]]), np.array([[1,1,0,0]]),np.array([[1,1,0,1]]),np.array([[1,1,1,0]]),np.array([[1,1,1,1]])]
position3 = [np.array([[0,0,0,0]]),np.array([[0,0,0,1]]),np.array([[0,0,1,0]]), np.array([[0,0,1,1]]), np.array([[1,0,0,0]]),np.array([[1,0,0,1]]),np.array([[1,0,1,0]]),np.array([[1,0,1,1]]), np.array([[1,1,0,0]]),np.array([[1,1,0,1]]),np.array([[1,1,1,0]]),np.array([[1,1,1,1]])]
combined_arrays = []
for arr1 in position1:
    for arr2 in position2:
        for arr3 in position3:
            combined_arrays.append(np.hstack((arr1, arr2, arr3)))
action_dict = {str(i): combined_arrays[i] for i in range(len(combined_arrays))}
# print(action_dict)
inv_action_dict = {tuple(v[0]): int(k) for k, v in action_dict.items()}

def original(control_trains,d_pre_cut,rho,d_real,l_real,state_n,start_index):
    oser = np.zeros([num_line, control_trains, 2 * max_station, 2], dtype=int)
    for m in range(num_line):
        for s in range(2 * num_station[m]):
            for k in range(1,control_trains):
                for j in range(2):
                    for i in range(control_trains):  # d_pre = a_pre + t_trans if we set t_trans = tau
                        if (d_pre_cut[olin[m, s], i + 1, opla[m, s, j]] <= d_pre_cut[m, k, s]) and (
                                d_pre_cut[olin[m, s], i + 1, opla[m, s, j]] > d_pre_cut[m, k-1, s]):
                            oser[m, k, s, j] = i
    or_n = np.zeros([num_line, control_trains+1, 2 * max_station])
    or_n_depart = np.zeros([num_line, control_trains, 2 * max_station])
    or_n_before = np.zeros([num_line, control_trains, 2 * max_station])
    or_n_after = np.zeros([num_line, control_trains, 2 * max_station])
    or_C = np.zeros([num_line, control_trains, 2 * max_station])
    or_n_arrive = np.zeros([num_line, control_trains, 2 * max_station])
    for m in range(num_line):
        for s in range(2 * num_station[m]):
            or_n[m, 0, s] = state_n[m, s]
            for k in range(control_trains):
                or_n_before[m, k, s] = or_n[m, k, s] + rho[m, k + 1, s] * (d_real[m, start_index[m, s] + k, s] - d_pre_cut[m, k, s])
                or_C[m, k, s] = l_real[m, start_index[m, s] + k, s] * Cmax
                or_n_depart[m, k, s] = min(or_C[m, k, s], or_n_before[m, k, s])
                or_n_after[m, k, s] = or_n_before[m, k, s] - or_n_depart[m, k, s]
                or_n[m, k + 1, s] = or_n[m, k, s] + rho[m, k + 1, s] * (d_pre_cut[m, k + 1, s] - d_pre_cut[m, k, s]) - or_n_depart[m, k, s]
        for s in range(1,2 * num_station[m]):
            for k in range(control_trains):
                or_n_arrive[m,k,s] = or_n_depart[m,k,s-1]

    or_n = np.zeros([num_line, control_trains + 1, 2 * max_station])
    or_n_depart = np.zeros([num_line, control_trains, 2 * max_station])
    or_n_before = np.zeros([num_line, control_trains, 2 * max_station])
    or_n_after = np.zeros([num_line, control_trains, 2 * max_station])
    or_C = np.zeros([num_line, control_trains, 2 * max_station])
    or_n_trans = np.zeros([num_line, control_trains, 2 * max_station])
    J_original = 0
    for m in range(num_line):
        for s in range(2 * num_station[m]):
            or_n[m, 0, s] = state_n[m, s]
            for k in range(control_trains):
                or_n_trans[m,k,s] = trans_rate*or_n_arrive[olin[m,s],oser[m,k,s,0],opla[m,s,0]] + trans_rate*or_n_arrive[olin[m,s],oser[m,k,s,1],opla[m,s,1]]
                or_n_before[m, k, s] = or_n[m, k, s] + rho[m, k + 1, s] * (d_real[m, start_index[m,s] + k, s] - d_pre_cut[m, k, s]) + or_n_trans[m,k,s]
                or_C[m, k, s] = l_real[m, start_index[m,s] + k, s] * Cmax
                or_n_depart[m, k, s] = min(or_C[m, k, s], or_n_before[m, k, s])
                or_n_after[m, k, s] = or_n_before[m, k, s] - or_n_depart[m, k, s]
                or_n[m, k + 1, s] = or_n[m, k, s] + rho[m, k + 1, s] * (d_pre_cut[m, k + 1, s] - d_pre_cut[m, k, s]) + or_n_trans[m,k,s] - or_n_depart[m, k, s]
            J_original=J_original+sum(eta0*or_n[m,k,s]*(d_pre_cut[m, k+1,s] - d_pre_cut[m, k,s]) + eta1*or_n_after[m, k,s]*(d_pre_cut[m, k+1,s]-d_pre_cut[m, k,s]) + eta2*l_real[m,start_index[m, s]+k, s]*E_regular[m, s] for k in range(2,control_trains))
    return J_original

def mdl_feasible(mdl):
    """
    checks whether the gurobi mdl is feasible
    """
    if mdl.status == 2 or mdl.status == 9 or mdl.status == 13 or mdl.status==11: # 2:optimal, 9: timelimit, 13: suboptimal, 11: interrupted
        feas = True
        return feas
    else:
        feas = False
        return feas

#added by Caio
#function for early termination: if mipgap does not decrease x% in y seconds, then terminate
epsilon_to_compare_gap = 0.005 # 0.5%
time_from_best = 10 # seconds
def callback(model, where):
    if where == GRB.Callback.MIP:
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        incumbent = model.cbGet(GRB.Callback.MIP_OBJBST)
        bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        gap = abs((bound - incumbent) / incumbent)

        # If an incumbent solution is found
        if incumbent != GRB.INFINITY:
            # If the current gap is different from the previous gap, update
            # the time_to_best and the gap
            if (model._gap - gap) > epsilon_to_compare_gap:
                model._time_to_best = runtime
                model._gap = gap
            # If the current gap is the same as the previous gap for more than
            # the time_from_best, terminate
            elif runtime - model._time_to_best > time_from_best:
                model.terminate()

def gurobi_milp(control_trains,d_pre_cut,rho,state_a,state_d,state_r,state_l,state_y,state_n,depot_real, mipgap, log, timelimit, n_threads):
    """
    function of gurobi optimization
    control_trains:         horizon
    """
    mdl = Model('OPT')
    mdl.Params.LogToConsole = log
    mdl.Params.Threads = n_threads
    mdl.Params.TimeLimit = max(1,timelimit)
    mdl.Params.MIPGap = mipgap # 0.1%
    # mdl.Params.MIPFocus = 2 # default is 0, it was set to 2
    # mdl.Params.Cuts = -1 # default is -1, it was set to 2
    # mdl.Params.NonConvex = 2
    oser = np.zeros([num_line, control_trains, 2 * max_station, 2], dtype=int)
    for m in range(num_line):
        for s in range(2 * num_station[m]):
            for k in range(1, control_trains):
                for j in range(2):
                    for i in range(control_trains):  # d_pre = a_pre + t_trans if we set t_trans = tau
                        if (d_pre_cut[olin[m, s], i + 1, opla[m, s, j]] <= d_pre_cut[m, k, s]) and (d_pre_cut[olin[m, s], i + 1, opla[m, s, j]] > d_pre_cut[m, k-1, s]):
                            oser[m, k, s, j] = i
    # decision variables
    d = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='departuretime')
    a = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='arrivaltime')
    r = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='runningtime')
    tau = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='dwelltime')
    # tau_add = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='dwelltime_add')
    l = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.INTEGER,name='composition')
    y = mdl.addVars(num_line,control_trains,2*max_station, lb=-l_max, ub=l_max, vtype=GRB.INTEGER,name='delta_composition')
    # o = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.INTEGER,name='delta_composition')
    # gamma = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.BINARY,name='delta_composition')
    # sign_o = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.BINARY,name='sign_composition')
    xi = mdl.addVars(num_line,control_trains,control_trains+1, vtype=GRB.BINARY,name='depot_wagon')

    n = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='waiting passengers')
    n_arrive = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='arrive passengers')
    n_trans = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='arriving transfer passengers')
    n_depart = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='departing passengers')
    n_before = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='passengers before departure')
    n_after = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='passengers after departure')
    C = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='Capacity')
    # objective
    mdl.setObjective(quicksum(eta0*n[0,k,s]*(d_pre_cut[0,k+1,s]-d_pre_cut[0,k,s])+eta1*n_after[0,k,s]*(d_pre_cut[0,k+1,s]-d_pre_cut[0,k,s]) + eta2*l[0,k,s]*E_regular[0,s]
                               for k in range(2,control_trains) for s in range(2 * num_station[0])) +
                     quicksum(eta0 * n[1, k, s] * (d_pre_cut[1, k+1, s] - d_pre_cut[1, k, s]) + eta1 * n_after[1, k, s] * (d_pre_cut[1, k + 1, s] - d_pre_cut[1, k, s]) + eta2*l[1, k, s] * E_regular[1, s]
                               for k in range(2,control_trains) for s in range(2 * num_station[1])) +
                     quicksum(eta1 * n[2, k, s] * (d_pre_cut[2, k+1, s] - d_pre_cut[2, k, s]) + eta1 * n_after[2, k, s] * (d_pre_cut[2, k + 1, s] - d_pre_cut[2, k, s]) + eta2*l[2, k, s] * E_regular[2, s]
                               for k in range(2,control_trains) for s in range(2 * num_station[2])))
    # constraints
    for m in range(num_line):
        mdl.addConstrs(a[m,k,s] == state_a[m,1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s] == state_d[m,1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s] == state_r[m,1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(l[m,k,s] == state_l[m,1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(y[m,k,s] == state_y[m,1+k,s] for k in range(2) for s in range(2*num_station[m]))
        for s in range(1,2*num_station[m]):
            for k in range(1,differ[m,s]):
                mdl.addConstr(a[m,k,s]==state_d[m,1+k-differ[m,s],s-1]+state_r[m,1+k-differ[m,s],s-1])
                mdl.addConstr(l[m,k,s]==state_l[m,1+k-differ[m,s],s-1] + sigma[m,s]*y[m,k,s])
            for k in range(differ[m,s],control_trains):
                mdl.addConstr(a[m,k,s] == d[m,k-differ[m,s],s-1] + r[m,k-differ[m,s],s-1])
                mdl.addConstr(l[m,k,s] == l[m,k-differ[m,s],s-1] + sigma[m,s]*y[m,k,s])
            for k in range(control_trains,control_trains+differ[m,s]):
                mdl.addConstr(d[m,k-differ[m,s], s-1] + r[m,k-differ[m,s], s-1] == state_a[m,1+k,s])
            # mdl.addConstr(l[round(k-differ[s]), s-1] + sigma[s]*uy[round(start_train[s]+k),s] == ul[round(start_train[s]+k),s])
        # mdl.addConstrs(a[k,s] == ud[round(start_train[s]+1+k-differ[s]),s-1] + ur[round(start_train[s]+1+k-differ[s]),s-1] for k in range(round(differ[s])) for s in range(1,2*num_station))
        # mdl.addConstrs(a[k,s] == d[round(k-differ[s]),s-1] + r[round(k-differ[s]),s-1] for k in range(round(differ[s]),control_trains) for s in range(1,2*num_station))
        # mdl.addConstrs(o[m,k,s] - y[m,k,s] >= 0 for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(o[m,k,s] - y[m,k,s] <= 2*l_max*(1-gamma[m,k,s]) for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(o[m,k,s] + y[m,k,s] >= 0 for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(o[m,k,s] + y[m,k,s] <= 2*l_max*gamma[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(o[m,k, s] <= sign_o[m,k,s]*l_max for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(o[m,k, s] >= epsilon + (1-sign_o[m,k,s])*(-l_max - epsilon) for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(tau_add[m,k, s] == sign_o[m,k,s]*t_constant for k in range(control_trains) for s in range(2*num_station[m]))

        mdl.addConstrs(d[m,k,s]>=d_pre_cut[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s]<=d_pre_cut[m,k+1,s] - epsilon for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(a[m,k+1,s]>=d[m,k,s]+h_min for k in range(control_trains-1) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s]==a[m,k,s]+tau[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,control_trains-1,s]+h_min <= state_a[m,round(1+control_trains),s] for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s]>=r_min[m,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s]<=r_max[m,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(l[m,k,0]==y[m,k,0] for k in range(control_trains))
        # mdl.addConstrs(y[m,k, num_station[m] - 1] == 0 for k in range(control_trains))
        mdl.addConstrs(y[m,k,2*num_station[m]-1]==-l[m,k-differ[m,2*num_station[m]-1],2*num_station[m]-2] for k in range(1, control_trains))
        mdl.addConstrs(l[m,k,s]>=l_min for k in range(control_trains) for s in range(2*num_station[m]-1))
        mdl.addConstrs(l[m,k,s]<=l_max for k in range(control_trains) for s in range(2*num_station[m]-1))
        # mdl.addConstrs(tau[m,k,s]>=tau_min+sigma[m,s]*tau_add[m,k,s] for k in range(control_trains) for s in range(1,2*num_station[m]-1))
        mdl.addConstrs(tau[m,k,s]>=tau_min for k in range(control_trains) for s in range(2*num_station[m]))

        mdl.addConstrs(d[m,i,2*num_station[m]-1]+t_roll-d[m,k,0]<=(1-xi[m,k,i])*(Mt-d[m,k,0]) for i in range(control_trains) for k in range(2,control_trains))
        mdl.addConstrs(d[m,i,2*num_station[m]-1]+t_roll-d[m,k,0]>=epsilon+xi[m,k,i]*(mt-d[m,k,0]-epsilon) for i in range(control_trains) for k in range(2,control_trains))
        mdl.addConstrs(xi[m,k,i]<=xi[m,k,i-1] for i in range(1,control_trains) for k in range(2,control_trains))
        mdl.addConstrs(xi[m,k,i]>=xi[m,k-1,i] for i in range(control_trains) for k in range(2,control_trains))
        mdl.addConstrs(quicksum(xi[m,k,i]*y[m,i,2*num_station[m]-1] for i in range(control_trains)) + quicksum(y[m,i,0] for i in range(2,k+1)) <= depot_real[m] for k in range(2,control_trains-1))
        # mdl.addConstrs(quicksum(y[m,i,num_station[m]] for i in range(2,k+1)) <= depot_real[m,num_station[m]-1] for k in range(2,control_trains-1))

        mdl.addConstrs(n[m,0,s]==state_n[m,s] for s in range(2*num_station[m]))
        mdl.addConstrs(n[m,k+1,s]==n[m,k,s]+rho[m,k+1,s]*(d_pre_cut[m,k+1,s]-d_pre_cut[m,k,s])+n_trans[m,k,s]-n_depart[m,k,s] for k in range(control_trains-1) for s in range(2*num_station[m]))
        mdl.addConstrs(n_before[m,k,s]==n[m,k,s]+rho[m,k+1,s]*(d[m,k,s]-d_pre_cut[m,k,s])+n_trans[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(n_depart[m,k,s]<=C[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(n_depart[m,k,s]<=n_before[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(n_depart[k,s]==min(C[k,s],n_before[k,s]) for k in range(control_trains) for s in range(2*num_station))
        mdl.addConstrs(n_arrive[m,k,s]==n_depart[m,k-differ[m,s],s-1] for k in range(2,control_trains) for s in range(1,2 * num_station[m]))
        mdl.addConstrs(n_trans[m,k,s]==trans_rate*n_arrive[olin[m,s],oser[m,k,s,0],opla[m,s,0]] + trans_rate*n_arrive[olin[m,s],oser[m,k,s,1],opla[m,s,1]] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(C[m,k,s]==l[m,k,s]*Cmax for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(n_after[m,k,s]==n_before[m,k,s]-n_depart[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        # print(depot_real[m,0])
        
    # start_time = time.time()
    # tune parameter of gurobi solver
    # solving
    mdl.optimize()
    mdl._Runtime = mdl.Runtime
    # mdl.remove(C1)
    # mdl.update()
    # end_time = time.time()
    # gurobi_runtime = end_time - start_time
    # print('gurobi runing time = %f seconds' % gurobi_runtime)
    a_values = np.zeros([num_line, control_trains, 2 * max_station])
    d_values = np.zeros([num_line, control_trains, 2 * max_station])
    r_values = np.zeros([num_line, control_trains, 2 * max_station])
    l_values = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    y_values = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    xi_values = np.zeros([num_line, control_trains, control_trains+1], dtype=int)
    delta_minlp = np.zeros([control_trains - 2, 4*num_line], dtype=int)
    n_values = np.zeros([num_line, control_trains, 2 * max_station])
    n_after_values = np.zeros([num_line, control_trains, 2 * max_station])
    # sign_o_values = np.zeros([num_line,control_trains, 2 * max_station])
    feas = mdl_feasible(mdl)
    if feas:
        # Print the objective function value
        # print(f"Optimal Objective Value: {mdl.objVal}")
        # obtain delta from the minlp solution
        link = np.zeros([num_line, control_trains, 2])
        index = 0
        for m in range(num_line):
            for k in range(1, control_trains):
                for i in range(control_trains+1):
                    if (d_pre_cut[m, k, 0] < d_pre_cut[m, i + 1, 2*num_station[m]-1] + t_roll) and (d_pre_cut[m,k + 1, 0] > d_pre_cut[m,i, 2*num_station[m]-1] + t_roll):
                        link[m, k, index] = i
                        index = round(index + 1)
                index = 0
            for k in range(2, control_trains):
                delta_minlp[k - 2, 0+4*m] = round(xi[m, k, link[m, k, 0]].x)
                # if link[m, k, 0, 1] <= control_trains - 1:
                delta_minlp[k - 2, 1+4*m] = round(xi[m, k, link[m, k, 1]].x)
            for k in range(2, control_trains):
                if round(l[m, k, 0].x) == 1:
                    delta_minlp[k - 2, 2+4*m] = 0
                    delta_minlp[k - 2, 3+4*m] = 0
                elif round(l[m, k, 0].x) == 2:
                    delta_minlp[k - 2, 2+4*m] = 1
                    delta_minlp[k - 2, 3+4*m] = 0
                elif round(l[m, k, 0].x) == 3:
                    delta_minlp[k - 2, 2+4*m] = 0
                    delta_minlp[k - 2, 3+4*m] = 1
                elif round(l[m, k, 0].x) == 4:
                    delta_minlp[k - 2, 2+4*m] = 1
                    delta_minlp[k - 2, 3+4*m] = 1
            for s in range(2 * num_station[m]):
                for k in range(control_trains):
                    a_values[m, k, s] = a[m, k, s].x
                    d_values[m, k, s] = d[m, k, s].x
                    r_values[m, k, s] = r[m, k, s].x
                    l_values[m, k, s] = round(l[m, k, s].x)
                    y_values[m, k, s] = round(y[m, k, s].x)
            for k in range(control_trains):
                for i in range(control_trains+1):
                    xi_values[m,k, i] = round(xi[m,k, i].x)
        n_values = np.array([v.X for v in n.values()]).reshape(num_line,control_trains, 2 * max_station)
        n_after_values = np.array([v.X for v in n_after.values()]).reshape(num_line,control_trains, 2 * max_station)
        # sign_o_values = np.array([v.X for v in sign_o.values()]).reshape(num_line,control_trains, 2 * max_station)
    else:
        print("minlp: optimization did not converge to an optimal solution.")

    return a_values, d_values, r_values, l_values, y_values, delta_minlp, n_values, n_after_values, mdl

def gurobi_minlp(control_trains,d_pre_cut,rho,state_a,state_d,state_r,state_l,state_y,state_n,depot_real, mipgap, log, timelimit, early_term, warm_start, n_threads):
    """
    function of gurobi optimization
    control_trains:         horizon
    """
    mdl = Model('OPT')
    mdl.Params.LogToConsole = log
    mdl.Params.Threads = n_threads
    mdl.Params.TimeLimit = max(1,timelimit)
    mdl.Params.MIPGap = mipgap # 0.1%
    mdl.Params.MIPFocus = 2 # default is 0
    mdl.Params.NonConvex = 2 # default is -1
    timelimit_warmstart = timelimit*0.1 # time for milp to warm-start
    oser = np.zeros([num_line, control_trains, 2 * max_station, 2], dtype=int)
    for m in range(num_line):
        for s in range(2 * num_station[m]):
            for k in range(1, control_trains):
                for j in range(2):
                    for i in range(control_trains):  # d_pre = a_pre + t_trans if we set t_trans = tau
                        if (d_pre_cut[olin[m, s], i + 1, opla[m, s, j]] <= d_pre_cut[m, k, s]) and (d_pre_cut[olin[m, s], i + 1, opla[m, s, j]] > d_pre_cut[m, k-1, s]):
                            oser[m, k, s, j] = i
    # decision variables
    d = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='departuretime')
    a = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='arrivaltime')
    r = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='runningtime')
    tau = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='dwelltime')
    # tau_add = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='dwelltime_add')
    l = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.INTEGER,name='composition')
    y = mdl.addVars(num_line,control_trains,2*max_station, lb=-l_max, ub=l_max, vtype=GRB.INTEGER,name='delta_composition')
    # o = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.INTEGER,name='delta_composition')
    # gamma = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.BINARY,name='delta_composition')
    # sign_o = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.BINARY,name='sign_composition')
    xi = mdl.addVars(num_line,control_trains,control_trains+1, vtype=GRB.BINARY,name='depot_wagon')

    n = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='waiting passengers')
    n_arrive = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='arrive passengers')
    n_trans = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='arriving transfer passengers')
    n_depart = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='departing passengers')
    n_before = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='passengers before departure')
    n_after = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='passengers after departure')
    C = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='Capacity')
    # objective
    mdl.setObjective(quicksum(eta1*n[0,k,s]*(d[0,k,s]-d_pre_cut[0,k,s])+eta1*n_after[0,k,s]*(d_pre_cut[0,k+1,s]-d[0,k,s]) + eta2*l[0,k,s]*E_regular[0,s]
                              for k in range(2,control_trains) for s in range(2 * num_station[0])) +
                     quicksum(eta1 * n[1, k, s] * (d[1, k, s] - d_pre_cut[1, k, s]) + eta1 * n_after[1, k, s] * (d_pre_cut[1, k + 1, s] - d[1, k, s]) + eta2*l[1, k, s] * E_regular[1, s]
                              for k in range(2,control_trains) for s in range(2 * num_station[1])) +
                     quicksum(eta1 * n[2, k, s] * (d[2, k, s] - d_pre_cut[2, k, s]) + eta1 * n_after[2, k, s] * (d_pre_cut[2, k + 1, s] - d[2, k, s]) + eta2*l[2, k, s] * E_regular[2, s]
                              for k in range(2,control_trains) for s in range(2 * num_station[2])))
    # constraints
    for m in range(num_line):
        mdl.addConstrs(a[m,k,s] == state_a[m,1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s] == state_d[m,1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s] == state_r[m,1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(l[m,k,s] == state_l[m,1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(y[m,k,s] == state_y[m,1+k,s] for k in range(2) for s in range(2*num_station[m]))
        for s in range(1,2*num_station[m]):
            for k in range(1,differ[m,s]):
                mdl.addConstr(a[m,k,s]==state_d[m,1+k-differ[m,s],s-1]+state_r[m,1+k-differ[m,s],s-1])
                mdl.addConstr(l[m,k,s]==state_l[m,1+k-differ[m,s],s-1] + sigma[m,s]*y[m,k,s])
            for k in range(differ[m,s],control_trains):
                mdl.addConstr(a[m,k,s] == d[m,k-differ[m,s],s-1] + r[m,k-differ[m,s],s-1])
                mdl.addConstr(l[m,k,s] == l[m,k-differ[m,s],s-1] + sigma[m,s]*y[m,k,s])
            for k in range(control_trains,control_trains+differ[m,s]):
                mdl.addConstr(d[m,k-differ[m,s], s-1] + r[m,k-differ[m,s], s-1] == state_a[m,1+k,s])
            # mdl.addConstr(l[round(k-differ[s]), s-1] + sigma[s]*uy[round(start_train[s]+k),s] == ul[round(start_train[s]+k),s])
        # mdl.addConstrs(a[k,s] == ud[round(start_train[s]+1+k-differ[s]),s-1] + ur[round(start_train[s]+1+k-differ[s]),s-1] for k in range(round(differ[s])) for s in range(1,2*num_station))
        # mdl.addConstrs(a[k,s] == d[round(k-differ[s]),s-1] + r[round(k-differ[s]),s-1] for k in range(round(differ[s]),control_trains) for s in range(1,2*num_station))
        # mdl.addConstrs(o[m,k,s] - y[m,k,s] >= 0 for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(o[m,k,s] - y[m,k,s] <= 2*l_max*(1-gamma[m,k,s]) for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(o[m,k,s] + y[m,k,s] >= 0 for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(o[m,k,s] + y[m,k,s] <= 2*l_max*gamma[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(o[m,k, s] <= sign_o[m,k,s]*l_max for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(o[m,k, s] >= epsilon + (1-sign_o[m,k,s])*(-l_max - epsilon) for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(tau_add[m,k, s] == sign_o[m,k,s]*t_constant for k in range(control_trains) for s in range(2*num_station[m]))

        mdl.addConstrs(d[m,k,s]>=d_pre_cut[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s]<=d_pre_cut[m,k+1,s] - epsilon for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(a[m,k+1,s]>=d[m,k,s]+h_min for k in range(control_trains-1) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s]==a[m,k,s]+tau[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,control_trains-1,s]+h_min <= state_a[m,round(1+control_trains),s] for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s]>=r_min[m,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s]<=r_max[m,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(l[m,k,0]==y[m,k,0] for k in range(control_trains))
        # mdl.addConstrs(y[m,k, num_station[m] - 1] == 0 for k in range(control_trains))
        mdl.addConstrs(y[m,k,2*num_station[m]-1]==-l[m,k-differ[m,2*num_station[m]-1],2*num_station[m]-2] for k in range(1, control_trains))
        mdl.addConstrs(l[m,k,s]>=l_min for k in range(control_trains) for s in range(2*num_station[m]-1))
        mdl.addConstrs(l[m,k,s]<=l_max for k in range(control_trains) for s in range(2*num_station[m]-1))
        # mdl.addConstrs(tau[m,k,s]>=tau_min+sigma[m,s]*tau_add[m,k,s] for k in range(control_trains) for s in range(1,2*num_station[m]-1))
        mdl.addConstrs(tau[m,k,s]>=tau_min for k in range(control_trains) for s in range(2*num_station[m]))

        mdl.addConstrs(d[m,i,2*num_station[m]-1]+t_roll-d[m,k,0]<=(1-xi[m,k,i])*(Mt-d[m,k,0]) for i in range(control_trains) for k in range(2,control_trains))
        mdl.addConstrs(d[m,i,2*num_station[m]-1]+t_roll-d[m,k,0]>=epsilon+xi[m,k,i]*(mt-d[m,k,0]-epsilon) for i in range(control_trains) for k in range(2,control_trains))
        mdl.addConstrs(xi[m,k,i]<=xi[m,k,i-1] for i in range(1,control_trains) for k in range(2,control_trains))
        mdl.addConstrs(xi[m,k,i]>=xi[m,k-1,i] for i in range(control_trains) for k in range(2,control_trains))
        mdl.addConstrs(quicksum(xi[m,k,i]*y[m,i,2*num_station[m]-1] for i in range(control_trains)) + quicksum(y[m,i,0] for i in range(2,k+1)) <= depot_real[m] for k in range(2,control_trains-1))
        # mdl.addConstrs(quicksum(y[m,i,num_station[m]] for i in range(2,k+1)) <= depot_real[m,num_station[m]-1] for k in range(2,control_trains-1))

        mdl.addConstrs(n[m,0,s]==state_n[m,s] for s in range(2*num_station[m]))
        mdl.addConstrs(n[m,k+1,s]==n[m,k,s]+rho[m,k+1,s]*(d_pre_cut[m,k+1,s]-d_pre_cut[m,k,s])+n_trans[m,k,s]-n_depart[m,k,s] for k in range(control_trains-1) for s in range(2*num_station[m]))
        mdl.addConstrs(n_before[m,k,s]==n[m,k,s]+rho[m,k+1,s]*(d[m,k,s]-d_pre_cut[m,k,s])+n_trans[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(n_depart[m,k,s]<=C[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(n_depart[m,k,s]<=n_before[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(n_depart[k,s]==min(C[k,s],n_before[k,s]) for k in range(control_trains) for s in range(2*num_station))
        mdl.addConstrs(n_arrive[m,k,s]==n_depart[m,k-differ[m,s],s-1] for k in range(2,control_trains) for s in range(1,2 * num_station[m]))
        mdl.addConstrs(n_trans[m,k,s]==trans_rate*n_arrive[olin[m,s],oser[m,k,s,0],opla[m,s,0]] + trans_rate*n_arrive[olin[m,s],oser[m,k,s,1],opla[m,s,1]] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(C[m,k,s]==l[m,k,s]*Cmax for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(n_after[m,k,s]==n_before[m,k,s]-n_depart[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        # print(depot_real[m,0])
        
    # start_time = time.time()
    # tune parameter of gurobi solver
    # solving
       
    #added by Caio
    mdl.update()
    if warm_start == 1:
        mdl.NumStart = 1
        mdl_milp = gurobi_milp(control_trains,d_pre_cut,rho,state_a,state_d,state_r,state_l,state_y,state_n,depot_real,mipgap,0,timelimit_warmstart,n_threads)[-1]
        # print(mdl_milp.Runtime)
        # for v in mdl.getVars():
        # for i in range(len(mdl.getVars())):
        #     mdl.getVars()[i].Start = mdl_milp.getVars()[i].x
        # print('warm start done.', mdl_milp.getVars()[i].x)
        # list_milp_solution = []
        # for v in mdl_milp.getVars():
        #     list_milp_solution.append(v.x)
        if mdl_milp.SolCount >= 1:
            list_milp_solution = mdl_milp.getAttr("X", mdl_milp.getVars())
            mdl.setAttr("Start", mdl.getVars(), list_milp_solution)    
        mdl.Params.TimeLimit = max(2,timelimit - mdl_milp.Runtime)
    
    #added by Caio
    if early_term == 1:
        mdl._gap, mdl._time_to_best = 1.0, float("inf")
        mdl.optimize(callback)
    elif early_term == 0:
        mdl.optimize()
        
    #added by Caio
    if warm_start==1:
        mdl._Runtime = mdl.Runtime+mdl_milp.Runtime
    else:
        mdl._Runtime = mdl.Runtime
        
    # mdl.remove(C1)
    # mdl.update()
    # end_time = time.time()
    # gurobi_runtime = end_time - start_time
    # print('gurobi runing time = %f seconds' % gurobi_runtime)
    a_values = np.zeros([num_line, control_trains, 2 * max_station])
    d_values = np.zeros([num_line, control_trains, 2 * max_station])
    r_values = np.zeros([num_line, control_trains, 2 * max_station])
    l_values = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    y_values = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    xi_values = np.zeros([num_line, control_trains, control_trains+1], dtype=int)
    delta_minlp = np.zeros([control_trains - 2, 4*num_line], dtype=int)
    n_values = np.zeros([num_line, control_trains, 2 * max_station])
    n_after_values = np.zeros([num_line, control_trains, 2 * max_station])
    # sign_o_values = np.zeros([num_line,control_trains, 2 * max_station])
    feas = mdl_feasible(mdl)
    if feas:
        # Print the objective function value
        # print(f"Optimal Objective Value: {mdl.objVal}")
        # obtain delta from the minlp solution
        link = np.zeros([num_line, control_trains, 2])
        index = 0
        for m in range(num_line):
            for k in range(1, control_trains):
                for i in range(control_trains+1):
                    if (d_pre_cut[m, k, 0] < d_pre_cut[m, i + 1, 2*num_station[m]-1] + t_roll) and (d_pre_cut[m, k + 1, 0] > d_pre_cut[m, i, 2*num_station[m]-1] + t_roll):
                        link[m, k, index] = i
                        index = round(index + 1)
                index = 0
            for k in range(2, control_trains):
                delta_minlp[k - 2, 0+4*m] = round(xi[m, k, link[m, k, 0]].x)
                # if link[m, k, 0, 1] <= control_trains - 1:
                delta_minlp[k - 2, 1+4*m] = round(xi[m, k, link[m, k, 1]].x)
            for k in range(2, control_trains):
                if round(l[m, k, 0].x) == 1:
                    delta_minlp[k - 2, 2+4*m] = 0
                    delta_minlp[k - 2, 3+4*m] = 0
                elif round(l[m, k, 0].x) == 2:
                    delta_minlp[k - 2, 2+4*m] = 1
                    delta_minlp[k - 2, 3+4*m] = 0
                elif round(l[m, k, 0].x) == 3:
                    delta_minlp[k - 2, 2+4*m] = 0
                    delta_minlp[k - 2, 3+4*m] = 1
                elif round(l[m, k, 0].x) == 4:
                    delta_minlp[k - 2, 2+4*m] = 1
                    delta_minlp[k - 2, 3+4*m] = 1
            for s in range(2 * num_station[m]):
                for k in range(control_trains):
                    a_values[m, k, s] = a[m, k, s].x
                    d_values[m, k, s] = d[m, k, s].x
                    r_values[m, k, s] = r[m, k, s].x
                    l_values[m, k, s] = round(l[m, k, s].x)
                    y_values[m, k, s] = round(y[m, k, s].x)
            for k in range(control_trains):
                for i in range(control_trains+1):
                    xi_values[m,k, i] = round(xi[m,k, i].x)
        n_values = np.array([v.X for v in n.values()]).reshape(num_line,control_trains, 2 * max_station)
        n_after_values = np.array([v.X for v in n_after.values()]).reshape(num_line,control_trains, 2 * max_station)
        # sign_o_values = np.array([v.X for v in sign_o.values()]).reshape(num_line,control_trains, 2 * max_station)
    else:
        print("minlp: optimization did not converge to an optimal solution.")

    return a_values, d_values, r_values, l_values, y_values, delta_minlp, n_values, n_after_values, mdl


"""
gurobi_qp: xi is not preprocessed, and other integer variables are preprocessed
"""
def gurobi_lp(control_trains,d_pre_cut,rho,state_a,state_d,state_r,state_l,state_y,state_n,depot_real,delta,mipgap,log,timelimit,n_threads):
    """
    function of gurobi optimization
    control_trains:         horizon
    """
    mdl = Model('OPT')
    mdl.Params.LogToConsole = log
    mdl.Params.Threads = n_threads
    mdl.Params.TimeLimit = timelimit
    mdl.Params.MIPGap = mipgap # 0.1%
    # mdl.Params.MIPFocus = 2 # default is 0, it was set to 2
    # mdl.Params.Cuts = -1 # default is -1, it was set to 2
    # mdl.Params.NonConvex = 2
    
    oser = np.zeros([num_line, control_trains, 2 * max_station, 2], dtype=int)
    for m in range(num_line):
        for s in range(2 * num_station[m]):
            for k in range(1,control_trains):
                for j in range(2):
                    for i in range(control_trains):  # d_pre = a_pre + t_trans if we set t_trans = tau
                        if (d_pre_cut[olin[m, s], i + 1, opla[m, s, j]] <= d_pre_cut[m, k, s]) and (d_pre_cut[olin[m, s], i + 1, opla[m, s, j]] > d_pre_cut[m, k-1, s]):
                            oser[m, k, s, j] = i
    l = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    y = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    # sign_o = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    for m in range(num_line):
        for s in range(2 * num_station[m]):
            for k in range(2):
                l[m, k, s] = state_l[m, 1 + k, s]
                y[m, k, s] = state_y[m, 1 + k, s]
        for k in range(2, control_trains):
            l[m, k, 0] = round(1 + delta[k - 2, 2+4*m] + 2 * delta[k - 2, 3+4*m])
            y[m, k, 0] = l[m, k, 0]
        for s in range(1, 2 * num_station[m] - 1):
            for k in range(2,control_trains):
                l[m, k, s] = l[m, k - differ[m,s], s - 1] + sigma[m,s] * y[m, k, s]
        for k in range(2,control_trains):
            y[m, k, 2 * num_station[m] - 1] = -l[m, k - differ[m, 2 * num_station[m] - 1], 2 * num_station[m] - 2]
        # for s in range(2 * num_station[m]):
        #     for k in range(control_trains):
        #         if round(y[m, k, s]) == 0:
        #             sign_o[m, k, s] = 0
        #         else:
        #             sign_o[m, k, s] = 1
    # decision variables
    d = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='departuretime')
    a = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='arrivaltime')
    r = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='runningtime')
    tau = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='dwelltime')
    # tau_add = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='dwelltime_add')
    xi = mdl.addVars(num_line,control_trains,control_trains+1, vtype=GRB.BINARY,name='depot_wagon')

    n = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='waiting passengers')
    n_arrive = mdl.addVars(num_line, control_trains, 2 * max_station, vtype=GRB.CONTINUOUS, name='arrive passengers')
    n_trans = mdl.addVars(num_line, control_trains, 2 * max_station, vtype=GRB.CONTINUOUS, name='arriving transfer passengers')
    n_depart = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='departing passengers')
    n_before = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='passengers before departure')
    n_after = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='passengers after departure')
    C = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='Capacity')
    # objective
    mdl.setObjective(quicksum(eta0 * n[0, k, s] * (d_pre_cut[0, k+1, s] - d_pre_cut[0, k, s]) + eta1 * n_after[0, k, s] * (d_pre_cut[0, k + 1, s] - d_pre_cut[0, k, s]) + eta2 * l[0, k, s] * E_regular[0, s]
                              for k in range(2, control_trains) for s in range(2 * num_station[0])) +
                     quicksum(eta0 * n[1, k, s] * (d_pre_cut[1, k+1, s] - d_pre_cut[1, k, s]) + eta1 * n_after[1, k, s] * (d_pre_cut[1, k + 1, s] - d_pre_cut[1, k, s]) + eta2 * l[1, k, s] * E_regular[1, s]
                              for k in range(2, control_trains) for s in range(2 * num_station[1])) +
                     quicksum(eta0 * n[2, k, s] * (d_pre_cut[2, k+1, s] - d_pre_cut[2, k, s]) + eta1 * n_after[2, k, s] * (d_pre_cut[2, k + 1, s] - d_pre_cut[2, k, s]) + eta2 * l[2, k, s] * E_regular[2, s]
                              for k in range(2, control_trains) for s in range(2 * num_station[2])))
    # fixed integer from delta
    link = np.zeros([num_line, control_trains, 2])
    index = 0
    for m in range(num_line):
        for k in range(1, control_trains):
            for i in range(control_trains+1):
                if (d_pre_cut[m, k, 0] < d_pre_cut[m, i + 1, 2 * num_station[m] - 1] + t_roll) and (
                        d_pre_cut[m, k + 1, 0] > d_pre_cut[m, i, 2 * num_station[m] - 1] + t_roll):
                    link[m, k, index] = i
                    index = round(index + 1)
            index = 0
    # xi = np.zeros([num_line, control_trains, control_trains + 1, 2 * max_station, 2 * max_station])
    for m in range(num_line):
        for k in range(2, control_trains):
            mdl.addConstr(xi[m, k, link[m, k, 0]] == delta[k - 2, 0+4*m])
            # if link[m, k, 0, 1] <= control_trains - 1:
            mdl.addConstr(xi[m, k, link[m, k, 1]] == delta[k - 2, 1+4*m])

    # constraints
    for m in range(num_line):
        mdl.addConstrs(a[m,k,s] == state_a[m, 1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s] == state_d[m, 1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s] == state_r[m, 1+k,s] for k in range(2) for s in range(2*num_station[m]))

        for s in range(1,2*num_station[m]):
            for k in range(1,differ[m,s]):
                mdl.addConstr(a[m,k,s]==state_d[m,1+k-differ[m,s],s-1]+state_r[m,1+k-differ[m,s],s-1])
            for k in range(differ[m,s],control_trains):
                mdl.addConstr(a[m,k,s] == d[m,k-differ[m,s],s-1] + r[m,k-differ[m,s],s-1])
            for k in range(control_trains,control_trains+differ[m,s]):
                mdl.addConstr(d[m,k-differ[m,s], s-1] + r[m,k-differ[m,s], s-1] == state_a[m,1+k,s])
                # mdl.addConstr(l[round(k-differ[s]), s-1] + sigma[s]*uy[round(start_train[s]+k),s] == ul[round(start_train[s]+k),s])
        # mdl.addConstrs(a[k,s] == ud[round(start_train[s]+1+k-differ[s]),s-1] + ur[round(start_train[s]+1+k-differ[s]),s-1] for k in range(round(differ[s])) for s in range(1,2*num_station))
        # mdl.addConstrs(a[k,s] == d[round(k-differ[s]),s-1] + r[round(k-differ[s]),s-1] for k in range(round(differ[s]),control_trains) for s in range(1,2*num_station))

        # mdl.addConstrs(tau_add[m, k, s] == sign_o[m, k,s]*t_constant for k in range(control_trains) for s in range(2*num_station[m]))

        mdl.addConstrs(d[m,k,s]>=d_pre_cut[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s]<=d_pre_cut[m,k+1,s] - epsilon for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(a[m,k+1,s]>=d[m,k,s]+h_min for k in range(control_trains-1) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s]==a[m,k,s]+tau[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,control_trains-1,s]+h_min <= state_a[m,round(1+control_trains),s] for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s]>=r_min[m,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s]<=r_max[m,s] for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(tau[m,k,s]>=tau_min+sigma[m,s]*tau_add[m,k,s] for k in range(control_trains) for s in range(1,2*num_station[m]-1))
        mdl.addConstrs(tau[m,k,s]>=tau_min for k in range(control_trains) for s in range(2*num_station[m]))

        mdl.addConstrs(d[m,i,2*num_station[m]-1]+t_roll-d[m,k,0] <= (1-xi[m,k,i])*(Mt-d[m,k,0]) for i in range(control_trains) for k in range(2, control_trains))
        mdl.addConstrs(d[m,i,2*num_station[m]-1]+t_roll-d[m,k,0] >= epsilon+xi[m,k,i]*(mt-d[m,k,0] - epsilon) for i in range(control_trains) for k in range(2, control_trains))
        mdl.addConstrs(xi[m, k, i] <= xi[m, k, i - 1] for i in range(1, control_trains) for k in range(2, control_trains))
        mdl.addConstrs(xi[m, k, i] >= xi[m, k - 1, i] for i in range(control_trains) for k in range(2, control_trains))
        mdl.addConstrs(quicksum(xi[m,k,i]*y[m,i,2*num_station[m]-1] for i in range(control_trains)) + quicksum(y[m, i, 0] for i in range(2, k + 1)) <= depot_real[m] for k in range(2, control_trains - 1))
        # mdl.addConstrs(quicksum(y[m,i,num_station[m]] for i in range(2, k + 1)) <= depot_real[m, num_station[m] - 1] for k in range(2, control_trains - 1))

        mdl.addConstrs(n[m,0,s] == state_n[m,s] for s in range(2 * num_station[m]))
        mdl.addConstrs(n[m,k+1,s] == n[m,k,s] + rho[m,k+1,s]*(d_pre_cut[m,k+1,s]-d_pre_cut[m,k,s])+n_trans[m,k,s]-n_depart[m,k,s] for k in range(control_trains - 1) for s in range(2 * num_station[m]))
        mdl.addConstrs(n_before[m,k,s] == n[m,k,s] + rho[m,k+1,s]*(d[m,k,s]-d_pre_cut[m,k,s])+n_trans[m,k,s] for k in range(control_trains) for s in range(2 * num_station[m]))
        mdl.addConstrs(n_depart[m, k, s] <= C[m, k, s] for k in range(control_trains) for s in range(2 * num_station[m]))
        mdl.addConstrs(n_depart[m, k, s] <= n_before[m, k, s] for k in range(control_trains) for s in range(2 * num_station[m]))
        # mdl.addConstrs(n_depart[k,s]==min(C[k,s],n_before[k,s]) for k in range(control_trains) for s in range(2*num_station))
        mdl.addConstrs(n_arrive[m,k,s] == n_depart[m,k-differ[m,s],s-1] for k in range(2, control_trains) for s in range(1, 2 * num_station[m]))
        mdl.addConstrs(n_trans[m,k,s] == trans_rate*n_arrive[olin[m,s],oser[m,k,s,0],opla[m,s,0]] + trans_rate*n_arrive[olin[m,s],oser[m,k,s,1], opla[m,s,1]] for k in range(control_trains) for s in range(2 * num_station[m]))
        mdl.addConstrs(C[m,k,s] == l[m,k,s]*Cmax for k in range(control_trains) for s in range(2 * num_station[m]))
        mdl.addConstrs(n_after[m,k,s] == n_before[m,k,s] - n_depart[m,k,s] for k in range(control_trains) for s in range(2 * num_station[m]))

    # print(depot_real[m,0])
    # start_time = time.time()
    # tune parameter of gurobi solver
    # mdl.Params.MIPGap = 10 ** (-10)
    # mdl.Params.Cuts = 2
    # mdl.Params.NonConvex = 2
    # mdl.Params.TimeLimit = 3600
    # solving
    mdl.optimize()
    # mdl.remove(C1)
    # mdl.update()
    # end_time = time.time()
    # gurobi_runtime = end_time - start_time
    # print('gurobi runing time = %f seconds' % gurobi_runtime)
    a_values = np.zeros([num_line, control_trains, 2 * max_station])
    d_values = np.zeros([num_line, control_trains, 2 * max_station])
    r_values = np.zeros([num_line, control_trains, 2 * max_station])
    l_values = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    y_values = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    # xi_values = np.zeros([num_line, control_trains, control_trains], dtype=int)
    n_values = np.zeros([num_line, control_trains, 2 * max_station])
    n_after_values = np.zeros([num_line, control_trains, 2 * max_station])
    feas = mdl_feasible(mdl)
    if feas:
        # Print the objective function value
        # print(f"Optimal Objective Value: {mdl.objVal}")
        for m in range(num_line):
            for s in range(2 * num_station[m]):
                for k in range(control_trains):
                    a_values[m, k, s] = a[m, k, s].x
                    d_values[m, k, s] = d[m, k, s].x
                    r_values[m, k, s] = r[m, k, s].x
        l_values = l
        y_values = y
        # xi_values = xi
        n_values = np.array([v.X for v in n.values()]).reshape(num_line, control_trains, 2 * max_station)
        n_after_values = np.array([v.X for v in n_after.values()]).reshape(num_line, control_trains, 2 * max_station)
    else:
        print("Optimization did not converge to an optimal solution.")

    return a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl

"""
gurobi_qp_presolve: all integer variables are preprocessed
"""
def gurobi_lp_presolve(control_trains,d_pre_cut,rho,state_a,state_d,state_r,state_l,state_y,state_n,depot_real,delta,mipgap,log,timelimit,n_threads):
    """
    function of gurobi optimization
    control_trains:         horizon
    """
    mdl = Model('OPT')
    mdl.Params.LogToConsole = log
    mdl.Params.Threads = n_threads
    mdl.Params.TimeLimit = max(1,timelimit)
    mdl.Params.MIPGap = mipgap # 0.001=0.1%
    # mdl.Params.MIPFocus = 2 # default is 0, it was set to 2
    # mdl.Params.Cuts = -1 # default is -1, it was set to 2
    # mdl.Params.NonConvex = 2
    
    oser = np.zeros([num_line, control_trains, 2 * max_station, 2], dtype=int)
    for m in range(num_line):
        for s in range(2 * num_station[m]):
            for k in range(1,control_trains):
                for j in range(2):
                    for i in range(control_trains):  # d_pre = a_pre + t_trans if we set t_trans = tau
                        if (d_pre_cut[olin[m, s], i + 1, opla[m, s, j]] <= d_pre_cut[m, k, s]) and (
                                d_pre_cut[olin[m, s], i + 1, opla[m, s, j]] > d_pre_cut[m, k-1, s]):
                            oser[m, k, s, j] = i
    l = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    y = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    # sign_o = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    for m in range(num_line):
        for s in range(2 * num_station[m]):
            for k in range(2):
                l[m, k, s] = state_l[m, 1 + k, s]
                y[m, k, s] = state_y[m, 1 + k, s]
        for k in range(2, control_trains):
            l[m, k, 0] = round(1 + delta[k - 2, 2+4*m] + 2 * delta[k - 2, 3+4*m])
            y[m, k, 0] = l[m, k, 0]
        for s in range(1, 2 * num_station[m] - 1):
            for k in range(2,control_trains):
                l[m, k, s] = l[m, k - differ[m,s], s - 1] + sigma[m,s] * y[m, k, s]
        for k in range(2,control_trains):
            y[m, k, 2 * num_station[m] - 1] = -l[m, k - differ[m, 2 * num_station[m] - 1], 2 * num_station[m] - 2]
        # for s in range(2 * num_station[m]):
        #     for k in range(control_trains):
        #         if round(y[m, k, s]) == 0:
        #             sign_o[m, k, s] = 0
        #         else:
        #             sign_o[m, k, s] = 1
                
    # decision variables
    d = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='departuretime')
    a = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='arrivaltime')
    r = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='runningtime')
    tau = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='dwelltime')
    # tau_add = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='dwelltime_add')

    n = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='waiting passengers')
    n_arrive = mdl.addVars(num_line, control_trains, 2*max_station, vtype=GRB.CONTINUOUS, name='arrive passengers')
    n_trans = mdl.addVars(num_line, control_trains, 2*max_station, vtype=GRB.CONTINUOUS, name='arriving transfer passengers')
    n_depart = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='departing passengers')
    n_before = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='passengers before departure')
    n_after = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='passengers after departure')
    C = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='Capacity')
    # objective
    mdl.setObjective(quicksum(eta0 * n[0, k, s] * (d_pre_cut[0, k+1, s] - d_pre_cut[0, k, s]) + eta1 * n_after[0, k, s] * (d_pre_cut[0, k + 1, s] - d_pre_cut[0, k, s]) + eta2 * l[0, k, s] * E_regular[0, s]
                              for k in range(2, control_trains) for s in range(2 * num_station[0])) +
                     quicksum(eta0 * n[1, k, s] * (d_pre_cut[1, k+1, s] - d_pre_cut[1, k, s]) + eta1 * n_after[1, k, s] * (d_pre_cut[1, k + 1, s] - d_pre_cut[1, k, s]) + eta2 * l[1, k, s] * E_regular[1, s]
                              for k in range(2, control_trains) for s in range(2 * num_station[1])) +
                     quicksum(eta0 * n[2, k, s] * (d_pre_cut[2, k+1, s] - d_pre_cut[2, k, s]) + eta1 * n_after[2, k, s] * (d_pre_cut[2, k + 1, s] - d_pre_cut[2, k, s]) + eta2 * l[2, k, s] * E_regular[2, s]
                              for k in range(2, control_trains) for s in range(2 * num_station[2])))
    # fixed integer from delta
    link = np.zeros([num_line, control_trains, 2], dtype=int)
    index = 0
    for m in range(num_line):
        for k in range(1, control_trains):
            for i in range(control_trains + 1):
                if (d_pre_cut[m, k, 0] < d_pre_cut[m, i + 1, 2 * num_station[m] - 1] + t_roll) and (
                        d_pre_cut[m, k + 1, 0] > d_pre_cut[m, i, 2 * num_station[m] - 1] + t_roll):
                    link[m, k, index] = i
                    index = round(index + 1)
            index = 0
    xi = np.zeros([num_line, control_trains, control_trains+1])
    for m in range(num_line):
        for k in range(2, control_trains):
            xi[m, k, link[m, k, 0]] = delta[k - 2, 0+4*m]
            # if link[m, k, 0, 1] <= control_trains - 1:
            xi[m, k, link[m, k, 1]] = delta[k - 2, 1+4*m]
        for k in range(control_trains):
            for i in range(control_trains+1):
                if d_pre_cut[m, k, 0] >= d_pre_cut[m, i + 1, 2*num_station[m]-1] + t_roll:
                    xi[m,k,i] = round(1)
                    # elif d_pre_cut[m, k+1, s] <= d_pre_cut[m,i, q] + t_roll:
                    #     xi[m, k, i, s, q] = round(0)
        for k in range(2):
            for i in range(2):
                if state_d[m, 1 + k, 0] >= state_d[m, 1 + i, 2*num_station[m]-1] + t_roll:
                    xi[m, k, i] = round(1)
                # else:
                #     xi[m, k, i, s, q] = round(0)
                if state_d[m, round(1 + k), 0] >= d_pre_cut[m, round(1 + 2), 2*num_station[m]-1] + t_roll:
                    xi[m, k, 2] = round(1)

    # constraints
    for m in range(num_line):
        mdl.addConstrs(a[m,k,s] == state_a[m, 1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s] == state_d[m, 1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s] == state_r[m, 1+k,s] for k in range(2) for s in range(2*num_station[m]))

        for s in range(1,2*num_station[m]):
            for k in range(1,differ[m,s]):
                mdl.addConstr(a[m,k,s]==state_d[m,1+k-differ[m,s],s-1]+state_r[m,1+k-differ[m,s],s-1])
            for k in range(differ[m,s],control_trains):
                mdl.addConstr(a[m,k,s] == d[m,k-differ[m,s],s-1] + r[m,k-differ[m,s],s-1])
            for k in range(control_trains,control_trains+differ[m,s]):
                mdl.addConstr(d[m,k-differ[m,s], s-1] + r[m,k-differ[m,s], s-1] == state_a[m,1+k,s])
                # mdl.addConstr(l[round(k-differ[s]), s-1] + sigma[s]*uy[round(start_train[s]+k),s] == ul[round(start_train[s]+k),s])
        # mdl.addConstrs(a[k,s] == ud[round(start_train[s]+1+k-differ[s]),s-1] + ur[round(start_train[s]+1+k-differ[s]),s-1] for k in range(round(differ[s])) for s in range(1,2*num_station))
        # mdl.addConstrs(a[k,s] == d[round(k-differ[s]),s-1] + r[round(k-differ[s]),s-1] for k in range(round(differ[s]),control_trains) for s in range(1,2*num_station))

        # mdl.addConstrs(tau_add[m, k, s] == sign_o[m, k,s]*t_constant for k in range(control_trains) for s in range(2*num_station[m]))

        mdl.addConstrs(d[m,k,s]>=d_pre_cut[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s]<=d_pre_cut[m,k+1,s] - epsilon for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(a[m,k+1,s]>=d[m,k,s]+h_min for k in range(control_trains-1) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s]==a[m,k,s]+tau[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,control_trains-1,s]+h_min <= state_a[m,round(1+control_trains),s] for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s]>=r_min[m,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s]<=r_max[m,s] for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(tau[m,k,s]>=tau_min+sigma[m,s]*tau_add[m,k,s] for k in range(control_trains) for s in range(1,2*num_station[m]-1))
        mdl.addConstrs(tau[m,k,s]>=tau_min for k in range(control_trains) for s in range(2*num_station[m]))

        mdl.addConstrs(d[m,i,2*num_station[m]-1]+t_roll-d[m,k,0]<=(1-xi[m,k,i])*(Mt-d[m,k,0]) for i in range(control_trains) for k in range(2, control_trains))
        mdl.addConstrs(d[m,i,2*num_station[m]-1]+t_roll-d[m,k,0]>=epsilon+xi[m,k,i]*(mt-d[m,k,0]-epsilon) for i in range(control_trains) for k in range(2, control_trains))
        # mdl.addConstrs(xi[m, k, i] <= xi[m, k, i - 1] for i in range(1, control_trains) for k in range(1, control_trains))
        # mdl.addConstrs(xi[m, k, i] >= xi[m, k - 1, i] for i in range(control_trains) for k in range(1, control_trains))
        mdl.addConstrs(quicksum(xi[m,k,i]*y[m,i,2*num_station[m]-1] for i in range(control_trains)) + quicksum(y[m,i,0] for i in range(2, k + 1)) <= depot_real[m] for k in range(2, control_trains - 1))
        # mdl.addConstrs(quicksum(y[m,i,num_station[m]] for i in range(2, k + 1)) <= depot_real[m, num_station[m] - 1] for k in range(2, control_trains - 1))

        mdl.addConstrs(n[m,0,s] == state_n[m,s] for s in range(2 * num_station[m]))
        mdl.addConstrs(n[m,k+1,s] == n[m,k,s] + rho[m,k+1,s]*(d_pre_cut[m,k+1,s]-d_pre_cut[m,k,s])+n_trans[m,k,s]-n_depart[m,k,s] for k in range(control_trains - 1) for s in range(2 * num_station[m]))
        mdl.addConstrs(n_before[m,k,s] == n[m,k,s] + rho[m,k+1,s]*(d[m,k,s]-d_pre_cut[m,k,s])+n_trans[m,k,s] for k in range(control_trains) for s in range(2 * num_station[m]))
        mdl.addConstrs(n_depart[m, k, s] <= C[m, k, s] for k in range(control_trains) for s in range(2 * num_station[m]))
        mdl.addConstrs(n_depart[m, k, s] <= n_before[m, k, s] for k in range(control_trains) for s in range(2 * num_station[m]))
        # mdl.addConstrs(n_depart[k,s]==min(C[k,s],n_before[k,s]) for k in range(control_trains) for s in range(2*num_station))
        mdl.addConstrs(n_arrive[m,k,s] == n_depart[m,k-differ[m,s],s-1] for k in range(2, control_trains) for s in range(1, 2 * num_station[m]))
        mdl.addConstrs(n_trans[m,k,s] == trans_rate*n_arrive[olin[m,s],oser[m,k,s,0],opla[m,s,0]] + trans_rate*n_arrive[olin[m,s],oser[m,k,s,1], opla[m,s,1]] for k in range(control_trains) for s in range(2 * num_station[m]))
        mdl.addConstrs(C[m,k,s] == l[m,k,s]*Cmax for k in range(control_trains) for s in range(2 * num_station[m]))
        mdl.addConstrs(n_after[m,k,s] == n_before[m,k,s] - n_depart[m,k,s] for k in range(control_trains) for s in range(2 * num_station[m]))

    # print(depot_real[m,0])
    # start_time = time.time()
    # tune parameter of gurobi solver
    # mdl.Params.MIPGap = 10 ** (-10)
    # mdl.Params.Cuts = 2
    # mdl.Params.NonConvex = 2
    # mdl.Params.TimeLimit = 3600
    # solving
    mdl.optimize()
    # mdl.remove(C1)
    # mdl.update()
    # end_time = time.time()
    # gurobi_runtime = end_time - start_time
    # print('gurobi runing time = %f seconds' % gurobi_runtime)
    a_values = np.zeros([num_line, control_trains, 2 * max_station])
    d_values = np.zeros([num_line, control_trains, 2 * max_station])
    r_values = np.zeros([num_line, control_trains, 2 * max_station])
    l_values = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    y_values = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    # xi_values = np.zeros([num_line, control_trains, control_trains + 1], dtype=int)
    n_values = np.zeros([num_line, control_trains, 2 * max_station])
    n_after_values = np.zeros([num_line, control_trains, 2 * max_station])
    feas = mdl_feasible(mdl)
    if feas == True:
        # Print the objective function value
        # print(f"Optimal Objective Value: {mdl.objVal}")
        for m in range(num_line):
            for s in range(2 * num_station[m]):
                for k in range(control_trains):
                    a_values[m, k, s] = a[m, k, s].x
                    d_values[m, k, s] = d[m, k, s].x
                    r_values[m, k, s] = r[m, k, s].x
        l_values = l
        y_values = y
        # xi_values = xi
        n_values = np.array([v.X for v in n.values()]).reshape(num_line, control_trains, 2 * max_station)
        n_after_values = np.array([v.X for v in n_after.values()]).reshape(num_line, control_trains, 2 * max_station)
    else:
        # print("lp_presolve: optimization did not converge to an optimal solution.")
        pass

    return a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl

def gurobi_nlp_presolve(control_trains,d_pre_cut,rho,state_a,state_d,state_r,state_l,state_y,state_n,depot_real,delta,mipgap,log,timelimit, early_term, warm_start, n_threads):
    """
    function of gurobi optimization
    control_trains:         horizon
    """
    mdl = Model('OPT')
    mdl.Params.LogToConsole = log
    mdl.Params.Threads = n_threads
    mdl.Params.TimeLimit = max(1,timelimit)
    mdl.Params.MIPGap = mipgap # 0.001=0.1%
    mdl.Params.MIPFocus = 2 # default is 0, it was set to 2
    mdl.Params.NonConvex = 2
    
    oser = np.zeros([num_line, control_trains, 2 * max_station, 2], dtype=int)
    for m in range(num_line):
        for s in range(2 * num_station[m]):
            for k in range(1,control_trains):
                for j in range(2):
                    for i in range(control_trains):  # d_pre = a_pre + t_trans if we set t_trans = tau
                        if (d_pre_cut[olin[m, s], i + 1, opla[m, s, j]] <= d_pre_cut[m, k, s]) and (
                                d_pre_cut[olin[m, s], i + 1, opla[m, s, j]] > d_pre_cut[m, k-1, s]):
                            oser[m, k, s, j] = i
    l = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    y = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    # sign_o = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    for m in range(num_line):
        for s in range(2 * num_station[m]):
            for k in range(2):
                l[m, k, s] = state_l[m, 1 + k, s]
                y[m, k, s] = state_y[m, 1 + k, s]
        for k in range(2, control_trains):
            l[m, k, 0] = round(1 + delta[k - 2, 2+4*m] + 2 * delta[k - 2, 3+4*m])
            y[m, k, 0] = l[m, k, 0]
        for s in range(1, 2 * num_station[m] - 1):
            for k in range(2,control_trains):
                l[m, k, s] = l[m, k - differ[m,s], s - 1] + sigma[m,s] * y[m, k, s]
        for k in range(2,control_trains):
            y[m, k, 2 * num_station[m] - 1] = -l[m, k - differ[m, 2 * num_station[m] - 1], 2 * num_station[m] - 2]
        # for s in range(2 * num_station[m]):
        #     for k in range(control_trains):
        #         if round(y[m, k, s]) == 0:
        #             sign_o[m, k, s] = 0
        #         else:
        #             sign_o[m, k, s] = 1
                       
    # decision variables
    d = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='departuretime')
    a = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='arrivaltime')
    r = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='runningtime')
    tau = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='dwelltime')
    # tau_add = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='dwelltime_add')

    n = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='waiting passengers')
    n_arrive = mdl.addVars(num_line, control_trains, 2*max_station, vtype=GRB.CONTINUOUS, name='arrive passengers')
    n_trans = mdl.addVars(num_line, control_trains, 2*max_station, vtype=GRB.CONTINUOUS, name='arriving transfer passengers')
    n_depart = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='departing passengers')
    n_before = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='passengers before departure')
    n_after = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='passengers after departure')
    C = mdl.addVars(num_line,control_trains,2*max_station, vtype=GRB.CONTINUOUS,name='Capacity')
    # objective
    mdl.setObjective(quicksum(eta1 * n[0, k, s] * (d[0, k, s] - d_pre_cut[0, k, s]) + eta1 * n_after[0, k, s] * (d_pre_cut[0, k + 1, s] - d[0, k, s]) + eta2 * l[0, k, s] * E_regular[0, s]
                              for k in range(2, control_trains) for s in range(2 * num_station[0])) +
                     quicksum(eta1 * n[1, k, s] * (d[1, k, s] - d_pre_cut[1, k, s]) + eta1 * n_after[1, k, s] * (d_pre_cut[1, k + 1, s] - d[1, k, s]) + eta2 * l[1, k, s] * E_regular[1, s]
                              for k in range(2, control_trains) for s in range(2 * num_station[1])) +
                     quicksum(eta1 * n[2, k, s] * (d[2, k, s] - d_pre_cut[2, k, s]) + eta1 * n_after[2, k, s] * (d_pre_cut[2, k + 1, s] - d[2, k, s]) + eta2 * l[2, k, s] * E_regular[2, s]
                              for k in range(2, control_trains) for s in range(2 * num_station[2])))
    # fixed integer from delta
    link = np.zeros([num_line, control_trains, 2], dtype=int)
    index = 0
    for m in range(num_line):
        for k in range(1, control_trains):
            for i in range(control_trains + 1):
                if (d_pre_cut[m, k, 0] < d_pre_cut[m, i + 1, 2 * num_station[m] - 1] + t_roll) and (
                        d_pre_cut[m, k + 1, 0] > d_pre_cut[m, i, 2 * num_station[m] - 1] + t_roll):
                    link[m, k, index] = i
                    index = round(index + 1)
            index = 0
    xi = np.zeros([num_line, control_trains, control_trains+1])
    for m in range(num_line):
        for k in range(2, control_trains):
            xi[m, k, link[m, k, 0]] = delta[k - 2, 0+4*m]
            # if link[m, k, 0, 1] <= control_trains - 1:
            xi[m, k, link[m, k, 1]] = delta[k - 2, 1+4*m]
        for k in range(control_trains):
            for i in range(control_trains+1):
                if d_pre_cut[m, k, 0] >= d_pre_cut[m, i + 1, 2*num_station[m]-1] + t_roll:
                    xi[m,k,i] = round(1)
                    # elif d_pre_cut[m, k+1, s] <= d_pre_cut[m,i, q] + t_roll:
                    #     xi[m, k, i, s, q] = round(0)
        for k in range(2):
            for i in range(2):
                if state_d[m, 1 + k, 0] >= state_d[m, 1 + i, 2*num_station[m]-1] + t_roll:
                    xi[m, k, i] = round(1)
                # else:
                #     xi[m, k, i, s, q] = round(0)
                if state_d[m, round(1 + k), 0] >= d_pre_cut[m, round(1 + 2), 2*num_station[m]-1] + t_roll:
                    xi[m, k, 2] = round(1)

    # constraints
    for m in range(num_line):
        mdl.addConstrs(a[m,k,s] == state_a[m, 1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s] == state_d[m, 1+k,s] for k in range(2) for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s] == state_r[m, 1+k,s] for k in range(2) for s in range(2*num_station[m]))

        for s in range(1,2*num_station[m]):
            for k in range(1,differ[m,s]):
                mdl.addConstr(a[m,k,s]==state_d[m,1+k-differ[m,s],s-1]+state_r[m,1+k-differ[m,s],s-1])
            for k in range(differ[m,s],control_trains):
                mdl.addConstr(a[m,k,s] == d[m,k-differ[m,s],s-1] + r[m,k-differ[m,s],s-1])
            for k in range(control_trains,control_trains+differ[m,s]):
                mdl.addConstr(d[m,k-differ[m,s], s-1] + r[m,k-differ[m,s], s-1] == state_a[m,1+k,s])
                # mdl.addConstr(l[round(k-differ[s]), s-1] + sigma[s]*uy[round(start_train[s]+k),s] == ul[round(start_train[s]+k),s])
        # mdl.addConstrs(a[k,s] == ud[round(start_train[s]+1+k-differ[s]),s-1] + ur[round(start_train[s]+1+k-differ[s]),s-1] for k in range(round(differ[s])) for s in range(1,2*num_station))
        # mdl.addConstrs(a[k,s] == d[round(k-differ[s]),s-1] + r[round(k-differ[s]),s-1] for k in range(round(differ[s]),control_trains) for s in range(1,2*num_station))

        # mdl.addConstrs(tau_add[m, k, s] == sign_o[m, k,s]*t_constant for k in range(control_trains) for s in range(2*num_station[m]))

        mdl.addConstrs(d[m,k,s]>=d_pre_cut[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s]<=d_pre_cut[m,k+1,s] - epsilon for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(a[m,k+1,s]>=d[m,k,s]+h_min for k in range(control_trains-1) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,k,s]==a[m,k,s]+tau[m,k,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(d[m,control_trains-1,s]+h_min <= state_a[m,round(1+control_trains),s] for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s]>=r_min[m,s] for k in range(control_trains) for s in range(2*num_station[m]))
        mdl.addConstrs(r[m,k,s]<=r_max[m,s] for k in range(control_trains) for s in range(2*num_station[m]))
        # mdl.addConstrs(tau[m,k,s]>=tau_min+sigma[m,s]*tau_add[m,k,s] for k in range(control_trains) for s in range(1,2*num_station[m]-1))
        mdl.addConstrs(tau[m,k,s]>=tau_min for k in range(control_trains) for s in range(2*num_station[m]))

        mdl.addConstrs(d[m,i,2*num_station[m]-1]+t_roll-d[m,k,0]<=(1-xi[m,k,i])*(Mt-d[m,k,0]) for i in range(control_trains) for k in range(2, control_trains))
        mdl.addConstrs(d[m,i,2*num_station[m]-1]+t_roll-d[m,k,0]>=epsilon+xi[m,k,i]*(mt-d[m,k,0]-epsilon) for i in range(control_trains) for k in range(2, control_trains))
        # mdl.addConstrs(xi[m, k, i] <= xi[m, k, i - 1] for i in range(1, control_trains) for k in range(1, control_trains))
        # mdl.addConstrs(xi[m, k, i] >= xi[m, k - 1, i] for i in range(control_trains) for k in range(1, control_trains))
        mdl.addConstrs(quicksum(xi[m,k,i]*y[m,i,2*num_station[m]-1] for i in range(control_trains)) + quicksum(y[m,i,0] for i in range(2, k + 1)) <= depot_real[m] for k in range(2, control_trains - 1))
        # mdl.addConstrs(quicksum(y[m,i,num_station[m]] for i in range(2, k + 1)) <= depot_real[m, num_station[m] - 1] for k in range(2, control_trains - 1))

        mdl.addConstrs(n[m,0,s] == state_n[m,s] for s in range(2 * num_station[m]))
        mdl.addConstrs(n[m,k+1,s] == n[m,k,s] + rho[m,k+1,s]*(d_pre_cut[m,k+1,s]-d_pre_cut[m,k,s])+n_trans[m,k,s]-n_depart[m,k,s] for k in range(control_trains - 1) for s in range(2 * num_station[m]))
        mdl.addConstrs(n_before[m,k,s] == n[m,k,s] + rho[m,k+1,s]*(d[m,k,s]-d_pre_cut[m,k,s])+n_trans[m,k,s] for k in range(control_trains) for s in range(2 * num_station[m]))
        mdl.addConstrs(n_depart[m, k, s] <= C[m, k, s] for k in range(control_trains) for s in range(2 * num_station[m]))
        mdl.addConstrs(n_depart[m, k, s] <= n_before[m, k, s] for k in range(control_trains) for s in range(2 * num_station[m]))
        # mdl.addConstrs(n_depart[k,s]==min(C[k,s],n_before[k,s]) for k in range(control_trains) for s in range(2*num_station))
        mdl.addConstrs(n_arrive[m,k,s] == n_depart[m,k-differ[m,s],s-1] for k in range(2, control_trains) for s in range(1, 2 * num_station[m]))
        mdl.addConstrs(n_trans[m,k,s] == trans_rate*n_arrive[olin[m,s],oser[m,k,s,0],opla[m,s,0]] + trans_rate*n_arrive[olin[m,s],oser[m,k,s,1], opla[m,s,1]] for k in range(control_trains) for s in range(2 * num_station[m]))
        mdl.addConstrs(C[m,k,s] == l[m,k,s]*Cmax for k in range(control_trains) for s in range(2 * num_station[m]))
        mdl.addConstrs(n_after[m,k,s] == n_before[m,k,s] - n_depart[m,k,s] for k in range(control_trains) for s in range(2 * num_station[m]))

    # print(depot_real[m,0])
    # start_time = time.time()
    # tune parameter of gurobi solver
    # mdl.Params.MIPGap = 10 ** (-10)
    # mdl.Params.Cuts = 2
    # mdl.Params.NonConvex = 2
    # mdl.Params.TimeLimit = 3600
    # solving
    
    #added by Caio
    mdl.update()
    if warm_start == 1:
        mdl.NumStart = 1
        mdl_lp = gurobi_lp_presolve(control_trains,d_pre_cut,rho,state_a,state_d,state_r,state_l,state_y,state_n,depot_real,delta, mipgap,0,20,n_threads)[-1]
        list_lp_solution = mdl_lp.getAttr("X", mdl_lp.getVars())
        mdl.setAttr("Start", mdl.getVars(), list_lp_solution)        
        mdl.Params.TimeLimit = max(1,timelimit - mdl_lp.Runtime)
    
    #added by Caio
    if early_term == 1:
        mdl._gap, mdl._time_to_best = 1.0, float("inf")
        mdl.optimize(callback)
    elif early_term == 0:
        mdl.optimize()
        
    #added by Caio
    if warm_start==1:
        mdl._Runtime = mdl.Runtime+mdl_lp.Runtime
    else:
        mdl._Runtime = mdl.Runtime
        
    # mdl.remove(C1)
    # mdl.update()
    # end_time = time.time()
    # gurobi_runtime = end_time - start_time
    # print('gurobi runing time = %f seconds' % gurobi_runtime)
    a_values = np.zeros([num_line, control_trains, 2 * max_station])
    d_values = np.zeros([num_line, control_trains, 2 * max_station])
    r_values = np.zeros([num_line, control_trains, 2 * max_station])
    l_values = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    y_values = np.zeros([num_line, control_trains, 2 * max_station], dtype=int)
    # xi_values = np.zeros([num_line, control_trains, control_trains + 1], dtype=int)
    n_values = np.zeros([num_line, control_trains, 2 * max_station])
    n_after_values = np.zeros([num_line, control_trains, 2 * max_station])
    feas = mdl_feasible(mdl)
    if feas == True:
        # Print the objective function value
        # print(f"Optimal Objective Value: {mdl.objVal}")
        for m in range(num_line):
            for s in range(2 * num_station[m]):
                for k in range(control_trains):
                    a_values[m, k, s] = a[m, k, s].x
                    d_values[m, k, s] = d[m, k, s].x
                    r_values[m, k, s] = r[m, k, s].x
        l_values = l
        y_values = y
        # xi_values = xi
        n_values = np.array([v.X for v in n.values()]).reshape(num_line, control_trains, 2 * max_station)
        n_after_values = np.array([v.X for v in n_after.values()]).reshape(num_line, control_trains, 2 * max_station)
    else:
        # print("nlp_presolve: optimization did not converge to an optimal solution.")
        pass

    return a_values, d_values, r_values, l_values, y_values, n_values, n_after_values, mdl

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
    def __init__(self, control_trains, mode=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])):

        self.cntr = 0  # termination counter
        self.control_trains = control_trains
        self.mode = mode
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.idx_cntr = 0
        self.idx_group = 0 # group index

    def setState(self, idx_cntr, idx_group, mode=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])):
        self.cntr = 0
        self.mode = mode
        self.terminated = False
        self.truncated = False
        self.idx_cntr = idx_cntr
        self.idx_group = idx_group

    def copyEnv(self, env):
        # self.state_start = env.state_start
        self.mode = copy.deepcopy(env.mode)
        self.cntr = copy.deepcopy(env.cntr)
        self.terminated = copy.deepcopy(env.terminated)
        self.truncated = copy.deepcopy(env.truncated)
        self.idx_cntr = copy.deepcopy(env.idx_cntr)
        self.idx_group = copy.deepcopy(env.idx_group)
        self.a_real = copy.deepcopy(env.a_real)
        self.d_real = copy.deepcopy(env.d_real)
        self.r_real = copy.deepcopy(env.r_real)
        self.l_real = copy.deepcopy(env.l_real)
        self.y_real = copy.deepcopy(env.y_real)
        self.n_real = copy.deepcopy(env.n_real)

        self.depot_real = copy.deepcopy(env.depot_real)
        self.start_index = copy.deepcopy(env.start_index)
        self.state_rho = copy.deepcopy(env.state_rho)
        self.d_pre_cut = copy.deepcopy(env.d_pre_cut)
        self.d_pre_cut_old = copy.deepcopy(env.d_pre_cut_old)

        self.state_a = copy.deepcopy(env.state_a)
        self.state_d = copy.deepcopy(env.state_d)
        self.state_r = copy.deepcopy(env.state_r)
        self.state_l = copy.deepcopy(env.state_l)
        self.state_y = copy.deepcopy(env.state_y)
        self.state_n = copy.deepcopy(env.state_n)
        self.state_depot = copy.deepcopy(env.state_depot)

    def set_randState(self, d_pre, rho_whole, un, ul, uy, ua, ud, ur, depot, mode=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])):

        i = np.random.randint(30, 210 - self.control_trains)  # equivalent for 1 year of data Ts=30m
        g = np.random.randint(1000)

        self.mode = mode
        self.cntr = 0
        self.terminated = False
        self.truncated = False
        self.idx_cntr = i
        self.idx_group = g
        self.a_real = np.zeros([num_line, num_train, 2 * max_station])
        self.d_real = np.zeros([num_line, num_train, 2 * max_station])
        self.r_real = np.zeros([num_line, num_train, 2 * max_station])
        self.l_real = np.zeros([num_line, num_train, 2 * max_station])
        self.y_real = np.zeros([num_line, num_train, 2 * max_station])
        self.n_real = np.zeros([num_line, num_train, 2 * max_station])
        self.depot_real = np.zeros([num_line, num_train])
        for m in range(num_line):
            for s in range(2 * num_station[m]):
                for k in range(num_train):
                    self.a_real[m, k, s] = ua[m, k, s, self.idx_group]
                    self.d_real[m, k, s] = ud[m, k, s, self.idx_group]
                    self.r_real[m, k, s] = ur[m, k, s, self.idx_group]
                    self.l_real[m, k, s] = ul[m, k, s, self.idx_group]
                    self.y_real[m, k, s] = uy[m, k, s, self.idx_group]
                    self.n_real[m, k, s] = un[m, k, s, self.idx_group]
            for k in range(num_train):
                self.depot_real[m, k] = depot[m, k, self.idx_group]

        '''cutting states for gurobi'''
        self.start_index = np.zeros([num_line, 2 * max_station], dtype=int)
        for m in range(num_line):
            self.start_index[m,0] = self.idx_cntr
            for s in range(1, 2 * num_station[m]):
                self.start_index[m, s] = self.start_index[m, s - 1] - differ[m, s]
        self.state_rho = np.zeros([num_line, self.control_trains + 1, 2 * max_station])
        for m in range(num_line):
            for s in range(2 * num_station[m]):
                for k in range(self.control_trains + 1):
                    self.state_rho[m, k, s] = rho_whole[m, self.start_index[m,s] + k, s, self.idx_group]
        self.state_a = np.zeros([num_line, self.control_trains + 3, 2 * max_station])
        self.d_pre_cut = np.zeros([num_line, self.control_trains + 3, 2 * max_station])
        for m in range(num_line):
            for s in range(2 * num_station[m]):
                for k in range(self.control_trains + 3):
                    self.state_a[m, k, s] = self.a_real[m, self.start_index[m,s] - 1 + k, s]
                    self.d_pre_cut[m, k, s] = d_pre[m, self.start_index[m, s] + k, s]
        self.state_d = np.zeros([num_line, 3, 2 * max_station])
        self.state_r = np.zeros([num_line, 3, 2 * max_station])
        self.state_l = np.zeros([num_line, 3, 2 * max_station])
        self.state_y = np.zeros([num_line, 3, 2 * max_station])
        for m in range(num_line):
            for s in range(2 * num_station[m]):
                for k in range(3):
                    self.state_d[m, k, s] = self.d_real[m, self.start_index[m,s] - 1 + k, s]
                    self.state_r[m, k, s] = self.r_real[m, self.start_index[m,s] - 1 + k, s]
                    self.state_l[m, k, s] = self.l_real[m, self.start_index[m,s] - 1 + k, s]
                    self.state_y[m, k, s] = self.y_real[m, self.start_index[m,s] - 1 + k, s]
        self.state_n = np.zeros([num_line, 2 * max_station])
        for m in range(num_line):
            for s in range(2 * num_station[m]):
                self.state_n[m, s] = self.n_real[m, self.start_index[m,s], s]
        self.state_depot = np.zeros([num_line])
        for m in range(num_line):
            self.state_depot[m] = self.depot_real[m, self.start_index[m,0] + 2]
            
        # self.state_sign_o = np.zeros([num_line, self.control_trains, 2 * max_station])

        self.d_pre_cut_old = self.d_pre_cut

    # def build_delta_vector(self, list_action: list) -> np.array:
    #     # from list of actions builds a np.array with the stacked deltas for each time step of the prediction horizon
    #     delta = action_dict[str(round(list_action[0]))]
    #     for i in range(1, self.control_trains - 2):
    #         delta = np.concatenate((delta, action_dict[str(round(list_action[i]))]))

    #     return delta

    # def step(self, list_action, d_pre, rho_whole, r_max, r_min, differ, Cmax, sigma, num_station,num_train, E_regular):
    def step(self, delta, d_pre, rho_whole, mipgap, log, timelimit, early_term, warm_start, n_threads, opt):

        # '''objective value of the on_control cost'''
        # J_original = original(self.control_trains,self.d_pre_cut,self.state_rho,self.d_real,self.l_real,self.state_n,Cmax,self.start_index)
        # print(J_original)
        #
        # a_minlp, d_minlp, r_minlp, l_minlp, y_minlp, delta_minlp, mdl_minlp = gurobi_minlp(self.control_trains, self.d_pre_cut, self.state_rho,
        #              self.state_a, self.state_d, self.state_r, self.state_l, self.state_y, self.state_n, self.state_depot,
        #              num_station, differ, sigma, t_constant, r_min, r_max, E_regular, Cmax)
        #
        #
        # delta = self.build_delta_vector(list_action)
        # '''use delta generated by minlp to guribi_qp'''
        # delta = delta_minlp
        # '''gurobi_qp: xi is not preprocessed, and other integer variables are preprocessed'''
        # a_qp, d_qp, r_qp, l_qp, y_qp, xi_qp, mdl = gurobi_qp(self.control_trains, self.d_pre_cut,
        #                                                      self.state_rho, self.state_a, self.state_d, self.state_r,
        #                                                      self.state_l, self.state_y, self.state_n,
        #                                                      self.state_depot, delta, num_station, differ, sigma,
        #                                                      t_constant, r_min, r_max, E_regular, Cmax)
        # '''gurobi_qp_presolve: all integer variables are preprocessed'''
        # a_qp, d_qp, r_qp, l_qp, y_qp, xi_qp, mdl = gurobi_qp_presolve(self.control_trains, self.d_pre_cut, self.state_rho,
        #                                               self.state_a, self.state_d, self.state_r, self.state_l,self.state_y, self.state_n, self.state_depot,
        #                                               delta, num_station, differ, sigma, t_constant, r_min, r_max, E_regular, Cmax)
        # J_original = original(self.control_trains,self.d_pre_cut,self.state_rho,self.d_real,self.l_real,self.state_n,Cmax,eta,self.start_index)

        self.d_pre_cut_old = self.d_pre_cut  # saves d_pre_cut

        # J_original = original(self.control_trains, self.d_pre_cut, self.state_rho, self.d_real, self.l_real,
        #                       self.state_n, self.start_index)
        # print(J_original)

        if opt == "milp":
            a_qp, d_qp, r_qp, l_qp, y_qp, delta, n, n_after, mdl = gurobi_milp(self.control_trains,self.d_pre_cut, self.state_rho,self.state_a,self.state_d,self.state_r, self.state_l,self.state_y,self.state_n,self.state_depot,mipgap,log,timelimit,n_threads)
        if opt == "minlp":
            a_qp, d_qp, r_qp, l_qp, y_qp, delta, n, n_after, mdl = gurobi_minlp(self.control_trains,self.d_pre_cut, self.state_rho,self.state_a,self.state_d,self.state_r, self.state_l,self.state_y,self.state_n,self.state_depot,mipgap,log,timelimit,early_term,warm_start,n_threads)

        if opt == "lp":
            a_qp, d_qp, r_qp, l_qp, y_qp, n, n_after, mdl = gurobi_lp_presolve(self.control_trains,self.d_pre_cut,self.state_rho,self.state_a, self.state_d,self.state_r, self.state_l,self.state_y, self.state_n,self.state_depot,delta,mipgap,log,timelimit,n_threads)
            
        if opt == "nlp":
            a_qp, d_qp, r_qp, l_qp, y_qp, n, n_after, mdl = gurobi_nlp_presolve(self.control_trains,self.d_pre_cut,self.state_rho,self.state_a, self.state_d,self.state_r, self.state_l,self.state_y, self.state_n,self.state_depot,delta,mipgap,log,timelimit,early_term,warm_start,n_threads)

        feas = mdl_feasible(mdl)
        # print(xi_qp-xi_qp_presolve)

        if not feas:

            self.reward = np.array(-1)
            self.terminated = True

            info = {'feasible': feas,
                    'mdl': mdl}

        else:
            # rew = cost2rew(mdl.ObjVal, lower_bound=-450, alpha=0.0015)
            # rew = cost2rew2(mdl.ObjVal, J_original)
            '''implement the first control input in MPC'''
            for m in range(num_line):
                for s in range(2 * num_station[m]):
                    self.a_real[m,round(self.start_index[m,s] + 2), s] = a_qp[m, 2, s]
                    self.r_real[m,round(self.start_index[m,s] + 2), s] = r_qp[m, 2, s]
                    self.d_real[m,round(self.start_index[m,s] + 2), s] = d_qp[m, 2, s]
                    self.l_real[m,round(self.start_index[m,s] + 2), s] = l_qp[m, 2, s]
                    self.y_real[m,round(self.start_index[m,s] + 2), s] = y_qp[m, 2, s]
                for k in range(num_train - 1):
                    if d_pre[m, k, 0] < d_pre[m, 2, 2 * num_station[m] - 1] + t_roll:
                        self.depot_real[m, k + 1] = self.depot_real[m, k] - self.y_real[m, k, 0]
                        # self.depot_real[m, k + 1, num_station[m] - 1] = self.depot_real[m, k, num_station[m] - 1] - (self.l_real[m, k, num_station[m]] - self.l_real[m, k, num_station[m] - 1])
                    else:
                        for i in range(2,num_train - 1):
                            if (d_pre[m, k, 0] >= d_pre[m, i, 2 * num_station[m] - 1] + t_roll) and (d_pre[m, k, 0] < d_pre[m, i + 1, 2 * num_station[m] - 1] + t_roll):
                                self.depot_real[m, k + 1] = self.depot_real[m, k] - self.y_real[m, k, 0] - self.y_real[m, i-2, 2 * num_station[m] - 1]
                                # self.depot_real[m, k + 1, num_station[m] - 1] = self.depot_real[m, k, num_station[m] - 1] - (self.l_real[m, k, num_station[m]] - self.l_real[m, k, num_station[m] - 1])
            self.n_real[m,:,:] = np.zeros([num_train, 2 * max_station])
            n_depart_real = np.zeros([num_line,num_train, 2 * max_station])
            n_before_real = np.zeros([num_line,num_train, 2 * max_station])
            C_real = np.zeros([num_line,num_train, 2 * max_station])
            n_after_real = np.zeros([num_line,num_train, 2 * max_station])
            n_arrive_real = np.zeros([num_line,num_train, 2 * max_station])
            for m in range(num_line):
                for s in range(2 * num_station[m]):
                    self.n_real[m, 0, s] = rho_whole[m, 0, s, self.idx_group] * (d_pre[m, 0, s] - 0)
                for s in range(2 * num_station[m]):
                    for k in range(num_train - 1):
                        n_before_real[m, k, s] = self.n_real[m, k, s] + rho_whole[m, k + 1, s, self.idx_group] * (self.d_real[m, k, s] - d_pre[m, k, s])
                        C_real[m, k, s] = self.l_real[m, k, s] * Cmax
                        n_depart_real[m, k, s] = min(C_real[m, k, s], n_before_real[m, k, s])
                        n_after_real[m, k, s] = n_before_real[m, k, s] - n_depart_real[m, k, s]
                        self.n_real[m, k + 1, s] = self.n_real[m, k, s] + rho_whole[m, k + 1, s, self.idx_group] * (d_pre[m, k + 1, s] - d_pre[m, k, s]) - n_depart_real[m, k, s]
                for s in range(1, 2 * num_station[m]):
                    for k in range(num_train - 1):
                        n_arrive_real[m, k, s] = n_depart_real[m, k, s - 1]

            self.n_real[m,:,:] = np.zeros([num_train, 2 * max_station])
            n_depart_real = np.zeros([num_line, num_train, 2 * max_station])
            n_before_real = np.zeros([num_line, num_train, 2 * max_station])
            C_real = np.zeros([num_line, num_train, 2 * max_station])
            n_after_real = np.zeros([num_line, num_train, 2 * max_station])
            n_trans_real = np.zeros([num_line, num_train, 2 * max_station])
            for m in range(num_line):
                for s in range(2 * num_station[m]):
                    self.n_real[m, 0, s] = rho_whole[m, 0, s, self.idx_group] * (d_pre[m, 0, s] - 0)
                for s in range(2 * num_station[m]):
                    for k in range(num_train - 1):
                        n_trans_real[m, k, s] = trans_rate*n_arrive_real[olin[m,s],otra[m,k,s,0],opla[m,s,0]] + trans_rate*n_arrive_real[olin[m,s],otra[m,k,s,1],opla[m,s,1]]
                        n_before_real[m, k, s] = self.n_real[m, k, s] + rho_whole[m, k + 1, s, self.idx_group] * (self.d_real[m, k, s] - d_pre[m, k, s]) + n_trans_real[m, k, s]
                        C_real[m, k, s] = self.l_real[m, k, s] * Cmax
                        n_depart_real[m, k, s] = min(C_real[m, k, s], n_before_real[m, k, s])
                        n_after_real[m, k, s] = n_before_real[m, k, s] - n_depart_real[m, k, s]
                        self.n_real[m, k + 1, s] = self.n_real[m, k, s] + rho_whole[m, k + 1, s, self.idx_group] * (d_pre[m, k + 1, s] - d_pre[m, k, s]) + n_trans_real[m, k, s] - n_depart_real[m, k, s]

            # print(np.amax(self.n_real))
            # print(differ)
            # self.mode = action_dict[str(round(list_action[0]))]
            # self.reward = rew
            self.reward = 0
            self.cntr += 1
            self.idx_cntr += 1
            self.start_index += 1

            '''update states'''
            self.state_rho = np.zeros([num_line, self.control_trains + 1, 2 * max_station])
            self.d_pre_cut = np.zeros([num_line, self.control_trains + 3, 2 * max_station])
            self.state_a = np.zeros([num_line, self.control_trains + 3, 2 * max_station])
            self.state_d = np.zeros([num_line, 3, 2 * max_station])
            self.state_r = np.zeros([num_line, 3, 2 * max_station])
            self.state_l = np.zeros([num_line, 3, 2 * max_station])
            self.state_y = np.zeros([num_line, 3, 2 * max_station])
            self.state_n = np.zeros([num_line, 2 * max_station])
            
            #added by Caio
            self.state_n_before = np.zeros([num_line, 2 * max_station])
            self.state_n_after = np.zeros([num_line, 2 * max_station])
            self.state_depot = np.zeros([num_line])
            
            for m in range(num_line):
                for s in range(2 * num_station[m]):
                    for k in range(self.control_trains + 1):
                        self.state_rho[m, k, s] = rho_whole[m, round(self.start_index[m, s] + k), s, self.idx_group]
                    for k in range(self.control_trains + 3):
                        self.state_a[m, k, s] = self.a_real[m, round(self.start_index[m,s] - 1 + k), s]
                        self.d_pre_cut[m, k, s] = d_pre[m, round(self.start_index[m,s] + k), s]
                    for k in range(3):
                        self.state_d[m, k, s] = self.d_real[m, round(self.start_index[m,s] - 1 + k), s]
                        self.state_r[m, k, s] = self.r_real[m, round(self.start_index[m,s] - 1 + k), s]
                        self.state_l[m, k, s] = self.l_real[m, round(self.start_index[m,s] - 1 + k), s]
                        self.state_y[m, k, s] = self.y_real[m, round(self.start_index[m,s] - 1 + k), s]
                    self.state_n[m, s] = self.n_real[m, round(self.start_index[m,s]), s]
                    #added by Caio
                    self.state_n_after[m, s] = n_after_real[m, round(self.start_index[m,s]), s]
                    self.state_n_before[m, s] = n_before_real[m, round(self.start_index[m,s]), s]
                self.state_depot[m] = self.depot_real[m, round(self.start_index[m,0] + 2)]

                #removed in 18/11/24 by Caio
                # if (self.cntr >= 210 - self.control_trains - self.idx_cntr)|(self.cntr >= 30):
                #     self.truncated = True
                #added in 18/11/24 by Caio
                if self.cntr >= 30:
                    self.truncated = True

                if self.idx_cntr >= 210 - self.control_trains:
                    self.truncated = True                 

                info = {'feasible': feas,
                        'objval': mdl.ObjVal,
                        'mdl': mdl
                        }

            # return (self.state_rho, self.d_pre_cut, self.state_a, self.state_d, self.state_r, self.state_l, self.state_y,self.state_n, self.state_depot,
            #         self.reward, self.terminated, self.truncated, info)
            self.n = n
            self.n_after = n_after
            self.d = d_qp
            self.l = l_qp
            # self.sign_o = sign_o
            
            self.n_before_real = n_before_real
            self.n_after_real = n_after_real

        return (self.state_rho, self.d_pre_cut, self.state_a, self.state_d, self.state_r, self.state_l, self.state_y,
                self.state_n, self.state_depot, self.reward, self.terminated, self.truncated, delta, info)