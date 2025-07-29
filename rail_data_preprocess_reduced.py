import numpy as np
import torch

from rail_fun import norm_state

#tests conversion from original list_actions to reduced list of actions
def reduce_list_actions(list_action, total_action_set, N_control):
    
    list_actions_reduced = np.zeros((N_control,), dtype=np.int32)
    for i in range(N_control):
        list_actions_reduced[i] = np.where(list_action[i] == total_action_set)[0][0]
        
    return list_actions_reduced

def split_train_val_test(stacked_array, val_split=0.8):
    
    # 80/10/10 split
    
    N_datapoints = stacked_array.shape[0]
    test_split = (1-(1-val_split)/2)
    
    stacked_array_train = stacked_array[:int(np.ceil(N_datapoints*val_split))]
    
    stacked_array_val = stacked_array[int(np.ceil(N_datapoints*val_split)):int(np.ceil(N_datapoints*test_split))]
    
    stacked_array_test = stacked_array[int(np.ceil(N_datapoints*test_split)):]
    
    return stacked_array_train, stacked_array_val, stacked_array_test

def reduce_Dataset(stacked_states, stacked_actions, total_action_set, N_control):
    
        """
        It removes from the dataset the state-actions pairs that are removed by thresholding, which is used to remove unlikely state-action pairs; thus reducing the output space and the complexity of the neural network.
        """
        
        stacked_states_reduced = []
        stacked_actions_reduced = []
        cntr_outlier = 0
        for i in range(stacked_actions.shape[0]):
            try:
                stacked_actions_reduced.append(reduce_list_actions(stacked_actions[i], total_action_set, N_control))
                stacked_states_reduced.append(stacked_states[i])
            except:
                cntr_outlier +=1
                
        return stacked_states_reduced, stacked_actions_reduced, cntr_outlier

######################################################################################################################

def get_preprocessed_data_reduced(opt_data, threshold_counts, N, opt_state, opt_label, testing):
    
    # import numpy as np
    # N=40
    # minlp_info_compressed = np.load('data_milp//data_milp_N%.2d_%.2d.npy' %(N, 0), allow_pickle=True)
    # for job_idx in range(1,28):
    #     tmp_vector = np.load('data_milp//data_milp_N%.2d_%.2d.npy' %(N, job_idx), allow_pickle=True)
    #     minlp_info_compressed = np.concatenate((minlp_info_compressed, tmp_vector))
    # minlp_info_compressed = minlp_info_compressed[:120000, :]
    # np.save('data_milp//data_milp_ol_N%.2d_condensed.npy' %N, minlp_info_compressed, allow_pickle=True)
    
    """
    opt_data:
        'milp_ol', 'milp_cl'
    
    opt_state:
        1: l_0 + mean
        2: l_{0,1,2} + mean
        3: l_0 + downsampling
        4: l_{0,1,2} + downsampling
        
    opt_label:
        'classification' or 'regression'
    
    """
    
    N_control = N - 2
        
    # str_data = 'data_optimal//data_reduced_%s_N%d_condensed.npy' %(opt_data, N)
    # str_data = 'data_optimal//data_reduced_%s_N%d_999.npy' %(opt_data, N) # for testing
    # array_data = np.load(str_data, allow_pickle=True)
    # dict_state_list = array_data[0]
    # dict_output_list = array_data[1]
    
    # state_n = dict_state_list['state_n']
    # state_depot = dict_state_list['state_depot']
    # idx_cntr = dict_state_list['idx_cntr']
    
    if testing==True:
        str_data = 'data_optimal_reduced//data_reduced_%s_N%d_%.3d.npy' %(opt_data, N, 0)
    elif testing==False:
        str_data = '//scratch//cfoliveiradasi//railway_largescale//training_data_reduced//data_reduced_%s_N%d_%3d.npy' %(opt_data, N, 0)
    array_data = np.load(str_data, allow_pickle=True)
    dict_state_list = array_data[0]
    dict_output_list = array_data[1]
    
    state_n = dict_state_list['state_n']
    state_depot = dict_state_list['state_depot']
    idx_cntr = dict_state_list['idx_cntr']
    state_rho_down= dict_state_list['state_rho_down']
    state_rho_mean= dict_state_list['state_rho_mean']
    state_l_0= dict_state_list['state_l_0']
    state_l_1= dict_state_list['state_l_1']
    state_l_2= dict_state_list['state_l_2']
    if opt_label=='classification':
        stacked_list_actions = dict_output_list['list_actions']
    elif opt_label=='regression':
        stacked_list_actions = dict_output_list['delta']
         
    for i in range(1,25):
        if testing==True:
            str_data = 'data_optimal_reduced//data_reduced_%s_N%d_%.3d.npy' %(opt_data, N, i)
        elif testing==False:
            str_data = '//scratch//cfoliveiradasi//railway_largescale//training_data_reduced//data_reduced_%s_N%d_%.3d.npy' %(opt_data, N, i)
        array_data = np.load(str_data, allow_pickle=True)
        dict_state_list = array_data[0]
        dict_output_list = array_data[1]
        
        state_n = np.concatenate((state_n, dict_state_list['state_n']))
        state_depot = np.concatenate((state_depot, dict_state_list['state_depot']))
        idx_cntr = np.concatenate((idx_cntr, dict_state_list['idx_cntr']))
        state_rho_down= np.concatenate((state_rho_down, dict_state_list['state_rho_down']))
        state_rho_mean= np.concatenate((state_rho_mean, dict_state_list['state_rho_mean']))
        state_l_0= np.concatenate((state_l_0, dict_state_list['state_l_0']))
        state_l_1= np.concatenate((state_l_1, dict_state_list['state_l_1']))
        state_l_2= np.concatenate((state_l_2, dict_state_list['state_l_2']))
        
        if opt_label=='classification':
            tmp = dict_output_list['list_actions']
        elif opt_label=='regression':
            tmp = dict_output_list['delta']    
        stacked_list_actions = np.concatenate((stacked_list_actions, tmp)) 
    
    if opt_state==1:
        # state_rho_mean= dict_state_list['state_rho_mean']
        # state_l_0= dict_state_list['state_l_0']
        stacked_states = np.concatenate((state_n, state_depot, idx_cntr, state_rho_mean, state_l_0), axis=1)
    elif opt_state==2:
        # state_rho_mean= dict_state_list['state_rho_mean']
        # state_l_0= dict_state_list['state_l_0']
        # state_l_1= dict_state_list['state_l_1']
        # state_l_2= dict_state_list['state_l_2']
        stacked_states = np.concatenate((state_n, state_depot, idx_cntr, state_rho_mean, state_l_0, state_l_1, state_l_2), axis=1)
    elif opt_state==3:
        # state_rho_down= dict_state_list['state_rho_down']
        # state_l_0= dict_state_list['state_l_0']
        stacked_states = np.concatenate((state_n, state_depot, idx_cntr, state_rho_down, state_l_0), axis=1)
    elif opt_state==4:
        # state_rho_down= dict_state_list['state_rho_down']
        # state_l_0= dict_state_list['state_l_0']
        # state_l_1= dict_state_list['state_l_1']
        # state_l_2= dict_state_list['state_l_2']
        stacked_states = np.concatenate((state_n, state_depot, idx_cntr, state_rho_down, state_l_0, state_l_1, state_l_2), axis=1)
    
    # cost_list = dict_output_list['mdl_Obj']
    # mipgap_list = dict_output_list['mdl_mipgap']
    # runtime_list = dict_output_list['mdl_runtime']
    # status_list = dict_output_list['mdl_status']
    
    state_min = np.min(stacked_states, axis=0)
    state_max = np.max(stacked_states, axis=0)
    stacked_states = norm_state(stacked_states, state_min, state_max)
    
    input_size = stacked_states.shape[1]

    # train/val/test split
    
    stacked_states_train, stacked_states_val, stacked_states_test = split_train_val_test(stacked_states, val_split=0.8)
    stacked_actions_train, stacked_actions_val, stacked_actions_test = split_train_val_test(stacked_list_actions, val_split=0.8)
    
    # cost_list_train, cost_list_val, cost_list_test = split_train_val_test(cost_list, val_split=0.8)
    # mipgap_list_train, mipgap_list_val, mipgap_list_test = split_train_val_test(mipgap_list, val_split=0.8)
    # runtime_list_train, runtime_list_val, runtime_list_test = split_train_val_test(runtime_list, val_split=0.8)
    # status_list_train, status_list_val, status_list_test = split_train_val_test(status_list, val_split=0.8)
    
    if opt_label == 'classification':
        #reduces the set of actions (subset of all optimal actions)
        action_set, counts = np.unique(stacked_list_actions[:,0], return_counts=True)
        action_set = action_set[np.where(counts>threshold_counts)]
        list_action_set = [action_set.astype(np.int32)]
        total_action_set = action_set.astype(np.int32)

        for i in range(1,N_control):
            action_set, counts = np.unique(stacked_list_actions[:,i], return_counts=True)
            action_set = action_set[np.where(counts>threshold_counts)]
            total_action_set = np.union1d(total_action_set, action_set.astype(np.int32))
            list_action_set.append(action_set.astype(np.int32))

        #creates the masking for each step of the control horizon
        list_masks = []

        for j in range(N_control):
            mask = np.zeros(total_action_set.shape, dtype=np.int32)
            for i in range(list_action_set[j].shape[0]):
                mask[np.where(total_action_set==list_action_set[j][i])] = 1
            list_masks.append(mask)
            
        num_actions = total_action_set.shape[0]
        print('num_actions =', num_actions)
        
        stacked_states_reduced_train, stacked_actions_reduced_train, cntr_outlier_train = reduce_Dataset(stacked_states_train, stacked_actions_train, total_action_set, N_control)
        stacked_states_reduced_val, stacked_actions_reduced_val, cntr_outlier_val = reduce_Dataset(stacked_states_val, stacked_actions_val, total_action_set, N_control)
        stacked_states_reduced_test, stacked_actions_reduced_test, cntr_outlier_test = reduce_Dataset(stacked_states_test, stacked_actions_test, total_action_set, N_control)
        
        stacked_actions_reduced_train = np.array(stacked_actions_reduced_train)
        stacked_actions_reduced_val = np.array(stacked_actions_reduced_val)
        stacked_actions_reduced_test = np.array(stacked_actions_reduced_test)
    
        stacked_states_train_tensor = torch.tensor(np.array(stacked_states_reduced_train), dtype=torch.float32)
        stacked_states_val_tensor = torch.tensor(np.array(stacked_states_reduced_val), dtype=torch.float32)
        stacked_states_test_tensor = torch.tensor(np.array(stacked_states_reduced_test), dtype=torch.float32) 
        
        print('number of training points (before reduction): %d' %stacked_actions_train.shape[0])
        print('number of validation points (before reduction): %d' %stacked_actions_val.shape[0])
        print('number of test points (before reduction): %d' %stacked_actions_test.shape[0])
        print('number of training points (after reduction): %d' %stacked_states_train_tensor.shape[0])
        print('number of validation points (after reduction): %d' %stacked_states_val_tensor.shape[0])
        print('number of test points (after reduction): %d' %stacked_states_test_tensor.shape[0])
        print('cntr_outlier_train: %d\t' %cntr_outlier_train, 'cntr_outlier_val: %d' %cntr_outlier_val, 'cntr_outlier_test: %d' %cntr_outlier_test)

    #TODO: finish implementation
    elif opt_label == 'regression':
        list_masks = []
        total_action_set = []
        stacked_states_train_tensor = torch.tensor(np.array(stacked_states_train), dtype=torch.float32)
        stacked_states_val_tensor = torch.tensor(np.array(stacked_states_val), dtype=torch.float32)
        stacked_states_test_tensor = torch.tensor(np.array(stacked_states_test), dtype=torch.float32) 

    # del stacked_actions_train, stacked_actions_val

    print('data-processing finished')
    
    dict_data = {
        'N': N,
        'N_control': N_control,
        'stacked_states_train_tensor': stacked_states_train_tensor,
        'stacked_states_val_tensor': stacked_states_val_tensor,
        'stacked_states_test_tensor': stacked_states_test_tensor,
        'stacked_actions_reduced_train': stacked_actions_reduced_train,
        'stacked_actions_reduced_val': stacked_actions_reduced_val,
        'stacked_actions_reduced_test': stacked_actions_reduced_test,
        'list_masks': list_masks,
        'state_min': state_min,
        'state_max': state_max,
        'input_size': input_size,
        'total_action_set': total_action_set
    }
    
    return dict_data