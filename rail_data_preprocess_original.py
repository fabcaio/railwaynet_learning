import numpy as np
import torch

from rail_rl_env import action_dict
from rail_fun import norm_state, preprocess_state

#this should be in rail_fun
inv_action_dict = {tuple(v[0]): int(k) for k, v in action_dict.items()}

def decompress_minlp_info(minlp_info_compressed, N_control):
    
    # input minlp_info_compressed[i]
    
    if N_control==18: #(N=20)    
        state_n = minlp_info_compressed[0:3*38].reshape(3,38)    
        state_rho = minlp_info_compressed[3*38 : 3*38 + 3*21*38].reshape(3,21,38)    
        state_depot = minlp_info_compressed[3*38 + 3*21*38 : 3*38 + 3*21*38 + 3].reshape(3,)    
        state_l = minlp_info_compressed[3*38 + 3*21*38 + 3 : 3*38 + 3*21*38 + 3 + 3*3*38].reshape(3,3,38)   
        delta_minlp = minlp_info_compressed[3*38 + 3*21*38 + 3 + 3*3*38 : 3*38 + 3*21*38 + 3 + 3*3*38 + 12*18].reshape(N_control,12)    
        obj_val = minlp_info_compressed[-4]
        mipgap = minlp_info_compressed[-3]
        runtime = minlp_info_compressed[-2]
        status = minlp_info_compressed[-1]
    elif N_control==38: #(N=40)
        state_n = minlp_info_compressed[0:3*38].reshape(3,38)    
        state_rho = minlp_info_compressed[3*38 : 3*38 + 3*41*38].reshape(3,41,38)    
        state_depot = minlp_info_compressed[3*38 + 3*41*38 : 3*38 + 3*41*38 + 3].reshape(3,)    
        state_l = minlp_info_compressed[3*38 + 3*41*38 + 3 : 3*38 + 3*41*38 + 3 + 3*3*38].reshape(3,3,38)   
        delta_minlp = minlp_info_compressed[3*38 + 3*41*38 + 3 + 3*3*38 : 3*38 + 3*41*38 + 3 + 3*3*38 + 12*38].reshape(N_control,12)    
        obj_val = minlp_info_compressed[-4]
        mipgap = minlp_info_compressed[-3]
        runtime = minlp_info_compressed[-2]
        status = minlp_info_compressed[-1]
    else:
        print('Invalid prediction horizon (N must be 20 or 40).')
    
    return state_n, state_rho, state_depot, state_l, delta_minlp, obj_val, mipgap, runtime, status

def build_list_action(delta: np.array, N_control: np.int32) -> list:
    delta = delta.astype(np.int32)
    list_action = [inv_action_dict[tuple(delta[0])]]
    for i in range(1, N_control):
        list_action.append(inv_action_dict[tuple(delta[i])])
        
    return list_action

#tests conversion from original list_actions to reduced list of actions
def reduce_list_actions(list_action, total_action_set, N_control):
    
    list_actions_reduced = np.zeros((N_control,), dtype=np.int32)
    for i in range(N_control):
        list_actions_reduced[i] = np.where(list_action[i] == total_action_set)[0][0]
        
    return list_actions_reduced
# i = 0
# list_action = stacked_list_actions[i].astype(np.int32)
# print(list_action)
# list_actions_reduced = reduce_list_actions(list_action)
# print(list_actions_reduced)
# print(total_action_set[list_actions_reduced])
# print(list_action == total_action_set[list_actions_reduced])

def split_train_validation(stacked_states, stacked_list_actions, split_position=0.8):
       
    N_datapoints = stacked_states.shape[0]
    
    stacked_states_train = stacked_states[:int(np.ceil(N_datapoints*split_position))]
    # stacked_labels_train = stacked_labels[:int(np.ceil(N_datapoints*split_position))]
    stacked_actions_train = stacked_list_actions[:int(np.ceil(N_datapoints*split_position))]
    
    stacked_states_val = stacked_states[int(np.ceil(N_datapoints*split_position)):]
    # stacked_labels_val = stacked_labels[int(np.ceil(N_datapoints*split_position)):]
    stacked_actions_val = stacked_list_actions[int(np.ceil(N_datapoints*split_position)):]
    
    return stacked_states_train, stacked_actions_train, stacked_states_val, stacked_actions_val

######################################################################################################################

def get_preprocessed_data(opt, threshold_counts, N):
    # import numpy as np
    # N=40
    # minlp_info_compressed = np.load('data_milp//data_milp_N%.2d_%.2d.npy' %(N, 0), allow_pickle=True)
    # for job_idx in range(1,28):
    #     tmp_vector = np.load('data_milp//data_milp_N%.2d_%.2d.npy' %(N, job_idx), allow_pickle=True)
    #     minlp_info_compressed = np.concatenate((minlp_info_compressed, tmp_vector))
    # minlp_info_compressed = minlp_info_compressed[:120000, :]
    # np.save('data_milp//data_milp_ol_N%.2d_condensed.npy' %N, minlp_info_compressed, allow_pickle=True)
    
    N_control = N - 2
    
    str_data = 'data_optimal//data_%s_N%d_condensed.npy' %(opt, N)
    minlp_info_compressed = np.load(str_data, allow_pickle=True)
        
    N_datapoints = minlp_info_compressed.shape[0]

    state_min = np.min(minlp_info_compressed, axis=0)
    state_max = np.max(minlp_info_compressed, axis=0)

    state_n_min, state_rho_min, state_depot_min, state_l_min, _, _, _, _, _ = decompress_minlp_info(state_min,N_control)
    state_min_reduced = preprocess_state(state_n_min, state_rho_min, state_depot_min, state_l_min)

    state_n_max, state_rho_max, state_depot_max, state_l_max, _, _, _, _, _ = decompress_minlp_info(state_max,N_control)
    state_max_reduced = preprocess_state(state_n_max, state_rho_max, state_depot_max, state_l_max)

    # does one iteration of the state preprocessing to find out the input_size of the vector
    normalized_state = norm_state(minlp_info_compressed[0], state_min, state_max)
    state_n_norm, state_rho_norm, state_depot_norm, state_l_norm, _, _, _, _, _ = decompress_minlp_info(normalized_state,N_control)
    state_learning = preprocess_state(state_n_norm, state_rho_norm, state_depot_norm, state_l_norm)
    input_size = state_learning.shape[0]

    stacked_states = np.zeros((N_datapoints,1,input_size))
    # stacked_states = np.zeros((N_datapoints,N_control,input_size))

    stacked_list_actions = np.zeros((N_datapoints, N_control))

    # num_actions = int(list(action_dict)[-1])+1 #index starts at 0
    # stacked_labels = np.zeros((N_datapoints,N_control,num_actions)) # occupies to much RAM

    stacked_obj_val = np.zeros((N_datapoints,))
    stacked_mipgap = np.zeros((N_datapoints,))
    stacked_runtime = np.zeros((N_datapoints,))
    stacked_status = np.zeros((N_datapoints,))

    for j in range(N_datapoints):
        
        #input preprocessing
        
        _, _, _, _, delta_minlp, obj_val, mipgap, runtime, status = decompress_minlp_info(minlp_info_compressed[j],N_control)
        normalized_state = norm_state(minlp_info_compressed[j], state_min, state_max)
        state_n_norm, state_rho_norm, state_depot_norm, state_l_norm, _, _, _, _, _ = decompress_minlp_info(normalized_state,N_control)
        
        stacked_states[j,0,:] = preprocess_state(state_n_norm, state_rho_norm, state_depot_norm, state_l_norm)
        # stacked_states[j,1:,:] = np.zeros((N_control-1, input_size)) # padding is done during training to save RAM
        
        #label preprocessing (label is preprocessed later to save RAM)

        # idx_0 = j*np.ones(N_control, dtype=np.int32)
        # idx_1 = np.arange(N_control)
        # idx_2 = build_list_action(np.round(delta_minlp,2), N_control)
        # idx_list = list(zip(idx_0, idx_1, idx_2))

        # for k in range(N_control):
        #     stacked_labels[idx_list[k]] = 1
            
        stacked_obj_val[j] = obj_val
        stacked_mipgap[j] = mipgap
        stacked_runtime[j] = runtime
        stacked_status[j] = status
        
        stacked_list_actions[j] = build_list_action(np.round(delta_minlp,2), N_control)
        
    del minlp_info_compressed

    #reduces the set of actions (subset of all optimal actions)
    # threshold_counts = 50 # to eliminate actions that appear rarely
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

    stacked_states_train, stacked_actions_train, stacked_states_val, stacked_actions_val = split_train_validation(stacked_states, stacked_list_actions, split_position=0.8)

    # stacked_actions_reduced_train = np.zeros(stacked_actions_train.shape)
    # for i in range(stacked_actions_train.shape[0]):
    #     stacked_actions_reduced_train[i] = reduce_list_actions(stacked_actions_train[i],total_action_set)
        
    # stacked_actions_reduced_val = np.zeros(stacked_actions_val.shape)
    # for i in range(stacked_actions_val.shape[0]):
    #     stacked_actions_reduced_val[i] = reduce_list_actions(stacked_actions_val[i],total_action_set)
    
    stacked_actions_reduced_train = []
    stacked_states_reduced_train = []
    cntr_outlier_train = 0
    for i in range(stacked_actions_train.shape[0]):
        try:
            stacked_actions_reduced_train.append(reduce_list_actions(stacked_actions_train[i],total_action_set, N_control))
            stacked_states_reduced_train.append(stacked_states_train[i])
        except:
            cntr_outlier_train +=1
       
    stacked_actions_reduced_val = []
    stacked_states_reduced_val = []
    cntr_outlier_val = 0        
    for i in range(stacked_actions_val.shape[0]):
        try:
            stacked_actions_reduced_val.append(reduce_list_actions(stacked_actions_val[i],total_action_set, N_control))
            stacked_states_reduced_val.append(stacked_states_val[i])
        except:
            cntr_outlier_val +=1
            
    print('number of training points (before reduction): %d' %stacked_actions_train.shape[0])
    print('number of validation points (before reduction): %d' %stacked_actions_val.shape[0])
        
    del stacked_actions_train, stacked_actions_val

    # stacked_states_train_tensor = torch.tensor(stacked_states_train, dtype=torch.float32)
    # stacked_states_val_tensor = torch.tensor(stacked_states_val, dtype=torch.float32)
    
    stacked_actions_reduced_train = np.array(stacked_actions_reduced_train)
    stacked_actions_reduced_val = np.array(stacked_actions_reduced_val)
    
    stacked_states_reduced_train = np.array(stacked_states_reduced_train)
    stacked_states_reduced_val = np.array(stacked_states_reduced_val)    
    
    stacked_states_train_tensor = torch.tensor(stacked_states_reduced_train, dtype=torch.float32)
    stacked_states_val_tensor = torch.tensor(stacked_states_reduced_val, dtype=torch.float32)
    
    # stacked_states_reduced_train_tensor = torch.tensor(stacked_states_reduced_train, dtype=torch.float32)
    # stacked_states_reduced_val_tensor = torch.tensor(stacked_states_reduced_val, dtype=torch.float32)
    
    print('number of training points (after reduction): %d' %stacked_states_train_tensor.shape[0])
    print('number of validation points (after reduction): %d' %stacked_states_val_tensor.shape[0])
    print('cntr_outlier_train: %d\t' %cntr_outlier_train, 'cntr_outlier_val: %d' %cntr_outlier_val)
    print('data-processing finished')
    
    return N, N_control, stacked_states_train, stacked_states_val, stacked_actions_reduced_train, stacked_actions_reduced_val, list_masks, stacked_states_train_tensor, stacked_states_val_tensor, state_min_reduced, state_max_reduced, input_size, total_action_set#, stacked_states_reduced_train_tensor, stacked_states_reduced_val_tensor