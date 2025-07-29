from rail_training_reduced import Network, Network_CE, Network_mask1, Network_mask2, Network_mask3, train_network, test_accuracy
import datetime
from rail_data_preprocess_reduced  import get_preprocessed_data_reduced
import torch
import numpy as np
import sys

N = 40
testing = True

opt_preprocess = 'reduced'
opt_label = 'classification' #(regression is not implemented yet)

if testing == True:
    opt = 'milp_ol'
    network_type = 'Network'
    hidden_size = 32
    dropout = 0.5
    seed = 4
    training_time = 1*60
    idx = 998
    n_threads = 4
    LR_scheduler = True
    opt_state=1
else:
    opt = sys.argv[1]
    network_type = sys.argv[2]
    hidden_size = int(sys.argv[3])
    dropout = float(sys.argv[4])
    seed = int(sys.argv[5])
    training_time = float(sys.argv[6]) - 45*60 # exclude time for data-preprocessing and testing the accuracy
    idx = int(sys.argv[7])
    n_threads = int(sys.argv[8])
    LR_scheduler = int(sys.argv[9])
    opt_state = int(sys.argv[10])

device='cpu'
torch.set_num_threads(n_threads)

num_layers=1
lr=1e-3
batch_size=32 # 4 works well too
threshold_counts = 50

print('opt: %s, network_type: %s, hidden_size: %d, dropout: %.2f, seed: %d, training time: %d, idx: %d, n_threads: %d, LR_scheduler: %d' %(opt, network_type, hidden_size, dropout, seed, training_time, idx, n_threads, LR_scheduler))
print('lr: %.5f, batch_size: %d, threshold_counts: %d, N: %d' %(lr, batch_size, threshold_counts, N))
    
dict_data = get_preprocessed_data_reduced(opt, threshold_counts, N, opt_state, opt_label,testing)

N_control = dict_data['N_control']
stacked_actions_reduced_val = dict_data['stacked_actions_reduced_val']
list_masks = dict_data['list_masks']
stacked_states_val_tensor = dict_data['stacked_states_val_tensor']
input_size = dict_data['input_size']
total_action_set = dict_data['total_action_set']

num_actions = total_action_set.shape[0]
seq_len=N_control

np.random.seed(seed)
torch.manual_seed(seed)

if network_type == 'Network':
    network = Network(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout)
elif network_type == 'Network_CE':
    network = Network_CE(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout)
elif network_type == 'Network_mask1':
    network = Network_mask1(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout, list_masks, device)
elif network_type == 'Network_mask2':
    network = Network_mask2(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout, list_masks, device)
elif network_type == 'Network_mask3':
    network = Network_mask3(input_size, hidden_size, num_layers, lr, num_actions, batch_size, dropout, list_masks, device)
else:
    raise KeyError('Network type has not been found.')
print(network)

print('number of parameters: ', network.count_parameters())

network, network_best, loss_train, loss_val, number_iterations, training_time = train_network(network, dict_data, training_time, N_control, LR_scheduler, device)

per_accuracy = test_accuracy(network_best, 100, stacked_states_val_tensor, stacked_actions_reduced_val, N_control, device)

x = datetime.datetime.now()
date_str = '_%.2d%.2d_%.2d%.2d' %(x.day, x.month, x.hour, x.minute)

parameters = 'number of iterations: %d, training time: %.2f, batch_size: %d, hidden_size: %d, num_actions: %d, N: %d, lr:%.5f, threshold_counts: %d, seed: %d' %(number_iterations, training_time, batch_size, hidden_size, num_actions, N, lr, threshold_counts, seed)
print(parameters)

dict_info = {
    'loss_train': loss_train,
    'loss_val': loss_val,
    'number_iterations': number_iterations,
    'lr': lr,
    'hidden_size': hidden_size,
    'network_type': network_type,
    'dropout': dropout,
    'seed': seed,
    'training_time': training_time,
    'per_accuracy': per_accuracy,
    'opt': opt,
    'threshold_counts': threshold_counts,
    'list_masks': list_masks,
    'batch_size': batch_size,
    'num_actions': num_actions,
    'num_layers': num_layers,
    'input_size': input_size,
    'device': device,
    'idx': idx,
    'n_threads': n_threads,
    'N': N,
    'N_control': N_control,
    'LR_scheduler': LR_scheduler,
    'parameters': parameters    
}

if testing==True:    
    torch.save(network_best.state_dict(), 'training_data_%s//' %opt_preprocess + opt + '_N%d' %N + '_%.3d' %idx + '_weight')
    np.save('training_data_%s//' %opt_preprocess + opt + '_N%d' %N + '_%.3d' %idx + '_info', dict_info)
else:   
    torch.save(network_best.state_dict(), '//scratch//cfoliveiradasi//railway_largescale//training_data_%s//' %opt_preprocess + opt + '_N%d' %N + '_%.3d' %idx + '_weight')
    np.save('//scratch//cfoliveiradasi//railway_largescale//training_data_%s//' %opt_preprocess + opt + '_N%d' %N + '_%.3d' %idx + '_info', dict_info)
    
# network.load_state_dict(torch.load('weights//railway_supervised_large_256_N20_26_07'))