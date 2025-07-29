import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import copy

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, lr, n_actions, batch_size, dropout, batch_first=True):
        super(Network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.batch_first = batch_first
        
        self.dense_input = nn.Linear(input_size, hidden_size)
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.dense = nn.Linear(hidden_size, n_actions)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss(reduction='mean')
        
    def forward(self, input, h0, c0):
        
        # input = torch.randn(batch_size, seq_len, input_size)
        # h0 = torch.randn(num_layers, batch_size, hidden_size)
        # c0 = torch.randn(num_layers, batch_size, hidden_size)
        
        output = self.dense_input(input)
        
        output = self.dropout1(output)

        output, _ = self.lstm(output, (h0, c0))
        
        output = self.dropout2(output)       
        
        output = self.dense(output)        
                
        output = F.softmax(output, dim=2)

        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class Network_CE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, lr, n_actions, batch_size, dropout, batch_first=True):
        super(Network_CE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.batch_first = batch_first
        
        self.dense_input = nn.Linear(input_size, hidden_size)
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.dense = nn.Linear(hidden_size, n_actions)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, input, h0, c0):
        
        # input = torch.randn(batch_size, seq_len, input_size)
        # h0 = torch.randn(num_layers, batch_size, hidden_size)
        # c0 = torch.randn(num_layers, batch_size, hidden_size)
        
        output = self.dense_input(input)
        
        output = self.dropout1(output)

        output, _ = self.lstm(output, (h0, c0))
        
        output = self.dropout2(output)       
        
        output = self.dense(output)        
                
        # output = F.softmax(output, dim=2)

        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
       
class Network_mask1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, lr, n_actions, batch_size, dropout, list_masks, device, batch_first=True):
        super(Network_mask1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.batch_first = batch_first
        
        self.dense_input = nn.Linear(input_size, hidden_size)
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.dense = nn.Linear(hidden_size, n_actions)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        mask = torch.tensor(np.array(list_masks, dtype=np.int32), device=device).unsqueeze(axis=0)
        masks_tensor = mask
        for i in range(1,batch_size):
            masks_tensor = torch.concatenate((masks_tensor, mask), dim=0)
        self.mask = masks_tensor
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss(reduction='mean')
        
    def forward(self, input, h0, c0):
        
        # input = torch.randn(batch_size, seq_len, input_size)
        # h0 = torch.randn(num_layers, batch_size, hidden_size)
        # c0 = torch.randn(num_layers, batch_size, hidden_size)
        
        output = self.dense_input(input)
        
        output = self.dropout1(output)

        output, _ = self.lstm(output, (h0, c0))
        
        output = self.dropout2(output)       
        
        output = self.dense(output)
        
        # output = output * self.mask        # COMMENT THIS FOR RESULTS FOR THE OPEN-LOOP TEST
    
        idx_zeros = torch.where(output==0)
        output[idx_zeros] = -1000 #this forces the output to be zero in the mask indices
                
        output = F.softmax(output, dim=2)

        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class Network_mask2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, lr, n_actions, batch_size, dropout, list_masks, device, batch_first=True):
        super(Network_mask2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.batch_first = batch_first
        
        self.dense_input = nn.Linear(input_size, hidden_size)
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.dense = nn.Linear(hidden_size, n_actions)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        mask = torch.tensor(np.array(list_masks, dtype=np.int32), device=device).unsqueeze(axis=0)
        masks_tensor = mask
        for i in range(1,batch_size):
            masks_tensor = torch.concatenate((masks_tensor, mask), dim=0)
        self.mask = masks_tensor
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss(reduction='mean')
        
    def forward(self, input, h0, c0):
        
        # input = torch.randn(batch_size, seq_len, input_size)
        # h0 = torch.randn(num_layers, batch_size, hidden_size)
        # c0 = torch.randn(num_layers, batch_size, hidden_size)
        
        output = self.dense_input(input)
        
        output = self.dropout1(output)

        output, _ = self.lstm(output, (h0, c0))
        
        output = self.dropout2(output)       
        
        output = self.dense(output)        
                
        output = F.softmax(output, dim=2)
        
        output = output * self.mask

        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class Network_mask3(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, lr, n_actions, batch_size, dropout, list_masks, device, batch_first=True):
        super(Network_mask3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.batch_first = batch_first
        
        self.dense_input = nn.Linear(input_size, hidden_size)
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.dense = nn.Linear(hidden_size, n_actions)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        mask = torch.tensor(np.array(list_masks, dtype=np.int32), device=device).unsqueeze(axis=0)
        masks_tensor = mask
        for i in range(1,batch_size):
            masks_tensor = torch.concatenate((masks_tensor, mask), dim=0)
        self.mask = masks_tensor
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss(reduction='mean')
        
    def forward(self, input, h0, c0):
        
        # input = torch.randn(batch_size, seq_len, input_size)
        # h0 = torch.randn(num_layers, batch_size, hidden_size)
        # c0 = torch.randn(num_layers, batch_size, hidden_size)
        
        output = self.dense_input(input)
        
        output = self.dropout1(output)

        output, _ = self.lstm(output, (h0, c0))
        
        output = self.dropout2(output)       
        
        output = self.dense(output)
        
        output = output * self.mask        
    
        idx_zeros = torch.where(output==0)
        output[idx_zeros] = -1000 #this forces the output to be zero in the mask indices
                
        output = F.softmax(output, dim=2)

        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
def build_label(list_action, num_actions, N_control):    
    #given the list of actions, it builds a single label
    #input: stacked_list_actions[i]
    
    # label = np.zeros((1,N_control,num_actions))
    label = np.zeros((N_control,num_actions))
    
    # idx_0 = j*np.ones(N_control, dtype=np.int32)
    idx_1 = np.arange(N_control)
    idx_2 = list_action.astype(np.int32)
    # idx_2 = reduce_list_actions(list_action.astype(np.int32))
    # idx_list = list(zip(idx_0, idx_1, idx_2))
    idx_list = list(zip(idx_1, idx_2))

    for k in range(N_control):
        label[idx_list[k]] = 1
        
    return label

def build_stacked_label(batch_list_actions,num_actions, N_control):
    #builds the label in batches for learning
    
    batch_size = batch_list_actions.shape[0]
    batch_labels = np.zeros((batch_size, N_control, num_actions))
    
    for i in range(batch_size):
        batch_labels[i] = build_label(batch_list_actions[i],num_actions, N_control)
        # batch_labels[i] = tmp_vector
    
    return batch_labels

def build_pad_batch_states_tensor(batch_states, batch_size, input_size, N_control):
    
    #pads the state sequence with zeros along the prediction horizon
    #input: stacked_states[0:50] (tensor)
    
    b = torch.zeros(batch_size, N_control-1, input_size)
    pad_batch_states = torch.cat((batch_states.unsqueeze(dim=1),b), dim=1)
    
    return pad_batch_states

def moving_average(a, n):
    #a: vector
    #n: moving average window size
    
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
    
# training loop

def train_network(network, dict_data, time_training, N_control, LR_schedule, device):
      
    network = network.to(device)
       
    stacked_actions_reduced_train = dict_data['stacked_actions_reduced_train']
    stacked_actions_reduced_val = dict_data['stacked_actions_reduced_val']
    stacked_states_train_tensor = dict_data['stacked_states_train_tensor']
    stacked_states_val_tensor = dict_data['stacked_states_val_tensor']
    
    loss_train = []
    loss_val = []
    
    num_batches_per_epoch = int(stacked_actions_reduced_train.shape[0]/network.batch_size)
    print(num_batches_per_epoch)

    h0 = torch.zeros(network.num_layers, network.batch_size, network.hidden_size, device=device)
    c0 = torch.zeros(network.num_layers, network.batch_size, network.hidden_size, device=device)
    
    loss_val_epoch_best = np.inf
    idx_val_epoch_best = 0
    idx_change_lr = 0

    start_time = time.time()
    network_best = copy.deepcopy(network)
    
    i = 0
    while time.time() < start_time + time_training:
    # for i in range(N_iter):
        
        network.train()

        batch = np.random.choice(stacked_states_train_tensor.shape[0], network.batch_size, replace=False)

        state_batch_train = build_pad_batch_states_tensor(stacked_states_train_tensor[batch], network.batch_size, network.input_size, N_control).to(device)
        labels_batch_train = build_stacked_label(stacked_actions_reduced_train[batch],network.n_actions, N_control)
        labels_batch_train = torch.tensor(labels_batch_train, dtype=torch.float32).to(device)

        net_eval = network(state_batch_train, h0, c0)

        loss = network.loss(net_eval, labels_batch_train)

        network.zero_grad()      
        loss.backward()
        network.optimizer.step()
        
        # loss_tmp = loss.detach().to('cpu').numpy() #uses too much RAM
        loss_tmp = loss.detach().item()
        loss_train.append(loss_tmp)
        
        del loss, loss_tmp, net_eval, state_batch_train, labels_batch_train, batch
        
        #evaluation on validation data
        with torch.no_grad():
            network.eval()
            batch = np.random.choice(stacked_states_val_tensor.shape[0], network.batch_size, replace=False)
            state_batch_val = build_pad_batch_states_tensor(stacked_states_val_tensor[batch], network.batch_size, network.input_size, N_control).to(device)
            labels_batch_val = build_stacked_label(stacked_actions_reduced_val[batch],network.n_actions, N_control)
            labels_batch_val = torch.tensor(labels_batch_val, dtype=torch.float32).to(device)
            net_eval = network(state_batch_val, h0, c0)
            # loss = network.loss(net_eval, labels_batch_val).detach().to('cpu').numpy() # user too much RAM
            loss = network.loss(net_eval, labels_batch_val).detach().item()
            loss_val.append(loss)
            
            del net_eval, loss, labels_batch_val, state_batch_val, batch 
            
            # check loss per epoch (1epoch approx 3000 mini-batches)
            if i%(num_batches_per_epoch-1) == 0:
                loss_val_epoch = np.mean(loss_val[-num_batches_per_epoch:])
                if loss_val_epoch < loss_val_epoch_best:
                    loss_val_epoch_best = loss_val_epoch
                    idx_val_epoch_best = i
                    network_best = copy.deepcopy(network)
                    print('New lowest validation loss per epoch.' + ' Loss: %.6f' %loss_val_epoch_best)
            
        # learning rate scheduler : decay on plateau (after 5 epochs of plateau, decrease the learning rate by half)
        if LR_schedule == True:
            if (i - idx_val_epoch_best) > 5*num_batches_per_epoch and (i-idx_change_lr) > 5*num_batches_per_epoch:
                idx_change_lr = i
                network.optimizer.param_groups[0]['lr'] = network.optimizer.param_groups[0]['lr']/2
                print('Learning scheduler: plateau detected. Learning rate reduced by half.')
                print('Learning rate: ', network.optimizer.param_groups[0]['lr'])
                
            if network.optimizer.param_groups[0]['lr'] < 1e-9: # minimum learning rate
                break
            
            #stops after 10 minutes of non-improvement (early termination to save computation time)
            # if (i - idx_val_epoch_best) > 50*num_batches_per_epoch:
            #     print('Training stopped because the validation loss did not decrease for 50 epochs')
            #     break
                    
        if i%1000 == 0:
            loss_train_avg = np.mean(loss_train[-100:])
            loss_val_avg = np.mean(loss_val[-100:])
            current_time = time.time() - start_time
            elapsed_time_hour = current_time/3600
            elapsed_time_minutes = (elapsed_time_hour % 1)*60
            elapsed_time_seconds = (elapsed_time_minutes % 1)*60
            print('number of iterations: %d, \t training loss: %.6f, \t validation loss: %.6f, \t elapsed time (hr/min/sec): %.2d:%.2d:%.2d' %(i, loss_train_avg, loss_val_avg, elapsed_time_hour, elapsed_time_minutes, elapsed_time_seconds))
        
        i+=1
    print('training is completed.')
    
    number_iterations = i
    training_time = time.time()-start_time
    print('number of training iterations: %d' %number_iterations, 'total_training_time: %.2f' %(training_time))
                
    return network, network_best, loss_train, loss_val, number_iterations, training_time

def test_accuracy(network, N_iter, stacked_states_val_tensor, stacked_actions_reduced_val, N_control, device):
    
    network.to(device)
    
    list_check = []
    network.eval()
    
    h0 = torch.zeros(network.num_layers, network.batch_size, network.hidden_size).to(device)
    c0 = torch.zeros(network.num_layers, network.batch_size, network.hidden_size).to(device)
    
    for _ in range(N_iter):
        batch = np.random.choice(stacked_states_val_tensor.shape[0], network.batch_size, replace=False)
        state_batch_val = build_pad_batch_states_tensor(stacked_states_val_tensor[batch], network.batch_size, network.input_size, N_control).to(device)
        labels_batch_val = build_stacked_label(stacked_actions_reduced_val[batch],network.n_actions, N_control)
        labels_batch_val = torch.tensor(labels_batch_val, dtype=torch.float32).to(device)

        net_eval = network(state_batch_val, h0, c0)
        
        for i in range(network.batch_size):
            for j in range(N_control):
                list_check.append((net_eval[i,j,:].argmax() == labels_batch_val[i,j,:].argmax()).cpu().numpy())
                
    num_correct = np.sum(list_check)
    num_total = np.prod(net_eval.shape[:2])*N_iter
    per_correct = num_correct/num_total * 100
    print(num_correct, 'out of', num_total)
    print('percentage of correct labels: %.2f' %per_correct)
    
    return per_correct