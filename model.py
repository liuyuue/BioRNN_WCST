import numpy as np; np.set_printoptions(precision=2)
import torch; torch.set_printoptions(precision=2)
import torch.nn as nn
import time
import sys
import itertools
import random; random.seed(0)
import datetime
import copy
import scipy




class BioRNN(torch.nn.Module):
    """ 
    RNN with elements from neurobiology
    """

    def __init__(self, hp):
        super().__init__()
        
#         self.cell_group_list = ['sr_esoma', 'sr_edend', 'sr_pv', 'sr_sst', 'sr_vip',
#                                 'pfc_esoma', 'pfc_edend', 'pfc_pv', 'pfc_sst', 'pfc_vip']
        self.cell_group_list = hp['cell_group_list']
        
        # Hyperparameters
        self.n = {}
        self.n['input'], self.n['output'], self.n['output_rule'] = hp['n_input'], hp['n_output'], hp['n_output_rule']
        self.n_branches = hp['n_branches']
        for cg in self.cell_group_list:
            self.n[cg] = hp['n_'+cg]
        self.decay = hp['dt']/hp['tau']
        self.initialization_weights = hp['initialization_weights']
        self.initialzation_bias = hp['initialization_bias']
        self.train_rule = hp['train_rule']    # if train PFC to produce the rule
        self.explicit_rule = hp['explicit_rule']
        self.batch_size = hp['batch_size']
        self.target_perf = hp['target_perf']
        self.network_noise = hp['network_noise']
        self.learning_rate = hp['learning_rate']
        self.save_name = hp['save_name']
        # regularization parameters
        self.l1_h = hp['l1_h']
        self.l2_h = hp['l2_h']
        self.l1_weights = hp['l1_weight']
        self.l2_weights = hp['l2_weight']
        self.l2_rec_e_weight_sr = hp['l2_rec_e_weight_sr']
        self.l2_h_sr = hp['l2_h_sr']
        self.loss_type=hp['loss_type']
        self.mglur = hp['mglur']    # True or False
        self.divide_sr_sst_vip = hp['divide_sr_sst_vip']
        self.no_pfcesoma_to_srsst = hp['no_pfcesoma_to_srsst']
        self.no_pfcesoma_to_sredend = hp['no_pfcesoma_to_sredend']
        self.no_pfcesoma_to_srpv = hp['no_pfcesoma_to_srpv']
        self.no_srsst_to_srvip = hp['no_srsst_to_srvip']
        self.structured_sr_sst_to_sr_edend = hp['structured_sr_sst_to_sr_edend']
        self.structured_sr_sst_to_sr_edend_branch_specific = hp['structured_sr_sst_to_sr_edend_branch_specific']
        self.sr_sst_high_bias = hp['sr_sst_high_bias']
        self.fdbk_to_vip = hp['fdbk_to_vip']
        self.dend_nonlinearity = hp['dend_nonlinearity']
        self.trainable_dend2soma = hp['trainable_dend2soma']
        self.divisive_dend_inh = hp['divisive_dend_inh']
        self.divisive_dend_ei = hp['divisive_dend_ei']
        self.scale_down_init_wexc = hp['scale_down_init_wexc']
        self.prev_rew_mag = 1
        self.prev_stim_mag = 1
        self.prev_choice_mag = 1
        self.activation = hp['activation']
        self.k_relu_satu = hp['k_relu_satu']
        self.sparse_pfcesoma_to_srvip = hp['sparse_pfcesoma_to_srvip']
            
        # nonlinearity
#         if hp['activation'] == 'relu':    # Type of activation runctions, relu, softplus, tanh, elu
#             self.nonlinearity_soma = nn.ReLU()
#         elif hp['activation'] == 'tanh': 
#             self.nonlinearity_soma = nn.Tanh()
#         elif hp['activation'] == 'sigmoid':
#             self.nonlinearity_soma = nn.Sigmoid()
#         elif hp['activation'] == 'relu_satu':
#             pass
#         else: 
#             raise NotImplementedError
            
        # compute the neuron indices for each group
        start_idx = 0
        self.cg_idx = {}
        for cg in self.cell_group_list:
            self.cg_idx[cg] = np.arange(start_idx, start_idx+self.n[cg])
            start_idx = start_idx + self.n[cg]
        self.total_n_neurons = start_idx
        
        # mGluRs
        if self.mglur==True:
            print('Model includes mGluRs.\n')
            self.mglur_e_idx = self.cg_idx['pfc_edend']    # let PFC E neurons have mGluRs
            self.tau_me = np.logspace(start=np.log10(100), stop=np.log10(5000), num=len(self.mglur_e_idx)//self.n_branches)    # time constants of the mGluR E current, ms
            self.tau_me = np.tile(self.tau_me, self.n_branches)
            print('initializing. self.tau_me shape:  {}\n self.tau_me = {}\n'.format(self.tau_me.shape, self.tau_me))
            self.alpha_me = hp['dt']/self.tau_me
            self.alpha_me = nn.Parameter(torch.Tensor(self.alpha_me), requires_grad=False)
        else:
            print('mGluRs set to 0.\n')
        
            
            
        # extract the indices for dendrites, soma, E and I (use later)
        self.soma_idx = []
        self.dend_idx = []
        for cg in self.cell_group_list:
            if 'dend' in cg:
                self.dend_idx.extend(self.cg_idx[cg])
            else:
                self.soma_idx.extend(self.cg_idx[cg])
        
        self.e_idx = []
        self.i_idx = []
        for cg in self.cell_group_list:
            if 'soma' in cg or 'dend' in cg:
                self.e_idx.extend(self.cg_idx[cg])
            else:
                self.i_idx.extend(self.cg_idx[cg])
        
        # cell targetd by input about previous reward 
        self.pfc_idx = []
        self.sr_idx = []
        for cg in self.cell_group_list:
            if 'pfc' in cg:
                self.pfc_idx.extend(self.cg_idx[cg])
            elif 'sr' in cg: 
                self.sr_idx.extend(self.cg_idx[cg])
                
        self.dend_idx_sr = [i for i in self.dend_idx if i in self.sr_idx]
        self.dend_idx_pfc = [i for i in self.dend_idx if i in self.pfc_idx]
                
        # input-targeting cell indices
        self.input_targ_idx_sr = []    
        for cg in self.cell_group_list:
            if cg=='sr_edend' or cg=='sr_pv':    # input target only SR PV and SR Edend
                self.input_targ_idx_sr.extend(self.cg_idx[cg])
        
        self.input_targ_idx_pfc = []     # where long-range input targets in PFC
        for cg in self.cell_group_list:
            if cg=='pfc_edend' or cg=='pfc_pv':    # input target only PFC PV and PFC Edend
                self.input_targ_idx_pfc.extend(self.cg_idx[cg])
           

        # Initialize connectivity matrices
        self.w_rec = nn.Parameter(torch.Tensor(self.total_n_neurons, self.total_n_neurons), requires_grad=True)
        
        # mask matrix. 1 if e connection; 0 if no connection; -1 if i conntion. identity between dendrite and soma
        self.mask = nn.Parameter(torch.ones(self.total_n_neurons, self.total_n_neurons), requires_grad=False)
        for cgs in itertools.product(self.cell_group_list, self.cell_group_list):
            cg1 = cgs[0]
            cg2 = cgs[1]
            if (cg1=='sr_edend' and cg2=='sr_esoma') or (cg1=='pfc_edend' and cg2=='pfc_esoma'):
                self.mask[np.ix_(self.cg_idx[cg1], self.cg_idx[cg2])] = 0    # connectivity from dendrites to somas are fixed. define later
            elif self.is_connected(cg2, cg1)==False:   # if cg1 is connected to cg2 (NOTICE THE ORDER!)
                self.mask[np.ix_(self.cg_idx[cg1], self.cg_idx[cg2])] = 0    # clamp weights to 0 if two regions are not connected.
            elif self.is_connected(cg2, cg1) and ('pv' in cg1 or 'sst' in cg1 or 'vip' in cg1):    # inhibitory connection
                self.mask[np.ix_(self.cg_idx[cg1], self.cg_idx[cg2])] = -1
            else:
                self.mask[np.ix_(self.cg_idx[cg1], self.cg_idx[cg2])] = 1
        for n in range(self.total_n_neurons):
            self.mask[n,n] = 0    # no autopse
            
        # modify connectivity
        if self.no_pfcesoma_to_srsst==True:
            self.mask[np.ix_(self.cg_idx['pfc_esoma'], self.cg_idx['sr_sst'])] = 0
            print('no PFC esoma to SR SST\n')
        if self.no_pfcesoma_to_sredend==True:
            self.mask[np.ix_(self.cg_idx['pfc_esoma'], self.cg_idx['sr_edend'])] = 0
            print('no PFC esoma to SR edend\n')
        if self.no_pfcesoma_to_srpv==True:
            self.mask[np.ix_(self.cg_idx['pfc_esoma'], self.cg_idx['sr_pv'])] = 0
            print('no PFC esoma to SR PV\n')
        if self.no_srsst_to_srvip==True:
            self.mask[np.ix_(self.cg_idx['sr_sst'], self.cg_idx['sr_vip'])] = 0
            print('no SR SST to SR VIP\n')
        
        # make PFC->SR VIP projection sparse
        if self.sparse_pfcesoma_to_srvip!=1:
            print('make PFC to SR VIP connections sparse, sparsity={}'.format(self.sparse_pfcesoma_to_srvip))
            if self.sparse_pfcesoma_to_srvip>1:
                raise ValueError('sparsity cannot > 1!')
            if (self.mask[np.ix_(self.cg_idx['pfc_esoma'], self.cg_idx['sr_vip'])]==0).all()==True:
                raise ValueError('no PFC to SR VIP connections!')
            n_disconnected = int(len(self.cg_idx['pfc_esoma'])*(1-self.sparse_pfcesoma_to_srvip))    # number of PFC Esoma that do not connect to SR VIP
            for n in self.cg_idx['sr_vip']:
                disconnect_idx = random.sample(list(self.cg_idx['pfc_esoma']), n_disconnected)
                self.mask[disconnect_idx, n] = 0
                print('SR VIP #{}, disconnect with PFC Esoma {}'.format(n, disconnect_idx))
                print(len(disconnect_idx))
            
        # divide SR SST and SR VIP into two subpopulations
        if self.divide_sr_sst_vip==True:
            print('divide SR SST and SR VIP\n')
            self.cg_idx['sr_vip_1'] = np.arange(self.cg_idx['sr_vip'][0], self.cg_idx['sr_vip'][0]+len(self.cg_idx['sr_vip'])//2)
            self.cg_idx['sr_vip_2'] = np.array([n for n in self.cg_idx['sr_vip'] if n not in self.cg_idx['sr_vip_1']])
            self.cg_idx['sr_sst_1'] = np.arange(self.cg_idx['sr_sst'][0], self.cg_idx['sr_sst'][0]+len(self.cg_idx['sr_sst'])//2)
            self.cg_idx['sr_sst_2'] = np.array([n for n in self.cg_idx['sr_sst'] if n not in self.cg_idx['sr_sst_1']])
            print('sr_vip_1 idx: {},\nsr_vip_2 idx: {},\nsr_sst_1 idx: {},\nsr_sst_2 idx: {}\n'.format(self.cg_idx['sr_vip_1'], self.cg_idx['sr_vip_2'], self.cg_idx['sr_sst_1'], self.cg_idx['sr_sst_2']))
            self.mask[np.ix_(self.cg_idx['sr_vip_1'], self.cg_idx['sr_sst_2'])] = 0    # assume VIP and SST from different subpopulations don't connect to each other
            self.mask[np.ix_(self.cg_idx['sr_vip_2'], self.cg_idx['sr_sst_1'])] = 0
            self.mask[np.ix_(self.cg_idx['sr_sst_1'], self.cg_idx['sr_vip_2'])] = 0
            self.mask[np.ix_(self.cg_idx['sr_sst_2'], self.cg_idx['sr_vip_1'])] = 0
                
        # make the SR SST -> SR Edend connectivity more structured
        if self.structured_sr_sst_to_sr_edend==True:
            print('Intializaing a structured SR Edend to SR SST connectivity using mask\n')
            self.mask[np.ix_(self.cg_idx['sr_sst'], self.cg_idx['sr_edend'])] = 0    
            n_dend_per_sst = self.n['sr_edend']//self.n['sr_sst']
            if self.n['sr_edend']%self.n['sr_sst']!=0:
                raise ValueError('# SR Edend is not an integer of # SR SST!')
            for n_srsst in range(self.n['sr_sst']):
                print('n_srsst={}'.format(n_srsst))
                n_sredend_start = self.cg_idx['sr_edend'][0] + n_dend_per_sst*n_srsst
                n_sredend_end = n_sredend_start + n_dend_per_sst
                self.mask[self.cg_idx['sr_sst'][n_srsst], n_sredend_start:n_sredend_end] = -1  
                print('sst idx: {}, start: {}, end: {}'.format(self.cg_idx['sr_sst'][n_srsst], n_sredend_start, n_sredend_end))
                
        # branch specific: in a given E cell, one branch is targeted by SST group 1, the other by SST group 2
        if self.structured_sr_sst_to_sr_edend_branch_specific==True:
            print('Intializaing a structured SR Edend to SR SST connectivity using mask (branch specific)\n')
            self.mask[np.ix_(self.cg_idx['sr_sst'], self.cg_idx['sr_edend'])] = 0
            dend1_idx = np.arange(self.cg_idx['sr_edend'][0], self.cg_idx['sr_edend'][0]+len(self.cg_idx['sr_edend'])//2)
            dend2_idx = np.array([nd for nd in self.cg_idx['sr_edend'] if nd not in dend1_idx])
            self.mask[np.ix_(self.cg_idx['sr_sst_1'], dend1_idx)] = -1
            self.mask[np.ix_(self.cg_idx['sr_sst_2'], dend2_idx)] = -1
            print(self.cg_idx['sr_sst_1'], dend1_idx)
            print(self.cg_idx['sr_sst_2'], dend2_idx)
                
        # TODO structured sr_esoma to sr_sst
        if False:    # cross inhibition from SR Esoma to SR SST
            self.cg_idx['sr_esoma_1'] = np.arange(self.cg_idx['sr_esoma'][0], self.cg_idx['sr_esoma'][0]+len(self.cg_idx['sr_esoma'])//2)
            self.cg_idx['sr_esoma_2'] = np.array([n for n in self.cg_idx['sr_esoma'] if n not in self.cg_idx['sr_esoma_1']])
            self.mask[np.ix_(self.cg_idx['sr_esoma_1'], self.cg_idx['sr_sst_1'])] = 0
            self.mask[np.ix_(self.cg_idx['sr_esoma_2'], self.cg_idx['sr_sst_2'])] = 0
        
        
        
        # matrix for the fixed part of the connectivity
        self.w_fix = nn.Parameter(torch.zeros(self.total_n_neurons, self.total_n_neurons), requires_grad=False)
        
        if self.trainable_dend2soma==False:
            self.dend2soma = nn.Parameter(torch.Tensor([1]), requires_grad=False)    # magnitude of the coupling between dendrite and soma
        elif self.trainable_dend2soma==True:
            print('trainable dendrite to soma coupling\n')
            self.dend2soma = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        if self.divisive_dend_inh==True or self.divisive_dend_ei==True:
            print('using divisive dendrites\n')
            if self.trainable_dend2soma==True:
                raise ValueError('when using divisive dendrite, should set the dend2soma to 0!')
            self.dend2soma = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            print('setting dend2soma to 0\n')
        
        # multiple dendritic branches per soma
        if len(self.dend_idx)!=0:
            for nb in range(self.n_branches):
                dend_idx_sr_b = (self.cg_idx['sr_edend'][0] + nb*len(self.cg_idx['sr_esoma'])) + np.arange(len(self.cg_idx['sr_esoma']))    # the dendrite idx for the nb'th dendritic branch
                self.w_fix[np.ix_(dend_idx_sr_b, self.cg_idx['sr_esoma'])] = self.dend2soma*torch.eye(self.n['sr_esoma'])
                dend_idx_pfc_b = (self.cg_idx['pfc_edend'][0] + nb*len(self.cg_idx['pfc_esoma'])) + np.arange(len(self.cg_idx['pfc_esoma']))    # the dendrite idx for the nb'th dendritic branch
                self.w_fix[np.ix_(dend_idx_pfc_b, self.cg_idx['pfc_esoma'])] = self.dend2soma*torch.eye(self.n['pfc_esoma'])
        
#         self.w_fix = self.w_fix.to(self.w_rec.device)
#         print('self.w_fix device: {}'.format(self.w_fix.device))
#         print('self.w_rec device: {}'.format(self.w_rec.device))
        
        
        
        # Set input weights.
        self.w_in = nn.Parameter(torch.Tensor(self.n['input'], self.total_n_neurons), requires_grad=True)
        self.mask_in = nn.Parameter(torch.zeros(self.n['input'], self.total_n_neurons), requires_grad=False)    # the mask for the input weights
        self.mask_in[:, self.input_targ_idx_sr] = 1    # long range connections are excitatory
        
        
        
        # Set output weights (mask will be set in the descendent class)
        self.w_out = nn.Parameter(torch.Tensor(self.total_n_neurons, self.n['output']), requires_grad=True)
        if self.train_rule==True:
            self.w_out_rule = nn.Parameter(torch.Tensor(self.total_n_neurons, 2), requires_grad=True)
        
        
        # set weights for reward on previous trial
        self.w_rew = nn.Parameter(torch.Tensor(2, self.total_n_neurons), requires_grad=True)
        self.mask_rew = nn.Parameter(torch.zeros(2, self.total_n_neurons), requires_grad=False)
        if self.fdbk_to_vip==True:
            print('feedback only targets PFC VIP cells\n')
            self.mask_rew[:, self.cg_idx['pfc_vip']] = 1    # reward targets VIP
        else:
            self.mask_rew[:, self.input_targ_idx_pfc] = 1    # reward information feeds into PFC only.
        
    
        # set weights for stimulus on previous trial
        self.w_prev_stim = nn.Parameter(torch.Tensor(self.n['input'], self.total_n_neurons), requires_grad=True)
        self.mask_prev_stim = nn.Parameter(torch.zeros(self.n['input'], self.total_n_neurons), requires_grad=False)
        if self.fdbk_to_vip==True:
            print('prev stim only targets PFC VIP cells\n')
            self.mask_prev_stim[:, self.cg_idx['pfc_vip']] = 1 
        else:
            self.mask_prev_stim[:, self.input_targ_idx_pfc] = 1    # reward information feeds into PFC only.
    
        # set weights for choice on previous trial
        self.w_prev_choice = nn.Parameter(torch.Tensor(2, self.total_n_neurons), requires_grad=True)
        self.mask_prev_choice = nn.Parameter(torch.zeros(2, self.total_n_neurons), requires_grad=False)
        if self.fdbk_to_vip==True:
            print('prev choice only targets PFC VIP cells\n')
            self.mask_prev_choice[:, self.cg_idx['pfc_vip']] = 1 
        else:
            self.mask_prev_choice[:, self.input_targ_idx_pfc] = 1    # reward information feeds into PFC only.
        
        # test: make the initial state trainable
        self.h0 = nn.Parameter(torch.Tensor(1, self.total_n_neurons), requires_grad=True)
        
        # Initialize biases
        if self.sr_sst_high_bias==False:
            self.bias = nn.Parameter(torch.zeros(self.total_n_neurons), requires_grad=False)    # fix bias at 0 so that the net has a trivial fixed point at 0
            self.mask_bias = nn.Parameter(torch.ones(self.total_n_neurons), requires_grad=False)
            self.mask_bias[self.dend_idx] = 0
        elif self.sr_sst_high_bias==True:
            print('SR SST has high bias')
            self.bias = nn.Parameter(torch.zeros(self.total_n_neurons), requires_grad=True)
            self.mask_bias = nn.Parameter(torch.zeros(self.total_n_neurons), requires_grad=False)
            self.mask_bias[self.cg_idx['sr_sst']] = 1
            
            
        # initialized weights and biases
        self.reset_parameters()
        
#         print('hi!')
        
        

    def reset_parameters(self):
        """ Initialize parameters """
        
        if self.initialization_weights=='diagonal':
            nn.init.eye_(self.w_rec)  
            nn.init.eye_(self.w_in)
            nn.init.eye_(self.w_out)
            if self.train_rule==True:
                nn.init.eye_(self.w_out_rule)
            nn.init.eye_(self.w_rew)
            nn.init.eye_(self.w_prev_stim)
            nn.init.eye_(self.w_prev_choice)
            nn.init.eye_(self.h0)
        elif self.initialization_weights=='orthogonal':
            nn.init.orthogonal_(self.w_rec)  
            nn.init.orthogonal_(self.w_in)
            nn.init.orthogonal_(self.w_out)
            if self.train_rule==True:
                nn.init.orthogonal_(self.w_out_rule)
            nn.init.orthogonal_(self.w_rew)
            nn.init.orthogonal_(self.w_prev_stim)
            nn.init.orthogonal_(self.w_prev_choice)
            nn.init.orthogonal_(self.h0)
        elif self.initialization_weights=='kaiming_normal':
            nn.init.kaiming_normal_(self.w_rec, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.w_in, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.w_out, mode='fan_in', nonlinearity='relu')
            if self.train_rule==True:
                nn.init.kaiming_normal_(self.w_out_rule, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.w_rew, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.w_prev_stim, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.w_prev_choice, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.h0, mode='fan_in', nonlinearity='relu')
        elif self.initialization_weights=='kaiming_uniform':
            nn.init.kaiming_uniform_(self.w_rec, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.w_in, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.w_out, mode='fan_in', nonlinearity='relu')
            if self.train_rule==True:
                nn.init.kaiming_uniform_(self.w_out_rule, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.w_rew, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.w_prev_stim, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.w_prev_choice, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.h0, mode='fan_in', nonlinearity='relu')
        elif self.initialization_weights=='uniform':
            a = -1/self.total_n_neurons
            b = 1/self.total_n_neurons
            nn.init.uniform_(self.w_rec, a=a, b=b)
            nn.init.uniform_(self.w_in, a=a, b=b)
            nn.init.uniform_(self.w_out, a=a, b=b)
            if self.train_rule==True:
                nn.init.uniform_(self.w_out_rule, a=a, b=b)
            nn.init.uniform_(self.w_rew, a=a, b=b)
            nn.init.uniform_(self.w_prev_stim, a=a, b=b)
            nn.init.uniform_(self.w_prev_choice, a=a, b=b)
            nn.init.uniform_(self.h0, a=a, b=b)
        elif self.initialization_weights=='normal':
            mean=0
            std=np.sqrt(1/self.total_n_neurons)    # see Rajan and Abbott, 2016
            nn.init.normal_(self.w_rec, mean=mean, std=std)
            nn.init.normal_(self.w_in, mean=mean, std=std)
            nn.init.normal_(self.w_out, mean=mean, std=std)
            if self.train_rule==True:
                nn.init.normal_(self.w_out_rule, mean=mean, std=std)
            nn.init.normal_(self.w_rew, mean=mean, std=std)
            nn.init.normal_(self.w_prev_stim, mean=mean, std=std)
            nn.init.normal_(self.w_prev_choice, mean=mean, std=std)
            nn.init.normal_(self.h0, mean=mean, std=std)
        elif self.initialization_weights=='constant':
            val = 1/self.total_n_neurons
            nn.init.constant_(self.w_rec, val)
            nn.init.constant_(self.w_in, val)
            nn.init.constant_(self.w_out, val)
            if self.train_rule==True:
                nn.init.constant_(self.w_out_rule, val)
            nn.init.constant_(self.w_rew, val) 
            nn.init.constant_(self.w_prev_stim, val)
            nn.init.constant_(self.w_prev_choice, val)
            nn.init.constant_(self.h0, val)
        else:
            raise NotImplementedError
        
        # scale E connections to maintain E-I balance
        if len(self.i_idx)!=0:
            self.w_rec.data[self.e_idx,:] /= (len(self.e_idx)/len(self.i_idx))
        if self.scale_down_init_wexc == True:
            self.w_rec.data[self.e_idx,:] /= 10
        
        # NEW: scale connections that EACH cell group receives similar E and I inputs - doesn't seem to work
#         for cg_receive in self.cell_group_list:
#             nE = 0    # number of afferent E connections
#             nI = 0    # number of afferent I connections
#             for cg_send in self.cell_group_list:    # count the number of E and I inputs
#                 if self.is_connected(cg_receive, cg_send) is False:
#                     continue 
#                 if 'soma' in cg_send:
#                     nE += len(self.cg_idx[cg_send])
#                 else:
#                     nI += len(self.cg_idx[cg_send])
#             if nI==0:
#                 print('{}, nI=0'.format(cg_receive))
#             else:
#                 print('{}, nE/nI={}'.format(cg_receive, nE/nI))
                    
#             for cg_send in self.cell_group_list:    # scale the connections
#                 if self.is_connected(cg_receive, cg_send) is False:
#                     continue
#                 if 'soma' in cg_send:
#                     self.w_rec.data[np.ix_(self.cg_idx[cg_send], self.cg_idx[cg_receive])] /= nE/nI
                
                
        
        # initialize coupling from dend to soma
#         torch.nn.init.constant_(self.dend2soma, 1)
        
        if self.initialzation_bias=='zero':
            nn.init.zeros_(self.bias)
            if self.sr_sst_high_bias==True:
#                 self.bias[self.cg_idx['sr_sst']] = 2    # this way the bias would not change!
#                 self.bias.requires_grad = True
                torch.nn.init.constant_(self.bias, 1)
                print(self.bias[self.cg_idx['sr_sst']])
                print('\n')
        else:
            raise NotImplementedError

        

    @staticmethod
    def is_connected(cg1, cg2):
        """ If the cell group cg2 connects to cg1 """
        
        # VIP also targets PV?
        
        if ((cg1=='sr_esoma' and cg2=='sr_esoma') or 
            (cg1=='sr_esoma' and cg2=='sr_pv') or
            (cg1=='sr_esoma' and cg2=='sr_edend') or
            (cg1=='sr_edend' and cg2=='sr_sst') or
            (cg1=='sr_edend' and cg2=='pfc_esoma') or     # top-down projection
            (cg1=='sr_pv' and cg2=='sr_pv') or 
            (cg1=='sr_pv' and cg2=='sr_sst') or 
            (cg1=='sr_pv' and cg2=='sr_esoma') or 
            (cg1=='sr_pv' and cg2=='pfc_esoma') or     # optional
            (cg1=='sr_sst' and cg2=='sr_vip') or 
            (cg1=='sr_sst' and cg2=='sr_esoma') or     # delete it just for testing
            (cg1=='sr_sst' and cg2=='pfc_esoma') or     # this connection is weak
            (cg1=='sr_vip' and cg2=='sr_sst') or    # delete for testing
            (cg1=='sr_vip' and cg2=='pfc_esoma') or     
            (cg1=='pfc_esoma' and cg2=='pfc_esoma') or 
            (cg1=='pfc_esoma' and cg2=='pfc_pv') or
            (cg1=='pfc_esoma' and cg2=='pfc_edend') or
            (cg1=='pfc_edend' and cg2=='pfc_sst') or
            (cg1=='pfc_edend' and cg2=='sr_esoma') or 
            (cg1=='pfc_pv' and cg2=='pfc_pv') or 
            (cg1=='pfc_pv' and cg2=='pfc_sst') or 
            (cg1=='pfc_pv' and cg2=='pfc_esoma') or 
            (cg1=='pfc_pv' and cg2=='sr_esoma') or     # optional
            (cg1=='pfc_sst' and cg2=='pfc_vip') or 
            (cg1=='pfc_sst' and cg2=='pfc_esoma') or 
#             (cg1=='pfc_sst' and cg2=='sr_esoma') or    # optional
            (cg1=='pfc_vip' and cg2=='pfc_sst') 
#             (cg1=='pfc_vip' and cg2=='sr_esoma')    # optional 
           ):
            return True
        else:
            return False
        
        
        
    def display_connection_prob(self):
        """ print the connectivity between cell groups """
        for cg1 in self.cell_group_list:
            for cg2 in self.cell_group_list:
                print('{} to {}: {}'.format(cg1, cg2, torch.mean(self.mask.data(np.ix_(self.cg_idx[cg1], self.cg_idx[cg2])))))
     
    
    def nonlinearity_soma(self, input_current, nonlinearity, k_relu_satu=10):
        """ somatic nonlinearity """

        relu = nn.ReLU()
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()

        if nonlinearity=='relu':
            f = relu(input_current)
        elif nonlinearity=='tanh':
            f = tanh(input_current)
        elif nonlinearity=='sigmoid':
            f = sigmoid(input_current)
        elif nonlinearity=='relu_satu':
            f = relu(k_relu_satu*tanh(input_current/k_relu_satu))    # this function approximates relu for [0,k_relu_satu] but is bounded by k_relu_satu
        else:
            raise NotImplementedError
            
        return f
        
        
    def nonlinearity_dend(self, g_e, g_i, b_g=5.56, g_LD=4, k=9.64, gamma=6.54, nonlinearity='old'):
        """ Dendritic nonlinearity adapted from Robert Yang's 2016 Nat Comm paper 
            Args:
                g_e: excitatory input
                g_i: inhibitory input
        """
        
        relu = nn.ReLU()
        
        if nonlinearity=='old':    # in this case the width of the tanh function does not change much with g_i
    #         g_half = b_g*(g_LD + g_i)
            g_half = g_i
            beta = torch.exp(g_i/100)
            dend_current = torch.tanh((g_e - g_half)/beta) 
#             print('dend_current={}'.format(torch.mean(dend_current)))
        elif nonlinearity=='v2':   # the width of the tanh function changes with g_i
            epsilon = 1e-3     # so that it is well-defined when g_e=g_i=0
            g_i_prime = g_i+epsilon 
            dend_current = torch.tanh((g_e-g_i_prime)/g_i_prime) - torch.tanh(torch.tensor(-1))    # to make it 0 when g_e is 0
        elif nonlinearity=='v3':   # a more dramatic gating function
            epsilon = 1e-3     # so that it is well-defined when g_e=g_i=0
            g_i_prime = g_i+epsilon
            dend_current = torch.tanh((relu(g_e-g_i_prime))/(g_i_prime))
        elif nonlinearity=='step':
            epsilon = 1e-3     # so that it is well-defined when g_i=0
            g_half = g_i 
            beta = g_i/100 + epsilon
            dend_current = torch.tanh((g_e - g_half)/beta) 

    
        return dend_current
    
    
    
        
        
    def effective_weight(self, w, mask, w_fix=0):
        """ compute the effective weight """
        
#         print(w, mask, w_fix)
#         print(w.device, mask.device, w_fix.device)
        w_eff = torch.abs(w) * mask + w_fix  
        
        
#         relu = nn.ReLU()    # or, try relu
#         w_eff = relu(w) * mask + w_fix
        
        return w_eff
    
    
    def forward(self, stim, I_prev_rew, I_prev_stim, I_prev_choice, hp, hp_task, record_h=False, i_me_init=None, h_init=None):
        """ Evolve the two-area RNN in time for the duration of 1 trial 
            Args:
            - stim: stimulus input to the SR network. batch_size*n_in*n_timesteps
            - h_init: the initial state of the network. If None, theinput_periodn initialize randomly.
            - record_h: whether output the hidden states over time
            Output:
            - output: a torch tensor of shape batch_size*n_out*n_timesteps
            - output_rule: a torch tensor of shape batch_size*n_out_pfc*n_timesteps
            - hidden_record: a dict where each elements is a tensor of shape batch_size*n_neurons*n_timesteps
        """
        
        start = time.time()
        n_tsteps = stim.shape[-1]
#         input_period = np.arange(int(hp_task['stim_start']/hp['dt']), int(hp_task['stim_end']/hp['dt']))    # input period for the trial history info
#         print('input period = {}'.format(input_period))
        hidden_record = []
        total_input_record = []
        input_dend_e_record = []
        input_dend_i_record = []
        w_rec_eff_record = []
        i_me_record = []
        sigmoid = nn.Sigmoid()
        
        # initialize the hidden state        
#         hidden = torch.normal(mean=torch.zeros([self.batch_size, self.total_n_neurons]), 
#                               std=torch.ones([self.batch_size, self.total_n_neurons]))
        if h_init is None:
            hidden = torch.zeros([self.batch_size, self.total_n_neurons]).to(self.w_rec.device)    # network state is 0 at t=0
#             hidden = self.h0.repeat((self.batch_size, 1))
        else:
            hidden = h_init
            
        next_h = torch.zeros(hidden.shape).to(self.w_rec.device)
        gain = torch.ones(hidden.shape).to(self.w_rec.device)
        
        if len(self.i_idx)!=0:
            self.w_rec_eff = self.effective_weight(w=self.w_rec, mask=self.mask, w_fix=torch.abs(self.dend2soma)*self.w_fix)
        elif len(self.i_idx)==0:    # only E cell, then do not apply Dale's law
            self.w_rec_eff = self.w_rec
        self.w_in_eff = self.effective_weight(w=self.w_in, mask=self.mask_in)
        self.w_rew_eff = self.effective_weight(w=self.w_rew, mask=self.mask_rew)
        self.w_prev_stim_eff = self.effective_weight(w=self.w_prev_stim, mask=self.mask_prev_stim)
        self.w_prev_choice_eff = self.effective_weight(w=self.w_prev_choice, mask=self.mask_prev_choice)

#         if torch.isnan(prev_stim)==False:
#             prev_stim2 = torch.mean(prev_stim[:,:,int(hp_task['stim_start']/hp['dt']):int(hp_task['stim_end']/hp['dt'])], axis=-1)
#         if torch.isnan(last_rew)==False:
#             last_rew2 = torch.outer(last_rew.float(), torch.Tensor([1,0])) + torch.outer((~last_rew).float(), torch.Tensor([0,1]))     # from Boolean to current
        if hp['timeit_print']==True:
            print('before looping over time takes {:.2e}s'.format(time.time()-start))
        
        
        for t in range(n_tsteps): 
            start_loop = time.time()
#             print('n_tsteps={}'.format(n_tsteps))
            if hp['timeit_print']==True:
                print('t = {}'.format(t))
            start_thisstep = time.time()
            stim_at_t = torch.squeeze(stim[:,:,t])
            i_rew = torch.squeeze(I_prev_rew[:,:,t])
#             print('i_rew shape: {}'.format(i_rew.shape))
            i_prev_stim = torch.squeeze(I_prev_stim[:,:,t])
            i_prev_choice = torch.squeeze(I_prev_choice[:,:,t])
#             print('stim_at_t shape: {}'.format(stim_at_t.shape))
            if hp['timeit_print']==True:
                print('squeeze takes {:.2e}s'.format(time.time()-start_thisstep))
        



            # update hidden state
            start_thisstep = time.time()
            ext_input = stim_at_t@self.w_in_eff \
                            + self.prev_rew_mag*i_rew@self.w_rew_eff \
                            + self.prev_stim_mag*i_prev_stim@self.w_prev_stim_eff \
                            + self.prev_choice_mag*i_prev_choice@self.w_prev_choice_eff     # external input
#             ext_input = stim_at_t@self.w_in_eff
#             print('stim_at_t={}'.format(torch.mean(stim_at_t)))
#             print('i_rew={}'.format(torch.mean(i_rew)))
#             print('i_prev_stim={}'.format(torch.mean(i_prev_stim)))
#             print('i_prev_choice={}'.format(torch.mean(i_prev_choice)))
#             print('ext_input={}'.format(torch.mean(ext_input)))
            if len(ext_input.shape)==1:    # when batch_size=1, add a new dimension in the batch dimension because torch.squeeze has deleted it
                ext_input = ext_input[None,:]
            if hp['timeit_print']==True:
                print('calculate ext input takes {:.2e}s'.format(time.time()-start_thisstep))  


            # compute input
            start_thisstep = time.time()
#             noise_input = hp['network_noise']*(torch.randn_like(ext_input))
#             print(noise_input[:5,:5], torch.mean(noise_input), torch.std(noise_input))
            total_input = hidden@self.w_rec_eff + ext_input    # total in put current to a cell
            
            i_exc = hidden[:,self.e_idx]@self.w_rec_eff[self.e_idx, :] + ext_input
#             print('i_exc={}'.format(torch.mean(i_exc)))
            i_inh = hidden[:,self.i_idx]@self.w_rec_eff[self.i_idx, :]
#             print('i_inh={}'.format(torch.mean(i_inh)))
            if hp['timeit_print']==True:
                print('calculate E/I/total input takes {:.2e}s'.format(time.time()-start_thisstep))
    
            # compute mGluR current
            start_thisstep = time.time()
            if self.mglur==True:
                if t==0:
                    if i_me_init==None:
                        i_me = torch.zeros([self.batch_size, len(self.mglur_e_idx)]).to(self.w_rec.device)
                    else:
                        i_me = i_me_init.to(self.w_rec.device)    # do not reset
                else:
                    i_me = (1-self.alpha_me)*i_me + self.alpha_me*i_exc[:, self.mglur_e_idx]
                    total_input[:,self.mglur_e_idx] += i_me
            elif self.mglur==False:
                i_me = torch.Tensor(0, device=self.w_rec.device)
            i_me_record.append(i_me)
            if hp['timeit_print']==True:
                print('calculate mGluR takes {:.2e}s'.format(time.time()-start_thisstep))  

#                     print('t={}, mean i_me={}'.format(t, torch.mean(i_me)))
                
            
            # dendritic current
            start_thisstep = time.time()
            if len(self.dend_idx)!=0:
                input_dend_e = i_exc[:, self.dend_idx]
#                 print('input_dend_e={}'.format(torch.mean(input_dend_e)))
                input_dend_i = i_inh[:, self.dend_idx]
                input_dend_e_sr = i_exc[:, self.dend_idx_sr]    # e input onto SR dendrites. assuming long range input is excitatory
                input_dend_i_sr = i_inh[:, self.dend_idx_sr]    # i input onto SR dendrites (no long-range inputs)
                input_dend_e_pfc = i_exc[:, self.dend_idx_pfc]    # e input onto PFC dendrites. assuming long range input is excitatory
                input_dend_i_pfc = i_inh[:, self.dend_idx_pfc]    # i input onto PFC dendrites (no long-range inputs)
            else:
                input_dend_e = torch.Tensor(0)
                input_dend_i = torch.Tensor(0)
            if hp['timeit_print']==True:
                print('compute inputs to dendrite takes {:.2e}s'.format(time.time()-start_thisstep))
            
            # update soma
            start_thisstep = time.time()
            if self.divisive_dend_inh==False and self.divisive_dend_ei==False:
#                 next_h = (1-self.decay)*hidden + self.decay*self.nonlinearity_soma(total_input + self.mask_bias*self.bias)     # old
                next_h = (1-self.decay)*hidden + self.decay*self.nonlinearity_soma(input_current=total_input+self.mask_bias*self.bias, nonlinearity=self.activation, k_relu_satu=self.k_relu_satu)
            elif self.divisive_dend_inh==True or self.divisive_dend_ei==True:
                input_dend_e_sum_sr = input_dend_e_sr.reshape((self.batch_size, self.n_branches, -1))
                input_dend_e_sum_sr = torch.sum(input_dend_e_sum_sr, axis=1)    # sum of exc current that goes into all the branches
                input_dend_i_sum_sr = input_dend_i_sr.reshape((self.batch_size, self.n_branches, -1))
                input_dend_i_sum_sr = torch.sum(input_dend_i_sum_sr, axis=1)    # sum of exc current that goes into all the branches
                input_dend_e_sum_pfc = input_dend_e_pfc.reshape((self.batch_size, self.n_branches, -1))
                input_dend_e_sum_pfc = torch.sum(input_dend_e_sum_pfc, axis=1)    # sum of exc current that goes into all the branches
                input_dend_i_sum_pfc = input_dend_i_pfc.reshape((self.batch_size, self.n_branches, -1))
                input_dend_i_sum_pfc = torch.sum(input_dend_i_sum_pfc, axis=1)    # sum of exc current that goes into all the branches
                input_dend_e_sum = torch.cat((input_dend_e_sum_sr, input_dend_e_sum_pfc), axis=1)
                input_dend_i_sum = torch.cat((input_dend_i_sum_sr, input_dend_i_sum_pfc), axis=1)
#                 input_dend_e_sum = input_dend_e_sum_sr
#                 input_dend_i_sum = input_dend_i_sum_sr
#                 input_dend_e_sum = input_dend_e.reshape((self.batch_size, self.n_branches, -1))
#                 input_dend_e_sum = torch.sum(input_dend_e_sum, axis=1)    # sum of exc current that goes into all the branches
#                 print('input_dend_e_sum shape: {}'.format(input_dend_e_sum.shape))
#                 input_dend_i_sum = input_dend_i.reshape((self.batch_size, self.n_branches, -1))
#                 input_dend_i_sum = torch.sum(input_dend_i_sum, axis=1)    # sum of inh current that goes into all the branches
#                 print('input_dend_e_sum shape: {}'.format(input_dend_e_sum.shape))
#                 print('input_dend_i_sum shape: {}'.format(input_dend_i_sum.shape))
#                 print('next_h shape: {}'.format(next_h[:, self.soma_idx].shape))
#                 print('hidden shape: {}'.format(hidden[:, self.soma_idx].shape))
#                 print('total_input shape: {}'.format(total_input[:, self.soma_idx].shape))
#                 print('ext_input shape: {}'.format(ext_input[:, self.soma_idx].shape))
                if self.divisive_dend_inh==True:
                    total_input[:,np.concatenate((self.cg_idx['sr_esoma'], self.cg_idx['pfc_esoma']))] += input_dend_e_sum
                    gain[:, np.concatenate((self.cg_idx['sr_esoma'], self.cg_idx['pfc_esoma']))] = sigmoid(input_dend_i_sum)
#                     next_h[:, self.soma_idx] = (1-self.decay)*hidden[:, self.soma_idx] + self.decay*gain[:, self.soma_idx]*self.nonlinearity_soma(total_input[:, self.soma_idx] + self.mask_bias[self.soma_idx]*self.bias[self.soma_idx] + hp['network_noise']*torch.randn_like(ext_input[:, self.soma_idx]))
                elif self.divisive_dend_ei==True:
                    gain[:, np.concatenate((self.cg_idx['sr_esoma'], self.cg_idx['pfc_esoma']))] = sigmoid(input_dend_i_sum + input_dend_e_sum)
#                     next_h[:, self.soma_idx] = (1-self.decay)*hidden[:, self.soma_idx] + self.decay*gain[:, self.soma_idx]*self.nonlinearity_soma(total_input[:, self.soma_idx] + self.mask_bias[self.soma_idx]*self.bias[self.soma_idx] + hp['network_noise']*torch.randn_like(ext_input[:, self.soma_idx])) 
#                 print('gain={}'.format(torch.mean(gain)))
#                 print('total_input={}'.format(torch.mean(total_input)))
#                 next_h = (1-self.decay)*hidden + self.decay*gain*self.nonlinearity_soma(total_input + self.mask_bias*self.bias)    # old
                next_h = (1-self.decay)*hidden + self.decay*gain*self.nonlinearity_soma(input_current=total_input+self.mask_bias*self.bias, nonlinearity=self.activation, k_relu_satu=self.k_relu_satu) 
            if hp['timeit_print']==True:
                print('update soma takes {:.2e}s'.format(time.time()-start_thisstep))   

            # update dendrite
            start_thisstep = time.time()
            if len(self.dend_idx)!=0:
                if self.divisive_dend_inh==True:
                    next_h[:,self.dend_idx] = input_dend_i
                elif self.divisive_dend_ei==True:
                    next_h[:,self.dend_idx] = input_dend_e + input_dend_i
                else:
#                     print('updating dendrite, input_e={}. input_i={}'.format(torch.mean(input_dend_e), torch.mean(input_dend_i)))
    #             start = time.time()
                    next_h[:,self.dend_idx] = self.nonlinearity_dend(nonlinearity=self.dend_nonlinearity, g_e=input_dend_e, g_i=torch.abs(input_dend_i))
            if hp['timeit_print']==True:
                print('update dendrite takes {:.2e}s'.format(time.time()-start_thisstep))   
    
            next_h += hp['network_noise']*torch.randn_like(next_h)    # add some noise
        
        
            # collect stuff
#             start = time.time()
            hidden_record.append(hidden)
#             print('hidden appended, hidden = {}'.format(hidden))
            total_input_record.append(total_input)
            input_dend_e_record.append(input_dend_e)
            input_dend_i_record.append(input_dend_i)

            hidden = next_h
#             print('collect takes {}s'.format(time.time()-start))   
            if hp['timeit_print']==True:
                print('1 timestep takes {:.2e}s'.format(time.time()-start_loop))
        
        start_thisstep = time.time()
        h_last = hidden     # the state at the last timepoint
        i_me_last = i_me
        w_rec_eff_record = self.w_rec_eff.detach()
        
#         print('hidden_record={}'.format(hidden_record))
        hidden_record = torch.stack(hidden_record, dim=0)    # n_timesteps*batch_size*n_neurons        
        total_input_record = torch.stack(total_input_record, dim=0)
        input_dend_e_record = torch.stack(input_dend_e_record, dim=0)
        input_dend_i_record = torch.stack(input_dend_i_record, dim=0)
        if hp['timeit_print']==True:
            print('everything after the loop takes {:.2e}s'.format(time.time()-start_thisstep))
        
        return hidden_record, total_input_record, input_dend_e_record, input_dend_i_record, w_rec_eff_record, h_last, i_me_record, i_me_last
        
          
            
    
class Net_readoutSR(torch.nn.Module):
    """ The network model. 
        Read out movement from SR.
        Read out rule from PFC.
    """
    
    def __init__(self, hp, **kwargs):
        super().__init__()
        self.rnn = BioRNN(hp)
#         self.readout = nn.Linear(self.rnn.n['sr_esoma'], self.rnn.n['output'], bias=False)   
#         # Set output weights.
#         self.w_out = nn.Parameter(torch.Tensor(self.rnn.total_n_neurons, self.rnn.n['output']), requires_grad=True)
        self.mask_out = nn.Parameter(torch.zeros(self.rnn.total_n_neurons, self.rnn.n['output']), requires_grad=False)    # the mask for the input weights
        self.mask_out[self.rnn.cg_idx['sr_esoma'], :] = 1    # long range connections are excitatory
        if hp['train_rule']==True:
#             self.readout_rule = nn.Linear(self.rnn.n['pfc_esoma'], self.rnn.n['output_rule'], bias=False)    # should not have bias. Otherwise just have to set the weight to 0 and have differnece biases for different output nodes
#             self.readout_rule.weight.requires_grad = False    # fix the readout weights for rule. otherwise it is just going to set one row to 0
            self.mask_out_rule = nn.Parameter(torch.zeros(self.rnn.w_out_rule.shape), requires_grad=False)
            self.mask_out_rule[self.rnn.cg_idx['pfc_esoma'], :] = 1    # long range connections are excitatory
           
    def forward(self, input, hp, hp_task, I_prev_rew, I_prev_stim, I_prev_choice, yhat=None, yhat_rule=None, h_init=None, i_me_init=None):
        
        # TODO: clean up this function (get rid of get_perf)
        
        rnn_activity, total_input_record, input_dend_e_record, input_dend_i_record, w_rec_eff_record, h_last, i_me_record, i_me_last \
        = self.rnn(input, I_prev_rew=I_prev_rew, I_prev_stim=I_prev_stim, I_prev_choice=I_prev_choice, h_init=h_init, i_me_init=i_me_init, 
                   hp=hp, hp_task=hp_task)

        # compute output
        self.w_out_eff = self.rnn.w_out*self.mask_out
#         self.w_out_eff = self.rnn.effective_weight(w=self.w_out, mask=self.mask_out)    # if want all E outpt weights
        out = rnn_activity@self.w_out_eff
#         out = self.readout(rnn_activity[:,:,self.rnn.cg_idx['sr_esoma']])
#         out = 1/(1+torch.exp(-out))    # bound between 0 and 1
        out = torch.movedim(out, 0, -1) 
        if hp['train_rule']==True:
            self.w_out_rule_eff = self.rnn.w_out_rule*self.mask_out_rule
            out_rule = rnn_activity@self.w_out_rule_eff
#             out_rule = self.readout_rule(rnn_activity[:,:,self.rnn.cg_idx['pfc_esoma']])
#             out_rule = 1/(1+torch.exp(-out_rule))    # test: bound between 0 and 1
#             out_rule = out_rule/5    # test: scale down so that PFC activity is stronger
            out_rule = torch.movedim(out_rule, 0, -1)
        else:
            out_rule = 0
            
        # softmax over output units
#         softmax = nn.Softmax(dim=1)
#         out_sfmx = torch.Tensor(out.shape).to(out.device)
#         out_rule_sfmx = torch.Tensor(out_rule.shape).to(out.device)
#         out_sfmx[:, 0:2, :] = softmax(out[:, 0:2, :])
#         out_sfmx[:, -1, :] = out[:, -1, :]    # does not change the fixation output
#         out_rule_sfmx = softmax(out_rule)
        
        
        # wrap the performance function (test)
#         perf, choice_prob, choice = self.get_perf(y=out, yhat=yhat, hp=hp, hp_task=hp_task)
#         if hp['train_rule']==True:
#             perf_rule, _, _ = self.get_perf(y=out_rule, yhat=yhat_rule, hp=hp, hp_task=hp_task)
#         else:
#             perf_rule = None
        perf = None
        choice_prob = None
        choice = None
        
   
        
        # collect stuff
        rnn_activity = torch.movedim(rnn_activity, 0, -1)    # batch*neuron*time
        total_input_record = torch.movedim(total_input_record, 0, -1)
        input_dend_e_record = torch.movedim(input_dend_e_record, 0, -1)
        input_dend_i_record = torch.movedim(input_dend_i_record, 0, -1)
        
        data = {'activity': rnn_activity, 'total_inp': total_input_record, 'inp_dend_e': input_dend_e_record, 
                'inp_dend_i': input_dend_i_record, 'w_rec_eff': w_rec_eff_record, 'h_last': h_last, 'i_me': i_me_record, 
                'i_me_last': i_me_last}
        
        return  out, out_rule, perf, choice_prob, choice, data
   
    
    
    
#     def get_perf(self, y, yhat, hp_task, hp):
#         """ From the output and target, get the performance of the network 
#             Args:
#                 y: batch_size*n_output*n_timesteps
#                 yhat: batch_size*n_output*n_timesteps
#             Returns:
#                 resp_correct: length batch_size binary vector
#         """
# #     if y.size()[1]!=3 or yhat.size()[1]!=3:
# #         raise ValueError('This function only works when there are 2 choices!')
#         resp_start_ts = int(hp_task['resp_start']/hp['dt'])
#         resp_end_ts = int(hp_task['resp_end']/hp['dt'])

#         softmax = nn.Softmax(dim=1)
    
#         y_choice = torch.mean(y[:, 0:2, resp_start_ts:resp_end_ts], dim=-1)    # batch_size * 2
#         choice_prob = y_choice    # softmax would soften the difference a lot and worsen the performance...
#     #     choice_prob = softmax(choice)
#         choice = torch.zeros([choice_prob.shape[0], 2])    # compute choices from choice probabilities
#         choice[:,0] = torch.gt(choice_prob[:,0], choice_prob[:,1])
#         choice[:,1] = torch.gt(choice_prob[:,1], choice_prob[:,0])

#         target = torch.mean(yhat[:, 0:2, resp_start_ts:resp_end_ts], dim=-1)
#         target_prob = target

#         match = torch.abs(choice - target_prob) <= 0.5    
#         resp_correct = match[:,0] * match[:,1]    # correct response if the probability from target is differed by less than threshold% for both choices

#         return resp_correct, choice_prob, choice
        
        
    
        
class Net_readoutPFC(torch.nn.Module):
    """ The network model.
        Read out movement from PFC.
        Does not read out rule.
    """
    
    def __init__(self, hp, **kwargs):
        super().__init__()
        self.rnn = TwoAreaNet_OneMatrix(hp)
        self.readout = nn.Linear(self.rnn.n['pfc_esoma'], self.rnn.n['output'])
        
    def forward(self, input):
        rnn_activity = self.rnn(input)
        out = self.readout(rnn_activity[:,:,self.rnn.cg_idx['pfc_esoma']])
#         out = 1/(1+torch.exp(-out))    # bound between 0 and 1
        out = torch.movedim(out, 0, -1)
        rnn_activity = torch.movedim(rnn_activity, 0, -1)    # batch*neuron*time
        return out, rnn_activity
    
    



        
# simplified models for speed testing    
class SimpleNet(nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
        rnn: str, type of RNN, lstm, rnn, ctrnn, or eirnn
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        rnn_activity, h_last = self.rnn(input)
        out = self.fc(rnn_activity)
        return out, {'rnn_activity': rnn_activity, 'h_last': h_last}
    
    
class ManualRNN(nn.Module):
    """ manually implement the Elman RNN """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()
        self.w_rec = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
        self.w_in = nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=True)
        self.nonlinearity = nn.Tanh()
        self.hidden_size = hidden_size
        
        nn.init.orthogonal_(self.w_rec)  
        nn.init.orthogonal_(self.w_in)
        
    def forward(self, input):
        batch_size = input.shape[1]
        hidden = torch.zeros(batch_size, self.hidden_size)
        hiddens = []
        for t in range(input.shape[0]):
#             print(batch_size, self.hidden_size, hidden.shape)
#             total_input = input[t,:,:]@self.w_in + hidden@self.w_rec
            hidden_next = self.nonlinearity(input[t,:,:]@self.w_in + hidden@self.w_rec)
#             hidden_next = self.nonlinearity(input[t,:,:]@self.w_in)
            hiddens.append(hidden_next)
            hidden = hidden_next
            
        return torch.stack(hiddens), hidden

    
class ManualSimpleNet(nn.Module):
    """Recurrent network model.

    Args:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.rnn = ManualRNN(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        rnn_activity, h_last = self.rnn(input)
        out = self.fc(rnn_activity)
        return out, {'rnn_activity': rnn_activity, 'h_last': h_last}

