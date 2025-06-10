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
import matplotlib.pyplot as plt

# This is added by VScode


class BioRNN_working(torch.nn.Module):
    """ 
        RNN with elements from neurobiology
    """

    def __init__(self, hp, **kwargs):
        super().__init__()
        
        self.cell_group_list = hp['cell_group_list']
        
        # Hyperparameters
        self.n = {}
        self.n['input'], self.n['output'], self.n['output_rule'] = hp['n_input'], hp['n_output'], hp['n_output_rule']
        self.n_branches = hp['n_branches']
        for cg in self.cell_group_list:
            self.n[cg] = hp['n_'+cg]
        self.decay = hp['dt']/hp['tau']
        self.dt = hp['dt']
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
#         self.divisive_dend_inh = hp['divisive_dend_inh']
#         self.divisive_dend_ei = hp['divisive_dend_ei']
#         self.divisive_dend_nonlinear = hp['divisive_dend_nonlinear']
        self.dendrite_type = hp['dendrite_type']
        if self.dendrite_type not in ['divisive_inh', 'divisive_ei', 'divisive_nonlinear', 'additive', 'none']:
            raise ValueError('Dendrite type not recognized!')
#         self.scale_down_init_wexc = hp['scale_down_init_wexc']
#         self.prev_rew_mag = 1
#         self.prev_stim_mag = 1
#         self.prev_choice_mag = 1
        self.activation = hp['activation']
        self.k_relu_satu = hp['k_relu_satu']
        self.sparse_pfcesoma_to_srvip = hp['sparse_pfcesoma_to_srvip']
        self.sparse_srsst_to_sredend = hp['sparse_srsst_to_sredend']
        self.time_it = hp['timeit_print']
        self.scale_down_wexc_init = hp['scale_down_wexc_init']
        
        
        if 'exc_to_vip' in hp.keys():    # some models might not have exc_to_vip defined
            self.exc_to_vip = hp['exc_to_vip']
        else:
            self.exc_to_vip = False
            
            
        # print the cell groups
        print('model has the following cell groups: {}\n'.format(self.cell_group_list))
            
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
            if 'esoma' in cg or 'edend' in cg:
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
        self.esoma_idx = [i for i in self.e_idx if (i in self.cg_idx['sr_esoma'] or i in self.cg_idx['pfc_esoma'])]    # used for divisive dendrites
                
        # input-targeting cell indices
        self.input_targ_idx_sr = []    
        for cg in self.cell_group_list:
            if cg=='sr_edend' or cg=='sr_pv':    # input target only SR PV and SR Edend
                self.input_targ_idx_sr.extend(self.cg_idx[cg])
        if len(self.i_idx)==0 or len(self.dend_idx)==0:
            self.input_targ_idx_sr = self.cg_idx['sr_esoma']
            print('no E dendrites, input target sr_esoma\n', flush=True)
        
        self.input_targ_idx_pfc = []     # where long-range input targets in PFC
        for cg in self.cell_group_list:
            if cg=='pfc_edend' or cg=='pfc_pv':    # input target only PFC PV and PFC Edend
                self.input_targ_idx_pfc.extend(self.cg_idx[cg])
        if len(self.i_idx)==0 or len(self.dend_idx)==0:
            self.input_targ_idx_pfc = self.cg_idx['pfc_esoma']
            print('no E dendrites, input target pfc_esoma\n', flush=True)
           

        # Initialize connectivity matrices
        self.w_rec = nn.Parameter(torch.Tensor(self.total_n_neurons, self.total_n_neurons), requires_grad=True)
        
        #=== define mask ===#
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
            
        # leave out connections between populations
        if self.no_pfcesoma_to_srsst==True and 'sr_sst' in self.cell_group_list:
            self.mask[np.ix_(self.cg_idx['pfc_esoma'], self.cg_idx['sr_sst'])] = 0
            print('no PFC esoma to SR SST\n')
        if self.no_pfcesoma_to_sredend==True and 'sr_edend' in self.cell_group_list:
            self.mask[np.ix_(self.cg_idx['pfc_esoma'], self.cg_idx['sr_edend'])] = 0
            print('no PFC esoma to SR edend\n')
        if self.no_pfcesoma_to_srpv==True and 'sr_pv' in self.cell_group_list:
            self.mask[np.ix_(self.cg_idx['pfc_esoma'], self.cg_idx['sr_pv'])] = 0
            print('no PFC esoma to SR PV\n')
        if self.no_srsst_to_srvip==True and 'sr_sst' in self.cell_group_list and 'sr_vip' in self.cell_group_list:
            self.mask[np.ix_(self.cg_idx['sr_sst'], self.cg_idx['sr_vip'])] = 0
            print('no SR SST to SR VIP\n')
            
        # have exc->vip connections
        if self.exc_to_vip==True:
            self.mask[np.ix_(self.cg_idx['pfc_exc'], self.cg_idx['pfc_vip'])] = 1
            self.mask[np.ix_(self.cg_idx['sr_exc'], self.cg_idx['sr_vip'])] = 1
            print('have exc to vip local connections\n', flush=True)
        
        # make PFC->SR VIP projection sparse
        if self.sparse_pfcesoma_to_srvip>0:
            print('make PFC to SR VIP connections sparse, sparsity={}\n'.format(self.sparse_pfcesoma_to_srvip))
            if self.sparse_pfcesoma_to_srvip>1 or self.sparse_pfcesoma_to_srvip<0:
                raise ValueError('sparsity must be between 0 and 1!')
            if (self.mask[np.ix_(self.cg_idx['pfc_esoma'], self.cg_idx['sr_vip'])]==0).all()==True:
                raise ValueError('no PFC to SR VIP connections!')
            n_disconnected = int(len(self.cg_idx['pfc_esoma'])*self.sparse_pfcesoma_to_srvip)    # for each SR VIP, the number of PFC Esoma that it does not receive input from
            for n in self.cg_idx['sr_vip']:
                disconnect_idx = random.sample(list(self.cg_idx['pfc_esoma']), n_disconnected)
                self.mask[disconnect_idx, n] = 0
#                 print('SR VIP #{}, disconnect with PFC Esoma {}'.format(n, disconnect_idx))
#                 print(len(disconnect_idx))
        
        # make SST->Edend projection sparse
        if 'sr_sst' in self.cell_group_list and self.sparse_srsst_to_sredend>0:
            print('make SR SST to SR Edend connections sparse, sparsity={}\n'.format(self.sparse_srsst_to_sredend))
            if self.sparse_srsst_to_sredend>1:
                raise ValueError('sparsity must be between 0 and 1!')
            if (self.mask[np.ix_(self.cg_idx['sr_sst'], self.cg_idx['sr_edend'])]==0).all()==True:
                raise ValueError('no SR SST to SR Edend connections!')
            n_disconnected = int(len(self.cg_idx['sr_edend'])*self.sparse_srsst_to_sredend)    # for each SR SST, the number of SR Edends that it does not connect to
            for n in self.cg_idx['sr_sst']:
                disconnect_idx = random.sample(list(self.cg_idx['sr_edend']), n_disconnected)
#                 print('sr_sst {} does not connect to sr_edend {}'.format(n, disconnect_idx), flush=True)
                self.mask[n, disconnect_idx] = 0
            
            
        # divide SR SST and SR VIP into two subpopulations
        if self.divide_sr_sst_vip==True:
            print('divide SR SST and SR VIP\n')
            self.cg_idx['sr_vip_1'] = np.arange(self.cg_idx['sr_vip'][0], self.cg_idx['sr_vip'][0]+len(self.cg_idx['sr_vip'])//2)
            self.cg_idx['sr_vip_2'] = np.array([n for n in self.cg_idx['sr_vip'] if n not in self.cg_idx['sr_vip_1']])
            self.cg_idx['sr_sst_1'] = np.arange(self.cg_idx['sr_sst'][0], self.cg_idx['sr_sst'][0]+len(self.cg_idx['sr_sst'])//2)
            self.cg_idx['sr_sst_2'] = np.array([n for n in self.cg_idx['sr_sst'] if n not in self.cg_idx['sr_sst_1']])
#             print('sr_vip_1 idx: {},\nsr_vip_2 idx: {},\nsr_sst_1 idx: {},\nsr_sst_2 idx: {}\n'.format(self.cg_idx['sr_vip_1'], self.cg_idx['sr_vip_2'], self.cg_idx['sr_sst_1'], self.cg_idx['sr_sst_2']))
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
#                 print('n_srsst={}'.format(n_srsst))
                n_sredend_start = self.cg_idx['sr_edend'][0] + n_dend_per_sst*n_srsst
                n_sredend_end = n_sredend_start + n_dend_per_sst
                self.mask[self.cg_idx['sr_sst'][n_srsst], n_sredend_start:n_sredend_end] = -1  
#                 print('sst idx: {}, start: {}, end: {}'.format(self.cg_idx['sr_sst'][n_srsst], n_sredend_start, n_sredend_end))
                
        # branch specific: in a given E cell, one branch is targeted by SST group 1, the other by SST group 2
        if self.structured_sr_sst_to_sr_edend_branch_specific==True:
            print('Intializaing a structured SR Edend to SR SST connectivity using mask (branch specific)\n')
            self.mask[np.ix_(self.cg_idx['sr_sst'], self.cg_idx['sr_edend'])] = 0
            dend1_idx = np.arange(self.cg_idx['sr_edend'][0], self.cg_idx['sr_edend'][0]+len(self.cg_idx['sr_edend'])//2)
            dend2_idx = np.array([nd for nd in self.cg_idx['sr_edend'] if nd not in dend1_idx])
            self.mask[np.ix_(self.cg_idx['sr_sst_1'], dend1_idx)] = -1
            self.mask[np.ix_(self.cg_idx['sr_sst_2'], dend2_idx)] = -1
#             print(self.cg_idx['sr_sst_1'], dend1_idx)
#             print(self.cg_idx['sr_sst_2'], dend2_idx)
                
        # TODO structured sr_esoma to sr_sst
        if False:    # cross inhibition from SR Esoma to SR SST
            self.cg_idx['sr_esoma_1'] = np.arange(self.cg_idx['sr_esoma'][0], self.cg_idx['sr_esoma'][0]+len(self.cg_idx['sr_esoma'])//2)
            self.cg_idx['sr_esoma_2'] = np.array([n for n in self.cg_idx['sr_esoma'] if n not in self.cg_idx['sr_esoma_1']])
            self.mask[np.ix_(self.cg_idx['sr_esoma_1'], self.cg_idx['sr_sst_1'])] = 0
            self.mask[np.ix_(self.cg_idx['sr_esoma_2'], self.cg_idx['sr_sst_2'])] = 0
        
        #=== define w_fix ===#
        self.w_fix = nn.Parameter(torch.zeros(self.total_n_neurons, self.total_n_neurons), requires_grad=False)
        if self.trainable_dend2soma==False:
            self.dend2soma = nn.Parameter(torch.Tensor([1]), requires_grad=False)    # magnitude of the coupling between dendrite and soma
        elif self.trainable_dend2soma==True:
            print('trainable dendrite to soma coupling\n')
            self.dend2soma = nn.Parameter(torch.Tensor([1]), requires_grad=True)
#         if self.divisive_dend_inh==True or self.divisive_dend_ei==True:
#             print('using divisive dendrites\n')
#             if self.divisive_dend_inh==True:
#                 print('inhibition-controlled division\n')
#             elif self.divisive_dend_ei==True:
#                 print('EI-controlled division\n')
#             if self.trainable_dend2soma==True:
#                 raise ValueError('when using divisive dendrite, should set the dend2soma to 0!')
#             self.dend2soma = nn.Parameter(torch.Tensor([0]), requires_grad=False)
#             print('setting dend2soma to 0\n')
        if 'divisive' in self.dendrite_type:
            print('divisive dendrite type: {}\n'.format(self.dendrite_type))
            self.dend2soma = nn.Parameter(torch.Tensor([0]), requires_grad=False)    # TODO: this needs change!!
            print('setting dend2soma to 0\n')
            
        # multiple dendritic branches per soma
        if len(self.dend_idx)!=0:
            print('# of branches per neuron: {}'.format(self.n_branches))
            for nb in range(self.n_branches):
                dend_idx_sr_b = (self.cg_idx['sr_edend'][0] + nb*len(self.cg_idx['sr_esoma'])) + np.arange(len(self.cg_idx['sr_esoma']))    # the dendrite idx for the nb'th dendritic branch
                self.w_fix[np.ix_(dend_idx_sr_b, self.cg_idx['sr_esoma'])] = self.dend2soma*torch.eye(self.n['sr_esoma'])
                dend_idx_pfc_b = (self.cg_idx['pfc_edend'][0] + nb*len(self.cg_idx['pfc_esoma'])) + np.arange(len(self.cg_idx['pfc_esoma']))    # the dendrite idx for the nb'th dendritic branch
                self.w_fix[np.ix_(dend_idx_pfc_b, self.cg_idx['pfc_esoma'])] = self.dend2soma*torch.eye(self.n['pfc_esoma'])
        
        
        
        
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
        if hp['task']=='cxtdm' or hp['task']=='salzman':
            self.w_prev_choice = nn.Parameter(torch.Tensor(self.n['output']-1, self.total_n_neurons), requires_grad=True)    # because have to ignore the fixation dimension
            self.mask_prev_choice = nn.Parameter(torch.zeros(self.n['output']-1, self.total_n_neurons), requires_grad=False)
        elif hp['task']=='wcst':
            self.w_prev_choice = nn.Parameter(torch.Tensor(self.n['output'], self.total_n_neurons), requires_grad=True)   
            self.mask_prev_choice = nn.Parameter(torch.zeros(self.n['output'], self.total_n_neurons), requires_grad=False)
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
        elif self.sr_sst_high_bias==True:    # initialize SST bias at a high value but allow change during training
            print('SR SST has high bias\n')
            self.bias = nn.Parameter(torch.zeros(self.total_n_neurons), requires_grad=False)
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
        if self.scale_down_wexc_init==True:
            if len(self.i_idx)!=0:
                self.w_rec.data[self.e_idx,:] /= (len(self.e_idx)/len(self.i_idx))
#         if self.scale_down_init_wexc == True:
#             self.w_rec.data[self.e_idx,:] /= 10
        
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
                print('initializing SR SST bias to be {}\n'.format(self.bias[self.cg_idx['sr_sst']]))
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

#         relu = nn.ReLU()    # this is slow!!!!!
#         tanh = nn.Tanh()
#         sigmoid = nn.Sigmoid()

        if nonlinearity=='relu':
            f = torch.relu(input_current)
        elif nonlinearity=='softplus':
            f = torch.log(1 + torch.exp(input_current))
        elif nonlinearity=='tanh':
            f = torch.tanh(input_current)
        elif nonlinearity=='sigmoid':
            f = torch.sigmoid(input_current)
        elif nonlinearity=='relu_satu':
            f = torch.relu(k_relu_satu*torch.tanh(input_current/k_relu_satu))    # this function approximates relu for [0,k_relu_satu] but is bounded by k_relu_satu
        else:
            raise NotImplementedError
        
        return f
        
        
    def nonlinearity_dend(self, g_e, g_i, threshold=1, width=1, b_g=5.56, g_LD=4, k=9.64, gamma=6.54, nonlinearity='subtractive'):
        """ Dendritic nonlinearity adapted from Robert Yang's 2016 Nat Comm paper 
            Args:
                g_e: excitatory input
                g_i: inhibitory input
                threshold, width: for divisive dendritic nonlinearity only
        """
                
#         if nonlinearity=='old':    # in this case the width of the tanh function does not change much with g_i
#             g_half = g_i
#             beta = g_i**0.5+1e-3
#             dend_current = torch.tanh((g_e - g_half)/beta) 
        if nonlinearity=='subtractive':    # 1. the width of the tanh function changes with g_i; 2. saturates at 1 irrespective of g_i; 3. output is inhibition-dependent when g_e=0
#             threshold = torch.tensor(g_i)    # original
#             threshold = g_i/10   # new
#             width = g_i**0.5+1e-3    # original
#             width = 1    # new
            dend_current = torch.tanh(g_e - g_i) 
        
        elif nonlinearity=='subtractive_width':    # where the width also increases with inhibition
            dend_current = torch.tanh((g_e - g_i)/(g_i**0.5+1e-4)) 
            
        elif nonlinearity=='subtractive_scaleInh':    # the effect of inhibition is scaled down
            dend_current = torch.tanh(g_e - 0.5*g_i) 
            
        elif nonlinearity=='subtractive_2':
            dend_current = self.nonlinearity_dend(g_e=g_e, g_i=g_i, nonlinearity='subtractive_3')
            dend_current = 2/(1-torch.tanh(torch.tensor(-1))) * dend_current - 1
            
        elif nonlinearity=='subtractive_3':   # the width of the tanh function changes with g_i
            epsilon = 1e-3     # so that it is well-defined when g_e=g_i=0
            g_i_prime = g_i+epsilon 
            dend_current = torch.tanh((g_e-g_i_prime)/g_i_prime) - torch.tanh(torch.tensor(-1))    # to make it 0 when g_e is 0
            
        elif nonlinearity=='subtractive_rectified':   # a more dramatic gating function
            epsilon = 1e-3     # so that it is well-defined when g_e=g_i=0
            g_i_prime = g_i+epsilon
            dend_current = torch.tanh((torch.relu(g_e-g_i_prime))/(g_i_prime))
            
        elif nonlinearity=='divisive':
            threshold = torch.tensor(threshold)
            width = torch.tensor(width)
#             threshold = torch.tensor(0.2)
#             width = torch.tensor(0.2)
            k1 = 1/torch.exp(g_i)
            k2 = - torch.tanh(-threshold/width)    # ensure that when g_i=g_e=0, the output is 0
            dend_current = k1*(torch.tanh((g_e-threshold)/width) + k2)    # when g_e=0, output is 0 independent of g_i
            
        elif nonlinearity=='divisive_2':
#             dend_current = self.nonlinearity_dend(g_e=g_e+1, g_i=g_i, nonlinearity='divisive') - 0.5    # when g_e=0, output is not 0 (old)
            threshold = torch.tensor(1)
            width = torch.tensor(1)
            k1 = 1/(torch.exp(g_i))
            k2 = 1 + torch.tanh(-threshold/width)
#             k3 = -1
            dend_current = k1 * (1 + torch.tanh((g_e-threshold)/width)) - k2
        
        elif nonlinearity=='divisive_3':
            threshold = torch.tensor(1)
            dend_current = 1/(torch.exp(g_i)) * (torch.tanh(g_e-threshold) - torch.tanh(-threshold))
            
        elif nonlinearity=='step':
            epsilon = 1e-3     # so that it is well-defined when g_i=0
            g_half = g_i 
            beta = g_i/100 + epsilon
            dend_current = torch.tanh((g_e - g_half)/beta) 
            
            
#         elif nonlinearity=='old_std':    # standardized 
#             g_half = g_i
#             beta = 1
#             dend_current = torch.tanh((g_e - g_half)/beta) 
        
#         elif nonlinearity=='v3_std':
#             pass
#         elif nonlinearity=='step_std':
#             pass
#         elif nonlinearity=='subtract':
#             # this version might be problematic, because when g_e=g_i=0, output=-1, which is inconsistent with exps
#             threshold = torch.tensor(1)
#             width = torch.tensor(1)
#             ceiling = torch.tanh(1/(g_i+1e-3))
#             k1 = (ceiling+1)/(1-torch.tanh(-threshold/width))
#             k2 = ceiling - k1
#             dend_current = k1*torch.tanh((g_e-threshold)/width) + k2    # ensure 1. when g_e=0. dend_current=-1; 2. when g_e=+infinity, dend_current=ceiling
#             dend_current = ceiling * torch.tanh((g_e - threshold)/width) - ceiling * (torch.tanh(-threshold/width)) - 1
#             dend_current = torch.tanh*(-threshold/width)
#             dend_current = torch.tanh((g_e-threshold)/width) - g_i**0.5
#         print(dend_current)


        return dend_current
    
    
    
        
        
    def effective_weight(self, w, mask, w_fix=0):
        """ compute the effective weight """
    
        w_eff = torch.abs(w) * mask + w_fix  
    
        return w_eff
    
    
    
    def forward(self, input, init, **kwargs):
        """ Evolve the two-area RNN in time for the duration of 1 trial 
            Args:
            - stim: stimulus input to the SR network. batch_size*n_in*n_timesteps
            - init: a dict that contains the initial state of the network. If None, initialize with 0.
            Output:
            - last_states: 
            - record: the activity over time
        """
        
        if self.time_it==True:
            start = time.time()
        
        n_steps = input.shape[0]
        hiddens = []
        hidden_dict = {}    # test, delete later
        i_mes = []
        device = next(self.parameters()).device
        
        # set init hidden state
        if init['h'] is None:
            hidden = torch.zeros([self.batch_size, self.total_n_neurons]).to(device)    # network state is 0 at t=0
        else:
            hidden = init['h']
        next_h = torch.zeros([self.batch_size, self.total_n_neurons]).to(device)
#         print('self.batch_size = {}'.format(self.batch_size), flush=True)
        
        # compute effective weight
        if len(self.i_idx)!=0:
            self.w_rec_eff = self.effective_weight(w=self.w_rec, mask=self.mask, w_fix=torch.abs(self.dend2soma)*self.w_fix)
        elif len(self.i_idx)==0:    # only E cell, then do not apply Dale's law
            self.w_rec_eff = self.w_rec
        self.w_in_eff = self.effective_weight(w=self.w_in, mask=self.mask_in)
        self.w_prev_rew_eff = self.effective_weight(w=self.w_rew, mask=self.mask_rew)
        self.w_prev_choice_eff = self.effective_weight(w=self.w_prev_choice, mask=self.mask_prev_choice)
        self.w_prev_stim_eff = self.effective_weight(w=self.w_prev_stim, mask=self.mask_prev_stim)
        
        
        # optogenetics params
        if 'opto' in kwargs and kwargs['opto']!=None:
            opto_ts = kwargs['opto']['t']    # the timesteps during which optogenetics is on
            opto_neuron_idx = kwargs['opto']['neuron_idx']    # the indices of neurons that are targeted by optogenetics
            opto_value = kwargs['opto']['value']    # the activation values that opto drives the neuron to
            
            
        # project input into the network
        total_input = input@self.w_in_eff
        if 'trial_history' in kwargs:    # add the trial history input
#             print('i_prev_rew shape: {}, self.w_prev_rew_eff shape: {}\n'.format(kwargs['trial_history']['i_prev_rew'].shape, self.w_prev_rew_eff.shape), flush=True)
#             print('i_prev_choice shape: {}, self.w_prev_choice_eff shape: {}\n'.format(kwargs['trial_history']['i_prev_choice'].shape, self.w_prev_choice_eff.shape), flush=True)
#             print('i_prev_stim shape: {}, self.w_prev_stim_eff shape: {}\n'.format(kwargs['trial_history']['i_prev_stim'].shape, self.w_prev_stim_eff.shape), flush=True)
#             print('i_prev_rew shape: {}, w_prev_rew_eff shape: {}, i_prev_choice shape: {}, w_prev_choice_eff shape: {}, i_prev_stim shape: {}, w_prev_stim_eff shape: {}'.format(kwargs['trial_history']['i_prev_rew'].shape, self.w_prev_rew_eff.shape, kwargs['trial_history']['i_prev_choice'].shape, self.w_prev_choice_eff.shape, kwargs['trial_history']['i_prev_stim'].shape, self.w_prev_stim_eff.shape))
            total_input += kwargs['trial_history']['i_prev_rew']@self.w_prev_rew_eff + kwargs['trial_history']['i_prev_choice']@self.w_prev_choice_eff + kwargs['trial_history']['i_prev_stim']@self.w_prev_stim_eff

        # compute the noise 
        ## 1. O-U process
        # kappa = 0.1
        # theta = 0
        # sigma = 1
        # W = scipy.stats.norm.rvs(loc=0, scale=1, size=(n_steps-1, self.batch_size))
        # if 'noise' not in init.keys() or init['noise'] is None:
        #     _noise0 = 0
        # else:
        #     _noise0 = init['noise']
        # _noise = torch.zeros([n_steps, self.batch_size])   
        # _noise[0, :] = _noise0
                
        # for t in range(0, n_steps-1):
        #     _noise[t+1, :] = _noise[t, :] + kappa*(theta - _noise[t, :])*self.dt + sigma*np.sqrt(self.dt)*W[t, :]    # O-U process
        # noise_last = _noise[-1, :]
        
        # noise = torch.zeros([n_steps, self.batch_size, self.total_n_neurons])
        # noise[:, :, self.pfc_idx] = _noise.unsqueeze(dim=2)    # only apply noise to pfc


        ## 2. independent noise
        # noise = torch.randn_like(hidden)*torch.sqrt(torch.tensor(self.decay))
        # noise_last = None

        ## 3. Markov process
        # noise = torch.zeros([n_steps, self.batch_size, self.total_n_neurons])
        # if 'noise' not in init.keys() or init['noise'] is None:
        #     state = random.choice(['high', 'low'])
        # else:
        #     state = init['noise']
        # for t in range(0, n_steps):
        #     if np.random.rand() < 0.001:    # switch hidden state
        #         if state=='high':
        #             state = 'low'
        #         elif state=='low':
        #             state = 'high'
        #     if state=='high':
        #         amp = 1    # noise amplitude
        #     elif state=='low':
        #         amp = 0
        #     noise[t, :, self.pfc_idx] = amp * torch.normal(mean=0, std=torch.ones(self.batch_size, 1))
        # noise_last = state
             
        ## plot the noise
    #     if np.random.rand()<0:    # don't plot all the time
    #         fig, ax = plt.subplots(3, 1)
    #         fig.patch.set_facecolor('white')
    #         for i in range(self.batch_size):
    #             ax[0].plot(torch.mean(self.network_noise * noise[:, i, :], axis=1))
    #         ax[1].plot(torch.mean(self.network_noise * noise, dim=(1,2)))
    #         ax[2].plot(torch.std(self.network_noise * noise, dim=(1,2)))
    # #         print('std of noise', torch.std(_noise, dim=1))
    #         ax[0].set_title('noise')
    #         ax[1].set_title('mean')
    #         ax[2].set_title('std')
    #         fig.tight_layout()
    #         plt.show()
        
        
        if self.time_it==True:
            print('before looping over time takes {:.2e}s'.format(time.time()-start))
    
        # start the loop over time
        for t in range(n_steps): 
            if self.time_it==True:
                print('\tt = {}'.format(t))
            
            # compute E and I input separately
            if self.time_it==True:
                start_computeEI = time.time()
            if not (len(self.dend_idx)==0 and self.mglur==False):   # in which case separate E and I inputs are needed to update dendrites or compute mGluR current
#                 i_exc = hidden[:,self.e_idx]@self.w_rec_eff[self.e_idx, :] + input[t,:,:]@self.w_in_eff
                i_exc = hidden[:,self.e_idx]@self.w_rec_eff[self.e_idx, :] + total_input[t,:,:]    # TODO: should be this??
                i_inh = hidden[:,self.i_idx]@self.w_rec_eff[self.i_idx, :]
            if self.time_it==True:
                print('\tcompute EI currents takes {:.2e}s, cumulative {:.2e}s'.format(time.time()-start_computeEI, time.time()-start), flush=True)  
            
            # compute mGluR current
            if self.time_it==True:
                start_mglur = time.time()
            if self.mglur==True: 
                if t==0:
                    if init['i_me']==None:
                        i_me = torch.zeros([self.batch_size, self.total_n_neurons]).to(device)
                    else:
                        i_me = init['i_me'].to(device)    # do not reset
                else:
                    i_me = (1-self.alpha_me)*i_me + self.alpha_me*i_exc[:, self.mglur_e_idx]
            elif self.mglur==False:
                i_me = torch.zeros([self.batch_size, self.total_n_neurons]).to(device)
            if self.time_it==True:
                print('\tcalculate mGluR takes {:.2e}s, cumulative {:.2e}s'.format(time.time()-start_mglur, time.time()-start), flush=True)  
                
                
            # update dendrite
            # TODO: update dend first?
            if self.time_it==True:
                start_updatedend = time.time()
            if len(self.dend_idx)!=0:
                if self.dendrite_type=='divisive_inh':
                    next_h[:,self.dend_idx] = i_inh[:,self.dend_idx]
                elif self.dendrite_type=='divisive_ei':
                    next_h[:,self.dend_idx] = i_exc[:,self.dend_idx] + i_inh[:,self.dend_idx]
                elif self.dendrite_type=='additive' or self.dendrite_type=='divisive_nonlinear':
                    next_h[:,self.dend_idx] = self.nonlinearity_dend(nonlinearity=self.dend_nonlinearity, g_e=i_exc[:,self.dend_idx], g_i=torch.abs(i_inh[:,self.dend_idx]))
            if self.time_it==True:
                print('\tupdate dendrite takes {:.2e}s, cumulative {:.2e}s'.format(time.time()-start_updatedend, time.time()-start), flush=True)   
            
            
            # update soma
            start_updatesoma = time.time()
            # TODO: change to "if self.dendrite_type=='additive' or something like that"
#             if self.divisive_dend_inh==False and self.divisive_dend_ei==False and self.divisive_dend_nonlinear==False:
            if self.dendrite_type=='additive':
                next_h = (1-self.decay)*hidden + self.decay*self.nonlinearity_soma(input_current=hidden@self.w_rec_eff + total_input[t,:,:] + self.mask_bias*self.bias, nonlinearity=self.activation, k_relu_satu=self.k_relu_satu)
#             elif self.divisive_dend_inh==True or self.divisive_dend_ei==True:
            elif self.dendrite_type=='divisive_ei' or self.dendrite_type=='divisive_inh' or self.dendrite_type=='divisive_nonlinear':
                gain = torch.ones([self.batch_size, self.total_n_neurons]).to(device)    # initialize gain
                input_dend_e_sum_sr = i_exc[:, self.dend_idx_sr].reshape((self.batch_size, self.n_branches, -1))
                input_dend_e_sum_sr = torch.sum(input_dend_e_sum_sr, axis=1)    # sum of exc current that goes into all the branches
                input_dend_i_sum_sr = i_inh[:, self.dend_idx_sr].reshape((self.batch_size, self.n_branches, -1))
                input_dend_i_sum_sr = torch.sum(input_dend_i_sum_sr, axis=1)    # sum of exc current that goes into all the branches
                input_dend_e_sum_pfc = i_exc[:, self.dend_idx_pfc].reshape((self.batch_size, self.n_branches, -1))
                input_dend_e_sum_pfc = torch.sum(input_dend_e_sum_pfc, axis=1)    # sum of exc current that goes into all the branches
                input_dend_i_sum_pfc = i_inh[:, self.dend_idx_pfc].reshape((self.batch_size, self.n_branches, -1))
                input_dend_i_sum_pfc = torch.sum(input_dend_i_sum_pfc, axis=1)    # sum of exc current that goes into all the branches
#                 input_dend_e_sum = torch.cat((input_dend_e_sum_sr, input_dend_e_sum_pfc), axis=1)    # concat SR and PFC
#                 input_dend_i_sum = torch.cat((input_dend_i_sum_sr, input_dend_i_sum_pfc), axis=1)
#                 print(input_dend_e_sum.shape, input_dend_i_sum.shape)
#                 if self.divisive_dend_inh==True:
                if self.dendrite_type=='divisive_inh':
                    raise ValueError('needs change!')
#                     gain = torch.sigmoid(input_dend_i_sum)
#                     total_input[t, : ,self.esoma_idx] += input_dend_e_sum
#                     next_h[:, self.esoma_idx] = (1-self.decay)*hidden[:, self.esoma_idx] + self.decay*gain*self.nonlinearity_soma(input_current=total_input[t, :, self.esoma_idx] + self.mask_bias[self.esoma_idx]*self.bias[self.esoma_idx], nonlinearity=self.activation, k_relu_satu=self.k_relu_satu) 
                elif self.dendrite_type=='divisive_ei': 
                    raise ValueError('needs change!')
# #                 elif self.divisive_dend_ei==True:
#                     gain = torch.sigmoid(input_dend_i_sum + input_dend_e_sum)
#                     # TODO: this is also problematic - the current to the dend does not get passed to soma. maybe add "total_input[t, : ,self.esoma_idx] += input_dend_e_sum + input_dend_i_sum" here
#                     next_h[:, self.esoma_idx] = (1-self.decay)*hidden[:, self.esoma_idx] + self.decay*gain*self.nonlinearity_soma(input_current=total_input[t, :, self.esoma_idx] + self.mask_bias[self.esoma_idx]*self.bias[self.esoma_idx], nonlinearity=self.activation, k_relu_satu=self.k_relu_satu) 
# #             elif self.divisive_dend_nonlinear==True:
                elif self.dendrite_type=='divisive_nonlinear':
    #                 print('hidden[dend_idx_sr]={}'.format(hidden[:, self.dend_idx_sr]))
                    dend_sum_sr = hidden[:, self.dend_idx_sr].reshape((self.batch_size, self.n_branches, -1))
                    dend_sum_sr = torch.sum(dend_sum_sr, axis=1)   # sum over branches
    #                 print('hidden[dend_idx_pfc]={}'.format(hidden[:, self.dend_idx_pfc]))
                    dend_sum_pfc = hidden[:, self.dend_idx_pfc].reshape((self.batch_size, self.n_branches, -1))
                    dend_sum_pfc = torch.sum(dend_sum_pfc, axis=1)   # sum over branches
#                     gain = torch.cat((dend_sum_sr, dend_sum_pfc), axis=1)    # nbatches * nEsomas
#                     gain = torch.relu(gain)/2    # so that it is always between 0 and 1
                    gain[:, self.cg_idx['sr_esoma']] = dend_sum_sr/2    # the gain of soma depends on the sum of the dendritic current
                    gain[:, self.cg_idx['pfc_esoma']] = dend_sum_pfc/2
    #                 print('dend_sum_sr={}, dend_sum_pfc={}, gain={}'.format(dend_sum_sr, dend_sum_pfc, gain))
#                     print('max gain among E somas={}'.format(torch.max(gain[:, self.esoma_idx])))
#                     next_h[:, self.esoma_idx] = (1-self.decay)*hidden[:, self.esoma_idx] + self.decay*gain*self.nonlinearity_soma(input_current=total_input[t, :, self.esoma_idx] + input_dend_e_sum + input_dend_i_sum + self.mask_bias[self.esoma_idx]*self.bias[self.esoma_idx], nonlinearity=self.activation, k_relu_satu=self.k_relu_satu) 
                    # TODO: update PV and others!!!!
                    total_input[t, :, self.cg_idx['sr_esoma']] += input_dend_e_sum_sr + input_dend_i_sum_sr
                    total_input[t, :, self.cg_idx['pfc_esoma']] += input_dend_e_sum_pfc + input_dend_i_sum_pfc
                    next_h = (1-self.decay)*hidden + self.decay*gain*self.nonlinearity_soma(input_current=total_input[t, :, :] + self.mask_bias*self.bias, nonlinearity=self.activation, k_relu_satu=self.k_relu_satu) 
            
            # optogenetic activation/silencing of specific populations
#             print(kwargs.keys())
            if 'opto' in kwargs and kwargs['opto']!=None:
                if t in opto_ts:
                    next_h[:, opto_neuron_idx] = opto_value
#                     print('opto working...')
            
            if self.time_it==True:
                print('\tupdate soma takes {:.2e}s, cumulative {:.2e}s'.format(time.time()-start_updatesoma, time.time()-start), flush=True)   

            
                
            # add noise
            if self.time_it==True:
                start_addnoise = time.time()
            next_h += self.network_noise*torch.randn_like(next_h)    # add some noise
#             next_h += self.network_noise*torch.randn_like(next_h)*torch.sqrt(torch.tensor(self.decay))    # add some noise. NEW: scale the noise with sqrt(\alpha). Think about a system with \tau * (x_{t+1} - x{t}) = noise. Want the same std if integrating for 1 time constant with different \tau and dt's
            # next_h += self.network_noise * noise[t, :, :]
            hidden = next_h
            if self.time_it==True:
                print('\tadd noise takes {:.2e}s, cumulative {:.2e}s'.format(time.time()-start_addnoise, time.time()-start), flush=True)
         
            # collect stuff
            if self.time_it==True:
#             print(kwargs.keys())
#             if kwargs['hp']['time_it']==True:
                start_collect = time.time()
            hiddens.append(hidden)
#             hidden_dict[t] = copy.deepcopy(hidden)    # test, delete later
#             print('hidden_dict[t][dend_idx] = {}'.format(hidden[:, self.dend_idx]))
#             print('hidden[dend]={}'.format(hidden[:, self.dend_idx]))
#             print('shape of hidden: {}'.format(hidden.shape))
#             print('mean of hidden={}'.format(torch.mean(hidden)))
#             print('if there is non-zeros in hidden: {}'.format((hidden.detach().numpy()!=0).any()))
#             print('hiddens={}'.format(hiddens))
            i_mes.append(i_me)
            if self.time_it==True:
                print('\tcollect stuff takes {:.2e}s, cumulative {:.2e}s'.format(time.time()-start_collect, time.time()-start), flush=True)
        
        # last state
        if self.time_it==True:
            start_afterloop = time.time()
        h_last = hidden     # the state at the last timepoint
        i_me_last = i_me
        if self.time_it==True:
            print('everything after the loop takes {:.2e}s, cumulative {:.2e}s'.format(time.time()-start_afterloop, time.time()-start))
            print('total time for the forward pass: {:.2e}s'.format(time.time()-start))
        
        last_states = {'hidden': h_last, 'i_me': i_me_last, 'noise': None}
        record = {'hiddens': hiddens, 'i_mes': i_mes, 'noise': None}
        
        hiddens = torch.stack(hiddens, dim=0).cpu().detach().numpy()
#         print('shape of hiddens: {}'.format(hiddens.shape))
#         print('the 6th timestep: {}'.format(hiddens[6, :, :][:, self.dend_idx]))
#         print('if there is non-zeros in hiddens: {}'.format((hiddens!=0).any()))
#         print('mean of hiddens: {}'.format(np.mean(hiddens)))
#         for t in hidden_dict.keys():
#             print('t={}, hidden={}, non-zero: {}'.format(t, hidden_dict[t][:, self.dend_idx], (hidden_dict[t]!=0).any()))
        
        return last_states, record
        
          
            
    
class Net_readoutSR_working(torch.nn.Module):
    """ The network model. 
        Read out movement from SR.
        Read out rule from PFC.
    """
    
    def __init__(self, hp, **kwargs):
        super().__init__()
        self.rnn = BioRNN_working(hp)
        self.pos_wout = hp['pos_wout']
        self.pos_wout_rule = hp['pos_wout_rule']
        self.mask_out = nn.Parameter(torch.zeros(self.rnn.total_n_neurons, self.rnn.n['output']), requires_grad=False)    # the mask for the input weights
        self.mask_out[self.rnn.cg_idx['sr_esoma'], :] = 1    # long range connections are excitatory
        if self.rnn.train_rule==True:
            self.mask_out_rule = nn.Parameter(torch.zeros(self.rnn.w_out_rule.shape), requires_grad=False)
            self.mask_out_rule[self.rnn.cg_idx['pfc_esoma'], :] = 1    # long range connections are excitatory
        if 'output_noise' not in hp.keys():
            self.output_noise = hp['output_noise'] = 0
        else:
            self.output_noise = hp['output_noise']
           
    def forward(self, input, init, **kwargs):
#         if 'trial_history' in kwargs.keys():
#             last_states, record = self.rnn(input, init, trial_history=kwargs['trial_history'])
#         else:    # do not have trial history input
#             last_states, record = self.rnn(input, init)
        last_states, record = self.rnn(input, init, **kwargs)
        # compute output
        if self.pos_wout==False:
            self.w_out_eff = self.rnn.w_out*self.mask_out
        elif self.pos_wout==True:
            self.w_out_eff = self.rnn.effective_weight(w=self.rnn.w_out, mask=self.mask_out)
        rnn_activity = torch.stack(record['hiddens'], dim=0)
        out = rnn_activity@self.w_out_eff
        out += self.output_noise * torch.rand_like(out)
        if self.rnn.train_rule==True:
            if self.pos_wout_rule==False:
                self.w_out_rule_eff = self.rnn.w_out_rule*self.mask_out_rule
            elif self.pos_wout_rule==True:
                self.w_out_rule_eff = self.rnn.effective_weight(w=self.rnn.w_out_rule, mask=self.mask_out_rule)
            out_rule = rnn_activity@self.w_out_rule_eff
        else:
            out_rule = 0
            
        # collect stuff
        out = {'out': out, 'out_rule': out_rule}
        data = {'record': record, 'last_states': last_states}
        
        return out, data