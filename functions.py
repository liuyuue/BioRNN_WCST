import numpy as np; np.set_printoptions(precision=2); np.random.seed(0)
import torch; torch.set_printoptions(precision=2)
import torch.nn as nn
import matplotlib.pyplot as plt; plt.rc('font', size=50); plt.rc('font', family='Arial')
import matplotlib 
from matplotlib.font_manager import FontProperties
from mpl_toolkits import mplot3d

import seaborn as sns
import time
import sys
import os
import itertools
import random; random.seed(0)
import datetime
import pickle
import copy
import pandas as pd
import scipy

from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from textwrap import wrap


from model import *
from task import *
from model_working import *

# print(torch.__version__)
# print(sys.version)
                
# %matplotlib inline



def get_default_hp():
    '''Get a default hp.

    Returns:
        hp : a dictionary containing training hyperparameters
        optimizer: the type of optimizer (needs to be instantiated later)
        loss_fnc: the type of loss function
    '''
#     num_ring = task.get_num_ring(ruleset)
#     n_rule   = task.get_num_rule(ruleset)

#     n_eachring = 32
#     n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1


    hp = {
            # batch size for training
#             'batch_size_train': 64,
            # batch_size for testing
#             'batch_size_test': 512,
            # input type: normal, multi
#             'in_type': 'normal',
            # Type of RNNs : LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
#             'rnn_type': 'LeakyRNN',
            # whether rule and stimulus inputs are represented separately
#             'use_separate_input': False,
            # Type of loss functions
            'loss_type': 'mse',
            # initialization: diagonal, orthogonal, kaiming_normal, kaiming_uniform, normal, uniform, constant
            'initialization_weights': 'orthogonal',
            'initialization_bias': 'zero',
            # Optimizer
            'optimizer': 'adam',
            # Type of activation runctions, relu, softplus, tanh, elu
            'activation': 'relu',
            'k_relu_satu': 10,    # the saturating bound if using the saturating ReLU function
            # Time constant (ms)
            'tau': 100,
            # discretization time step (ms)
            'dt': 50,
            # discretization time step/time constant
#             'alpha': 0.2,
            # recurrent noise
#             'sigma_rec': 0.05,
            # input noise
#             'sigma_x': 0.01,
            # leaky_rec weight initialization, diag, randortho, randgauss
#             'w_rec_init': 'randortho',
            # a default weak regularization prevents instability
            'l1_h': 1e-3*0,
            # l2 regularization on activity
            'l2_h': 1e-3*0,
            # l2 regularization on weight
            'l1_weight': 1e-3*0,
            # l2 regularization on weight
            'l2_weight': 1e-3*0,
            # l2 regularization on recurrent E synapses of the SR network
            'l2_rec_e_weight_sr': 0,
            # l2 regulariztion on the neural activity of the E neurons in the SR network
            'l2_h_sr': 0,
            # l2 regulariztion on the neural activity of the E neurons in the PFC network
            'l2_h_pfc': 0,
            # l2 regularization on deviation from initialization
#             'l2_weight_init': 0,   
            # proportion of weights to train, None or float between (0, 1)
#             'p_weight_train': None,
            # Stopping performance
            'target_perf': 1,
            # number of units each ring
#             'n_eachring': n_eachring,
            # number of rings
#             'num_ring': num_ring,
            # number of rules
#             'n_rule': n_rule,
            # first input index for rule units
#             'rule_start': 1+num_ring*n_eachring,
            # number of input units
            'n_input': 5,
            # number of input units for rule cue
            'n_input_rule_cue': 4,
            # number of output units
            'n_output': 3,
            # number of PFC readout units
            'n_output_rule': 2,
            # number of recurrent units
            'cell_group_list':['sr_esoma', 'sr_edend', 'sr_pv', 'sr_sst', 'sr_vip',
                                'pfc_esoma', 'pfc_edend', 'pfc_pv', 'pfc_sst', 'pfc_vip'],
            'n_sr_esoma': int(70),     # default: 70
            'n_sr_edend': int(140),     # default: 140
            'n_sr_pv': int(10),     # default: 10
            'n_sr_sst': int(10),     # default: 10
            'n_sr_vip': int(10),    # default: 10
            'n_pfc_esoma': int(70),     # default: 70
            'n_pfc_edend': int(140),     # default: 140
            'n_pfc_pv': int(10),     # default: 10
            'n_pfc_sst': int(10),     # default: 10
            'n_pfc_vip': int(10),    # default: 10
            # number of input units
#             'ruleset': ruleset,
            # if save model and figure
            'save_model': True,
            'save_figures': True,
            # name to save
            'save_name': 'NA',
            # learning rate
            'learning_rate': 1e-3,
            # intelligent synapses parameters, tuple (c, ksi)
#             'c_intsyn': 0,
#             'ksi_intsyn': 0,
            'explicit_rule': False,
            'train_rule': True,
            'block_len': 20,
            'n_switches': 3,
            'n_batches_per_block': int(2e8),
            'n_blocks': int(1),
            'batch_size': int(10),
#             'batch_size_test': 1,
            'network_noise': 0.01,
            'input_noise_perceptual': 0.01,
            'input_noise_rule': 0.01,
            'switch_every': 10,    # switch every x batches
            'test': True,    # whether to freeze the weight and test once in a while during training
            'n_branches': 2,    # number of dendritic branches for each E cell
            'mglur': False,    # metabotropic glutamate receptor
            'divide_sr_sst_vip': False,     # two subgroups of SR SST and SR VIP (to encourage gating)
            'no_pfcesoma_to_srsst': False,
            'no_pfcesoma_to_sredend': False,
            'no_pfcesoma_to_srpv': False, 
            'no_srsst_to_srvip': False,
            'sr_sst_high_bias': False,
            'fdbk_to_vip': False,
            'dend_nonlinearity': 'old',    # old, v2, v3
            'trainable_dend2soma': False,
#             'divisive_dend_inh': False,
#             'divisive_dend_ei': False,
#             'divisive_dend_nonlinear': False,  
            'dendrite_type': 'additive',    # none/additive/divisive_nonlinear/divisive_ei/divisive_inh
            'scale_down_init_wexc': False,
            'grad_remove_history': True,
            'plot_during_training': False,
            'structured_sr_sst_to_sr_edend': False,
            'structured_sr_sst_to_sr_edend_branch_specific': False,
            'sparse_pfcesoma_to_srvip': 1,
            'pos_wout': False,    # whether the readout weight for response is positive
            'pos_wout_rule': False,    # whether the readout weight for rule is positive
            'task': 'cxtdm',
            'jobname': 'testjob',    # determined by the batch file
            'timeit_print': False,
            'torch_seed': 1
            }
    
#     if hp['optimizer']=='adam':
#         optimizer = torch.optim.Adam
#     elif hp['optimizer']=='Rprop':
#         optimizer = torch.optim.Rprop
#     else:
#         raise NotImplementedError
    optimizer = hp['optimizer']
    
    if hp['loss_type']=='mse':
        loss_fnc = nn.MSELoss()
    else:
        raise NotImplementedError

    return hp, optimizer, loss_fnc


















# import torch
# from . import _functional as F
# from .optimizer import Optimizer


# class Adam_OWM(Optimizer):
#     r"""Implements Adam algorithm 
#     with the orthogonal weight modification (Zeng et al., Nature Machine Intelligence 2019) 
#     It has been proposed in `Adam: A Method for Stochastic Optimization`_.
#     The implementation of the L2 penalty follows changes proposed in
#     `Decoupled Weight Decay Regularization`_.
#     Args:
#         params (iterable): iterable of parameters to optimize or dicts defining
#             parameter groups
#         lr (float, optional): learning rate (default: 1e-3)
#         betas (Tuple[float, float], optional): coefficients used for computing
#             running averages of gradient and its square (default: (0.9, 0.999))
#         eps (float, optional): term added to the denominator to improve
#             numerical stability (default: 1e-8)
#         weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#         amsgrad (boolean, optional): whether to use the AMSGrad variant of this
#             algorithm from the paper `On the Convergence of Adam and Beyond`_
#             (default: False)
#     .. _Adam\: A Method for Stochastic Optimization:
#         https://arxiv.org/abs/1412.6980
#     .. _Decoupled Weight Decay Regularization:
#         https://arxiv.org/abs/1711.05101
#     .. _On the Convergence of Adam and Beyond:
#         https://openreview.net/forum?id=ryQu7f-RZ
#     """

#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
#                  weight_decay=0, amsgrad=False):
#         if not 0.0 <= lr:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if not 0.0 <= eps:
#             raise ValueError("Invalid epsilon value: {}".format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
#         if not 0.0 <= weight_decay:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         defaults = dict(lr=lr, betas=betas, eps=eps,
#                         weight_decay=weight_decay, amsgrad=amsgrad)
#         super(Adam, self).__init__(params, defaults)

#     def __setstate__(self, state):
#         super(Adam, self).__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('amsgrad', False)

#     @torch.no_grad()
#     def step(self, A, closure=None):
#         """Performs a single optimization step.
#         Args:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#             A: columns are the data points that we want the weight update to be orthogonal to
#         """
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         for group in self.param_groups:
#             params_with_grad = []
#             grads = []
#             exp_avgs = []
#             exp_avg_sqs = []
#             max_exp_avg_sqs = []
#             state_steps = []
#             beta1, beta2 = group['betas']

#             for p in group['params']:
#                 if p.grad is not None:
#                     params_with_grad.append(p)
#                     if p.grad.is_sparse:
#                         raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
#                     grads.append(p.grad)

#                     state = self.state[p]
#                     # Lazy state initialization
#                     if len(state) == 0:
#                         state['step'] = 0
#                         # Exponential moving average of gradient values
#                         state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                         # Exponential moving average of squared gradient values
#                         state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                         if group['amsgrad']:
#                             # Maintains max of all exp. moving avg. of sq. grad. values
#                             state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

#                     exp_avgs.append(state['exp_avg'])
#                     exp_avg_sqs.append(state['exp_avg_sq'])

#                     if group['amsgrad']:
#                         max_exp_avg_sqs.append(state['max_exp_avg_sq'])

#                     # update the steps for each param group update
#                     state['step'] += 1
#                     # record the step after step update
#                     state_steps.append(state['step'])

#             F.adam(params_with_grad,
#                    grads,
#                    exp_avgs,
#                    exp_avg_sqs,
#                    max_exp_avg_sqs,
#                    state_steps,
#                    amsgrad=group['amsgrad'],
#                    beta1=beta1,
#                    beta2=beta2,
#                    lr=group['lr'],
#                    weight_decay=group['weight_decay'],
#                    eps=group['eps'])    # TODO: need to write a new F.adam_owm function
            
#         return loss




def train(model, hp, optimizer, loss_fnc, train_y, train_y_rule, y, y_rule, rnn_activity=None, retain_graph=True, freeze_sr_rec=False):
    """ Train the network """
    
    loss_mse = loss_fnc(train_y, y)
    if hp['train_rule']==True:
        loss_mse = loss_mse + loss_fnc(train_y_rule, y_rule)
    
    loss_reg_weights = model.rnn.l1_weights*torch.norm(model.rnn.w_rec_eff, p=1) + \
                       model.rnn.l2_weights*torch.norm(model.rnn.w_rec_eff, p=2)
    
    if rnn_activity is not None:
        loss_reg_activity = model.rnn.l1_h*torch.norm(rnn_activity, p=1) + model.rnn.l2_h*torch.norm(rnn_activity, p=2)
        loss_reg_h_sr = model.rnn.l2_h_sr*torch.norm(rnn_activity[:, model.rnn.cg_idx['sr_esoma'], :], p=2)
    else:
        loss_reg_activity = 0
        loss_reg_h_sr = 0
    
    
    loss_reg_exc_weights_sr = model.rnn.l2_rec_e_weight_sr*torch.norm(
                            model.rnn.w_rec_eff[np.ix_(model.rnn.cg_idx['sr_esoma'], model.rnn.cg_idx['sr_esoma'])], p=2)
    
    
    
    loss_reg = loss_reg_weights + loss_reg_activity + loss_reg_exc_weights_sr + loss_reg_h_sr
    total_loss = loss_mse + loss_reg
    
    optimizer.zero_grad()           # clear gradients for this training step
    total_loss.backward(retain_graph=retain_graph)           # backpropagation, compute gradients
    
    # freeze part of weight during update
    if freeze_sr_rec==True:
#         sr_idx = []
#         for cg in model.rnn.cell_group_list:
#             if 'sr' in cg:
#                 sr_idx.extend(model.rnn.cg_idx[cg])
        model.rnn.w_rec.grad[np.ix_(model.rnn.sr_idx, model.rnn.sr_idx)] = 0
    
    optimizer.step()                # apply gradients
    
#     return total_loss
    return total_loss, loss_mse, loss_reg





def get_perf(y, yhat, hp, hp_task):
    """ From the output and target, get the performance of the network 
        Args:
            y: batch_size*n_output*n_timesteps
            yhat: batch_size*n_output*n_timesteps
        Returns:
            resp_correct: length batch_size binary vector
    """
#     if y.size()[1]!=3 or yhat.size()[1]!=3:
#         raise ValueError('This function only works when there are 2 choices!')
    resp_start_ts = int(hp_task['resp_start']/hp['dt'])
    resp_end_ts = int(hp_task['resp_end']/hp['dt'])

    softmax = nn.Softmax(dim=1)

    y_choice = torch.mean(y[resp_start_ts:resp_end_ts, :, 0:2], dim=0)    # batch_size * 2
    choice_prob = y_choice    # softmax would soften the difference a lot and worsen the performance...

    choice = torch.zeros([choice_prob.shape[0], 2]).to(choice_prob.device)    # compute choices from choice probabilities
    choice[:,0] = torch.gt(choice_prob[:,0], choice_prob[:,1])
    choice[:,1] = torch.gt(choice_prob[:,1], choice_prob[:,0])
    
    target = torch.mean(yhat[resp_start_ts:resp_end_ts, :, 0:2], dim=0)
    target_prob = target
    
#     print('choice device: {}. target_prob device: {}'.format(choice.device, target_prob.device))
    match = torch.abs(choice - target_prob) <= 0.5    
#     match = torch.abs(choice_prob - target_prob) <= 0.5    # to prevent low activity for both output nodes
    resp_correct = match[:,0] * match[:,1]    # correct response if the probability from target is differed by less than threshold% for both choices

    return resp_correct, choice_prob, choice









def save_file(folder, file, name):
    with open(folder+'/'+ name + '.pkl', 'wb') as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)

def load_file(name):
    with open(folder+'/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def smooth(x, window_size=50):
    """ Smoothing a time series """
    
    x_smoothed = []
    for i in range(len(x)-window_size):
        x_smoothed.append(np.mean(x[i:i+window_size]))
    
    return x_smoothed
        
    
    
    
    
def compute_trial_history(last_rew, prev_stim, prev_choice, input_period, batch_size, n_steps, input_dim, hp_task, hp):
    """ Compute the input for trial history """
    
    # an additional input indicating the reward on the previous trial
#     I_prev_rew = torch.zeros([batch_size, 2, n_steps]).to(model.rnn.w_rec.device)
    I_prev_rew = torch.zeros([n_steps, batch_size, 2])
        
    if last_rew!=None:
        # test: does pytorch broadcast? 
        rew = torch.outer(last_rew.cpu().float(), torch.Tensor([1,0])) + torch.outer((~last_rew).cpu().float(), torch.Tensor([0,1]))
        rew = torch.unsqueeze(rew, 0)
#         I_prev_rew[:,:,input_period] = rew
        I_prev_rew[input_period,:,:] = rew

        
    # an additional input indicating the stimulus on the previous trial
#     I_prev_stim = torch.zeros([batch_size, input_dim, n_steps]).to(device)
    I_prev_stim = torch.zeros([n_steps, batch_size, input_dim])
    if prev_stim!=None:
#         stim = torch.mean(prev_stim[:,:,int(hp_task['stim_start']/hp['dt']):int(hp_task['stim_end']/hp['dt'])], axis=-1)   # the mean over the stimulus presentation period
        stim = torch.mean(prev_stim[int(hp_task['stim_start']/hp['dt']):int(hp_task['stim_end']/hp['dt']), :, :], axis=0)
#         I_prev_stim[:,:,input_period] = torch.unsqueeze(stim, -1)
        I_prev_stim[input_period,:,:] = torch.unsqueeze(stim.cpu(), 0)

    # an additional input indicating the choice on the previous trial
#     I_prev_choice = torch.zeros([batch_size, 2, n_steps]).to(device)
    I_prev_choice = torch.zeros([n_steps, batch_size, 2])
    if prev_choice!=None:    # because prev_choice is a numpy array
#         I_prev_choice[:,:,input_period] = torch.unsqueeze(prev_choice, -1)
        I_prev_choice[input_period,:,:] = torch.unsqueeze(prev_choice.cpu(), 0)
        
#         for t in input_period:
#             I_prev_choice[:,:,t] = prev_choice
            
    return I_prev_rew, I_prev_stim, I_prev_choice


def plot_perf(perf_list, title='performance', xlabel='trial', ylabel='performance'):
    fig = plt.figure(figsize=[10,3])
    fig.patch.set_facecolor('white')
    plt.style.use('classic')
    plt.title(title)
    plt.plot(perf_list)
    plt.ylim([-0.1,1.1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
#     print(perf_list)

def plot_y_yhat(y, yhat):
    """ plot the output of the network 
        y: timestep * batch * feature
    """
    
    for k in random.sample(range(y.shape[1]), 1):    # randomly pick a sample to plot
        fig = plt.figure(figsize=[10,4])
        fig.patch.set_facecolor('white')
        plt.style.use('classic')
        plt.title('sample {} in a batch'.format(k))
        for i in range(y.shape[-1]):
            plt.plot(y.detach().cpu().numpy()[:, k, i], label=i)
        plt.legend()
        plt.xlabel('timestep')
        plt.ylabel('y')
        plt.show()
        
        fig = plt.figure(figsize=[10,4])
        fig.patch.set_facecolor('white')
        plt.style.use('classic')
        plt.title('sample {} in a batch, target'.format(k))
        for i in range(yhat.shape[-1]):
            plt.plot(yhat.detach().numpy()[:, k, i], label=i)
        plt.legend()
        plt.xlabel('timestep')
        plt.ylabel('yhat')
        plt.show()
        
#         fig = plt.figure(figsize=[10,4])
#         fig.patch.set_facecolor('white')
#         fig.style.use('classic')
#         fig.title('sample {} in a batch'.format(tr))
#         for i in range(y.shape[1]):
#             fig.plot(y.detach().numpy()[tr, i, :], label=i)
#         fig.legend()
#         fig.add_xlabel('timestep')
#         fig.add_ylabel('y')
#         fig.show()
#         return fig

        
        
        
        
def test_frozen_weights(model, n_trials_test, switch_every_test, init_rule, hp, hp_task, loss_fnc, task, delay_var=0, reset_network=False, give_prev_stim=True, give_prev_choice=True, give_prev_rew=True, plot=False, tr_type_after_switch='n/a', tr_type_otherwise='n/a', toprint=True, noiseless=False):
    
    
    """ delay_var: variability in the delay duration """
    
    if task=='salzman':
        rule_list_2 = ['cxt1', 'cxt2']
    elif task=='cxtdm':
        rule_list_2 = ['color', 'motion']
    
    current_rule = init_rule
    if init_rule not in rule_list_2:
        raise ValueError('initial rule not in rule list!')
        
        
    start = time.time()
    model.eval()
    device = model.rnn.w_rec.device
#     model.rnn.batch_size = 10    # change batch size
    
    perf_list_test = []
    perf_rule_list_test = []
    loss_list = []
    activity_list = []
    rule_list = []
    stim_list = []
    resp_list = []
    i_me_list = []
    
    hp_copy = copy.deepcopy(hp)    # to prevent changing hp
    hp_copy['network_noise'] = hp['network_noise']    # set the noise in testing
    if noiseless==True:
        hp_copy['network_noise'] = 0    # make the network noiseless
        print('test with no network noise\n')
    if toprint==True:
        print('network noise in hp: {}'.format(hp['network_noise']), flush=True)
        print('network noise in hp_copy: {}'.format(hp_copy['network_noise']), flush=True)
    
    
    
    # initialize the network at a fixed point (not used now)
#     rdm_probes = np.random.uniform(0, 1, [model.rnn.batch_size, model.rnn.total_n_neurons])
#     rdm_probes = torch.Tensor(rdm_probes).to(device)
#     print('rdm_probes shape: {}'.format(rdm_probes.shape))
#     rnn_activity = probe_net(model=model, probes=rdm_probes, hp_task=hp_task, hp=hp_copy, n_timesteps=1000, noise=0)
#     steady_state = torch.Tensor(rnn_activity)[:,:,-1].to(device)
#     print('computed steady state={}. shape={}. mean={}'.format(steady_state, steady_state.shape, torch.mean(steady_state)))


    
    with torch.no_grad():
        for tr in range(n_trials_test):
            if tr==0:
                last_rew = None
                h_init = None    # initial activity is 0
                i_me_init = None
#                 h_init = steady_state    # initialize at the steady state
                prev_stim = None
                prev_choice = None
            else:
                if reset_network==False:
                    h_init = h_last
                    i_me_init = i_me_last
                else:
                    h_init = None
                    i_me_init = None
                if give_prev_rew==True:
                    last_rew = perf.detach()
                else:
                    last_rew = None
                if give_prev_stim==True:
                    prev_stim = _x.detach()
                else:
                    prev_stim = None
                if give_prev_choice==True:
                    prev_choice = choice.detach()
                else:
                    prev_choice = None
                    
                    
            # implement variable trial duration
            hp_task_var_delay = copy.deepcopy(hp_task)
            hp_task_var_delay['resp_start'] = hp_task['resp_start'] + np.random.uniform(low=-delay_var, high=delay_var)    # adjust this 
            hp_task_var_delay['resp_end'] = hp_task_var_delay['resp_start'] + (hp_task['resp_end']-hp_task['resp_start'])    # resp duration remains the same
            hp_task_var_delay['trial_end'] = hp_task_var_delay['resp_end']      
                    
             # compute the trial history current
            input_period = np.arange(int(hp_task_var_delay['trial_history_start']/hp_copy['dt']), int(hp_task_var_delay['trial_history_end']/hp_copy['dt']))    # input period for the trial history info
            n_steps = int((hp_task_var_delay['trial_end'] - hp_task_var_delay['trial_start'])//hp_copy['dt'])
#             if last_rew is not None:
#                 print('last_rew shape: {}'.format(last_rew.shape))
            I_prev_rew, I_prev_stim, I_prev_choice = compute_trial_history(last_rew=last_rew, prev_stim=prev_stim, 
                                                                           prev_choice=prev_choice, input_period=input_period,
                                                                           batch_size=model.rnn.batch_size, n_steps=n_steps, 
                                                                           input_dim=model.rnn.n['input'], 
                                                                           hp_task=hp_task_var_delay, hp=hp_copy)
            I_prev_rew, I_prev_stim, I_prev_choice = I_prev_rew.to(device), I_prev_stim.to(device), I_prev_choice.to(device)
            trial_history = {'i_prev_rew': I_prev_rew, 'i_prev_choice': I_prev_choice, 'i_prev_stim': I_prev_stim}
#             print(I_prev_rew[:,0,:], I_prev_choice[:,0,:], I_prev_stim[:,0,:])
#             print('I_prev_rew shape: {}'.format(I_prev_rew.shape))
            
            
            # generate data for 1 trial
#             print('current_rule={}'.format(current_rule))
            if task=='salzman':
                _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_fusi(n_trials=model.rnn.batch_size, rule=current_rule, hp=hp_copy, 
                                                                  hp_fusi=hp_task_var_delay)
            elif task=='cxtdm':
                if (tr%switch_every_test==0):    # first trial after switch is incongruent trial
                    _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_siegel(n_trials=model.rnn.batch_size, 
                                                                                 rule=current_rule, hp=hp_copy, 
                                                                                 hp_cxtdm=hp_task_var_delay, trial_type='incongruent')
                else:
                    _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_siegel(n_trials=model.rnn.batch_size, 
                                                                                 rule=current_rule, hp=hp_copy, 
                                                                                 hp_cxtdm=hp_task_var_delay, trial_type='no_constraint')
#                 print('_x shape: {}'.format(_x.shape))
                
                
            rule = task_data['rules']
            stim = task_data['stims']
            _x, _yhat, _yhat_rule = _x.to(device), _yhat.to(device), _yhat_rule.to(device)
            if _x_rule is not None:
                _x_rule.to(device)
            
            # run model forward 1 trial
            out, data = model(input=_x, init={'h': h_init, 'i_me': i_me_init}, trial_history=trial_history)
            rnn_activity = data['record']['hiddens']
            rnn_activity = torch.stack(rnn_activity, dim=0)
            h_last = data['last_states']['hidden']
            i_me_last = data['last_states']['i_me']
#             print('printing data[record] shapes')
#             print(len(data['record']['i_mes']), len(data['record']['hiddens']))
#             print(data['record']['i_mes'][0].shape, data['record']['hiddens'][0].shape)

#             _y, _y_rule, _, _, _, data = model(input=_x, h_init=h_init, i_me_init=i_me_init, 
#                                                I_prev_rew=I_prev_rew, I_prev_stim=I_prev_stim, 
#                                                I_prev_choice=I_prev_choice, yhat=_yhat, 
#                                                yhat_rule=_yhat_rule, hp_task=hp_task_var_delay, hp=hp_copy)
# #             print('_y={}'.format(_y))
# #             print('_y shape {}'.format(_y.shape))
#             h_last = data['h_last']
#             i_me_last = data['i_me'][-1]
#             rnn_activity = data['activity']
            
            # get the performance
            _y = out['out']
            perf, choice_prob, choice = get_perf(y=_y, yhat=_yhat, hp=hp_copy, hp_task=hp_task_var_delay)
            if hp_copy['train_rule']==True:
                _y_rule = out['out_rule']
                perf_rule, _, _ = get_perf(y=_y_rule, yhat=_yhat_rule, hp=hp_copy, hp_task=hp_task_var_delay)
            # accumulate loss
            total_loss = loss_fnc(_y, _yhat)
            if hp_copy['train_rule']==True:
                total_loss += loss_fnc(_y_rule, _yhat_rule)
                
                
            ## plot
            if plot==True and tr%switch_every_test==2:
                fig, axs = plt.subplots(2,3, figsize=[20,5]); plt.style.use('classic'); fig.patch.set_facecolor('white')
                fig.suptitle('Trial {}'.format(tr), fontsize=20)
                axs[0,0].set_title(perf[0])
                for i in range(2):
                    if i==0:
                        color='blue'
                    elif i==1:
                        color='red'
                    axs[0,0].plot(_y[:, 0, i], color=color)
                    axs[0,0].plot(_yhat[:, 0, i], color=color, linestyle='dashed')
#                 axs[0,0].legend(loc="upper right")
                plot_task_epochs(hp_task=hp_task_var_delay, hp=hp_copy, ax=axs[0,0])
                
                axs[0,1].set_title('I_prev_rew')
                for i in range(2):
                    axs[0,1].plot(I_prev_rew[:, 0 ,i], label='channel {}'.format(i))
                axs[0,1].legend(loc="upper right")
                plot_task_epochs(hp_task=hp_task_var_delay, hp=hp_copy, ax=axs[0,1])
                
                axs[0,2].set_title('I_prev_choice')
                for i in range(2):
                    axs[0,2].plot(I_prev_choice[:, 0, i], label='channel {}'.format(i))
                axs[0,2].legend(loc="upper right")
                plot_task_epochs(hp_task=hp_task_var_delay, hp=hp_copy, ax=axs[0,2])
                
                axs[1,0].set_title('I_prev_stim')
                for i in range(5):
                    axs[1,0].plot(I_prev_stim[:, 0, i], label='channel {}'.format(i))
                axs[1,0].legend(loc="upper right")
                plot_task_epochs(hp_task=hp_task_var_delay, hp=hp_copy, ax=axs[1,0])
                
                axs[1,1].set_title('x')
                for i in range(5):
                    axs[1,1].plot(_x[:, 0, i], label='channel {}'.format(i))
                axs[1,1].legend(loc="upper right")
                plot_task_epochs(hp_task=hp_task_var_delay, hp=hp_copy, ax=axs[1,1])
                
                axs[1,2].set_title('unit activity')
                for n in range(model.rnn.total_n_neurons):
                    axs[1,2].plot(rnn_activity[:, 0, n])
#                 axs[1,2].legend(loc="upper right")
                plot_task_epochs(hp_task=hp_task_var_delay, hp=hp_copy, ax=axs[1,2])
                    
                fig.tight_layout()
                plt.show()
            
            
            

                
            # collect stuff
            loss_list.append(total_loss.detach().cpu())
#             perf_list_test.append(torch.mean(perf.float().detach()).cpu())
            perf_list_test.append(perf.float().detach().cpu().numpy())
            if model.rnn.train_rule==True:
#                 perf_rule_list_test.append(torch.mean(perf_rule.float().detach()).cpu())
                perf_rule_list_test.append(perf_rule.float().detach().cpu().numpy())
            else:
                perf_rule_list_test.append(0)
            activity_list.append(rnn_activity.detach().cpu())
            rule_list.append(rule)
            stim_list.append(stim)
            resp_list.append(choice)
            i_me = np.array(torch.stack(data['record']['i_mes']).detach().cpu().numpy())
#             print('in test_frozen_weight, i_me shape: {}'.format(i_me.shape))
            i_me_list.append(i_me)
            

            # concatenate x and yhat across trials
            if tr==0:
                x = _x
                yhat = _yhat
                yhat_rule = _yhat_rule
                y = _y
                y_rule = _y_rule
            else:
                x = torch.cat((x, _x), axis=0)
                yhat = torch.cat((yhat, _yhat), axis=0)
                y = torch.cat((y, _y), axis=0)
                if model.rnn.train_rule==True:
                    yhat_rule = torch.cat((yhat_rule, _yhat_rule), axis=0)
                    y_rule = torch.cat((y_rule, _y_rule), axis=0)
#                     print('_y_rule shape={}, y_rule shape={}'.format(_y_rule.shape, y_rule.shape))
                    
    
            # switch rule if necessary
            if (tr+1)%switch_every_test==0 and (tr+1)!=0:
#                 print('rule_list_2={}\n\ncurrent_rule={}\n\n'.format(rule_list_2, current_rule))
                next_rule = random.choice([r for r in rule_list_2 if r!=current_rule])
                current_rule = next_rule
#                 print('rule switch, new rule={}\n\n'.format(current_rule))
        
                    
                    
                    
                    
        if plot==True:
            perf_list_test_mean = [np.mean(p) for p in perf_list_test]
            perf_rule_list_test_mean = [np.mean(p) for p in perf_rule_list_test]
            plot_perf(perf_list_test_mean, title='test performance')
            print('y shape={}, yhat shape={}'.format(y.shape, yhat.shape))
            plot_y_yhat(y.cpu(), yhat.cpu())
#             print('y shape={}'.format(y.shape))
            if hp_copy['train_rule']==True:
                plot_perf(perf_rule_list_test_mean, title='test performance (rule)', ylabel='performance (rule)')  
                plot_y_yhat(y_rule.cpu(), yhat_rule.cpu())
            
            fig = plt.figure(figsize=[10,3])
            fig.patch.set_facecolor('white')
            plt.style.use('classic')
            for i in range(0,n_trials_test,switch_every_test):
                plt.plot(perf_list_test_mean[i:i+switch_every_test], color='k', alpha=0.2)
            plt.xlabel('trial after switch')  
            plt.ylabel('perf')
            plt.ylim([-0.1, 1.1])
            plt.show()
        
            if model.rnn.train_rule==True:
                fig = plt.figure(figsize=[10,3])
                fig.patch.set_facecolor('white')
                plt.style.use('classic')
                for i in range(0,n_trials_test,switch_every_test):
                    plt.plot(perf_rule_list_test_mean[i:i+switch_every_test], color='k', alpha=0.2)
                plt.xlabel('trial after switch')  
                plt.ylabel('perf (rule)')
                plt.ylim([-0.1, 1.1])
                plt.show()
        
        if toprint==True:
            print('mean test loss={:0.4f}, mean test perf={:0.4f}, mean test perf rule={:0.4f}, max test perf={:0.4f}, time={:0.2f}s\n'
                  .format(np.mean(loss_list), np.mean(perf_list_test), np.mean(perf_rule_list_test), 1-1/switch_every_test, time.time()-start))
            print('switch_every_test={}, n_trials_test={}\n'.format(switch_every_test, n_trials_test))
        
        data = {'y': y, 'yhat': yhat, 'y_rule': y_rule, 'yhat_rule': yhat_rule, 'rnn_activity': activity_list, 
               'rules': rule_list, 'stims': stim_list, 'resps': resp_list, 'perfs': perf_list_test, 'perf_rules': perf_rule_list_test,
               'i_mes': i_me_list}
        
    return np.mean(perf_list_test), np.mean(perf_rule_list_test), np.mean(loss_list), data




# def test_changing_weights(model, n_trials_test, switch_every_test, init_rule, hp, hp_task, loss_fnc, optim,
#                           reset_network=False, give_prev_stim=True, give_prev_choice=True, give_prev_rew=True, variable_delay=True):
    
#     start = time.time()
#     perf_list_test = []
#     perf_rule_list_test = []
#     perf_list_test_each_sample = []    # do not average across each batch
#     perf_rule_list_test_each_sample = []
#     loss_list_test = []
#     activity_list = []
#     rule_list = []
#     stim_list = []
#     perf_list_test_each_sample = []
#     perf_rule_list_test_each_sample = []
#     for tr in range(n_trials_test):
#         if tr==0:
#             last_rew = None
#             h_init = None
#             prev_stim = None
#             prev_choice = None
#         else:
#             if reset_network==False:
#                 h_init = h_last.detach()
#             else:
#                 h_init = None
#             if give_prev_rew==True:
#                 last_rew = perf.detach()
#             else:
#                 last_rew = None
#             if give_prev_stim==True:
#                 prev_stim = x.detach()
#             else:
#                 prev_stim = None
#             if give_prev_choice==True:
#                 prev_choice = choice.detach()
#             else:
#                 prev_choice = None
        
#         # generate data for 1 trial
#         hp_task_var_delay = copy.deepcopy(hp_task)
#         if variable_delay==True:   
#             hp_task_var_delay['resp_start'] = hp_task['resp_start'] + np.random.uniform(low=hp_task['stim_end']-hp_task['resp_start'], 
#                                                                                         high=hp_task['resp_end']-hp_task['resp_start'])
        
#         _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_fusi(n_trials=model.rnn.batch_size, rule=init_rule, hp=hp, 
#                                                           hp_fusi=hp_task_var_delay)
#         rule = task_data['rules']
#         stim = task_data['stims']
#         # run model forward 1 trial
#         _y, _y_rule, _, _, _, data = model(input=_x, h_init=h_init, last_rew=last_rew, prev_stim=prev_stim, 
#                                            prev_choice=prev_choice, yhat=_yhat, yhat_rule=_yhat_rule, 
#                                            hp_task=hp_task_var_delay, hp=hp)
#         h_last = data['h_last']
        
#         rnn_activity = data['activity']

#         # get the performance
#         perf, choice_prob, choice = get_perf(y=_y, yhat=_yhat, hp=hp, hp_task=hp_task_var_delay)
#         if hp['train_rule']==True:
#             perf_rule, _, _ = get_perf(y=_y_rule, yhat=_yhat_rule, hp=hp, hp_task=hp_task_var_delay)



#         total_loss = loss_fnc(_y, _yhat)
#         if hp['train_rule']==True:
#             total_loss += loss_fnc(_y_rule, _yhat_rule)
#         optim.zero_grad()           # clear gradients for this training step
#         total_loss.backward(retain_graph=True)           # backpropagation, compute gradients
#         optim.step()                # apply gradients

        
#         # collect stuff
#         loss_list_test.append(total_loss.detach())
#         perf_list_test.append(torch.mean(perf.float().detach()))
#         perf_list_test_each_sample.append(perf.float().detach())
#         if model.rnn.train_rule==True:
#             perf_rule_list_test.append(torch.mean(perf_rule.float().detach()))
#             perf_list_test_each_sample.append(perf_rule.float().detach())
#         else:
#             perf_rule_list_test.append(0)
#         activity_list.append(rnn_activity.detach())
#         rule_list.append(rule)
#         stim_list.append(stim)


#         # concatenate x and yhat across trials
#         if tr==0:
#             x = _x
#             yhat = _yhat
#             yhat_rule = _yhat_rule
#             y = _y
#             y_rule = _y_rule
#         else:
#             x = torch.cat((x, _x), axis=0)
#             yhat = torch.cat((yhat, _yhat), axis=0)
#             y = torch.cat((y, _y), axis=0)
#             if model.rnn.train_rule==True:
#                 yhat_rule = torch.cat((yhat_rule, _yhat_rule), axis=0)
#                 y_rule = torch.cat((y_rule, _y_rule), axis=0)

#         # switch rule if necessary
#         if tr%switch_every_test==0 and tr!=0:
#             if init_rule=='cxt1':
#                 init_rule = 'cxt2'
#             elif init_rule=='cxt2':
#                 init_rule = 'cxt1'
#             if tr==n_trials_test - switch_every_test:
#                 model_state_dict_other = copy.deepcopy(model.state_dict())    # the model that performs the other rule
                
    
    
#     # plotting
#     plot_perf(perf_list_test, title='test performance')
#     plot_y_yhat(y, yhat)
# #         print('y shape={}'.format(y.shape))
#     if hp['train_rule']==True:
#         plot_perf(perf_rule_list_test, title='test performance (rule)', ylabel='performance (rule)')  
#         plot_y_yhat(y_rule, yhat_rule)

#     fig = plt.figure(figsize=[15,5])
#     for i in range(0,n_trials_test,switch_every_test):
#         plt.plot(perf_list_test[i:i+switch_every_test], color='k', alpha=0.2)
#     plt.xlabel('trial after switch')  
#     plt.ylabel('perf')
#     plt.ylim([-0.1, 1.1])
#     plt.show()

#     if model.rnn.train_rule==True:
#         fig = plt.figure(figsize=[15,5])
#         for i in range(0,n_trials_test,switch_every_test):
#             plt.plot(perf_rule_list_test[i:i+switch_every_test], color='k', alpha=0.2)
#         plt.xlabel('trial after switch')  
#         plt.ylabel('perf (rule)')
#         plt.ylim([-0.1, 1.1])
#         plt.show()


#     print('mean test loss={:0.4f}, mean test perf={:0.4f}, mean test perf rule={:0.4f}, time={:0.2f}s'
#       .format(np.mean(loss_list_test), np.mean(perf_list_test), np.mean(perf_rule_list_test), time.time()-start), flush=True)

#     data = {'y_rule': y_rule, 'yhat_rule': yhat_rule, 'rnn_activity': torch.stack(activity_list, dim=0), 'rules': rule_list, 'stims': stim_list,
#             'model_state_dict': model.state_dict(), 'model_state_dict_other': model_state_dict_other, 'loss_list': loss_list_test, 
#             'perf_list_test_each_sample': perf_list_test_each_sample, 'perf_rule_list_test_each_sample': perf_rule_list_test_each_sample}

#     return np.mean(perf_list_test), np.mean(perf_rule_list_test), np.mean(loss_list_test), data



def plot_task_epochs(hp_task, hp, ax):
    """ Plot the different epochs of the task """
    if hp['explicit_rule']==True:
        ax.axvspan(int(hp_task['rule_start']/hp['dt']), int(hp_task['rule_end']/hp['dt']), color='k', alpha=0.1)
    ax.axvspan(int(hp_task['stim_start']/hp['dt']), int(hp_task['stim_end']/hp['dt'])-1, color='k', alpha=0.1)    # plt.axvspan(a, b) will cover [a, b] whereas we want [a, b-1]
    ax.axvspan(int(hp_task['resp_start']/hp['dt']), int(hp_task['resp_end']/hp['dt'])-1, color='k', alpha=0.1)
    ax.axvspan(int(hp_task['trial_history_start']/hp['dt']), int(hp_task['trial_history_end']/hp['dt'])-1, color='k', alpha=0.1)
    
    
def load_model(name, simple=False):
    if torch.cuda.is_available():
        saved_data = torch.load('saved_models/{}'.format(name))
    else:
        saved_data = torch.load('saved_models/{}'.format(name), map_location=torch.device('cpu'))
    hp = saved_data['hp']
    if 'n_branches' not in hp.keys():
        print('yes')
        hp['n_branches'] = 1    # some older model does not have the option to change the number of dendritic branches per neuron
    hp_task = saved_data['hp_task']
    
    print('n_branches = {}\n'.format(hp['n_branches']))
    
    if simple==True:
        model = SimpleNet_readoutSR(hp)
    else:
        model = Net_readoutSR(hp)
    model.load_state_dict(saved_data['model_state_dict'], strict=False)
    optimizer = torch.optim.Adam    # just load Adam. 
    optim = optimizer(params=model.parameters(), lr=hp['learning_rate'])    # instantiate the optimizer here
    optim.load_state_dict(saved_data['optim_state_dict'])
    
    return model, hp, hp_task, optim, saved_data


def load_model_v2(path_to_file, model_name, simple=False, plot=False, toprint=True):
    start = time.time()
    if torch.cuda.is_available():
        saved_data = torch.load(path_to_file)
    else:
        saved_data = torch.load(path_to_file, map_location=torch.device('cpu'))
#     print('load model takes {}s'.format(time.time()-start))
    hp = saved_data['hp']
    # to fill in the more recent hps
    hp_default, _, _ = get_default_hp()
    for key in list(hp_default.keys()):
        if key not in list(hp.keys()):
            hp[key] = hp_default[key]
#     if 'n_branches' not in hp.keys():
# #         print('yes')
#         hp['n_branches'] = 1    # some older model does not have the option to change the number of dendritic branches per neuron
    hp_task = saved_data['hp_task']
    
#     print('n_branches = {}\n'.format(hp['n_branches']))
    
    if simple==True:
        model = SimpleNet_readoutSR(hp)
    else:
        model = Net_readoutSR_working(hp)
    model.load_state_dict(saved_data['model_state_dict'], strict=False)
#     model = saved_data['model'] 
    optimizer = torch.optim.Adam    # just load Adam. 
    optim = optimizer(params=model.parameters(), lr=hp['learning_rate'])    # instantiate the optimizer here
    optim.load_state_dict(saved_data['optim_state_dict'])
#     print('load optim takes {}s'.format(time.time()-start))
    
    # in case model does not have these (trained earlier than the modification)
    if 'dend_nonlinearity' not in hp.keys():
        hp['dend_nonlinearity'] = 'old'
        model.rnn.dend_nonlinearity = 'old'
    if 'divisive_dend_inh' not in hp.keys():
        hp['divisive_dend_inh'] = False
        model.rnn.divisive_dend_inh = False
    if 'divisive_dend_ei' not in hp.keys():
        hp['divisive_dend_ei'] = False
        model.rnn.divisive_dend_ei = False


    model.rnn.dend_idx_sr = [i for i in model.rnn.dend_idx if i in model.rnn.sr_idx] 
    model.rnn.dend_idx_pfc = [i for i in model.rnn.dend_idx if i in model.rnn.pfc_idx]
#     print('make indices takes {}s'.format(time.time()-start))

    if toprint==True:
        print(hp)
        print('\n')
        print(hp_task)
        print('\n')
        print(saved_data['optimizer'])
    
    if plot==True:
        # plot the trainig and testing performance
        fig, ax = plt.subplots(2,1,figsize=[7,8])
        for i in range(2):
            ax[i].tick_params(axis='both', which='major', labelsize=10)
        fig.patch.set_facecolor('white')
        plt.style.use('classic')
        title = ax[0].set_title("\n".join(wrap(model_name, 60)))
        title.set_y(1.05)
        ax[0].plot(saved_data['perf_list'], label='training perf')
        ax[0].plot(saved_data['perf_rule_list'], label='training perf for rule')
        ax[0].plot(saved_data['loss_list'][1:], label='training loss')
        ax[0].legend(bbox_to_anchor=(1.1, 0.85))
        ax[0].set_xlabel('step (x100)', fontsize=12)
        ax[0].set_ylabel('Loss/Perf', fontsize=12)
#         fig.tight_layout()
#         plt.show()


#         fig = plt.figure(figsize=[7,4])
#         fig.patch.set_facecolor('white')
#         ax[].style.use('classic')
#         ax.title(name) 
        ax[1].plot(saved_data['test_perf_list'], label='testing perf')
        ax[1].plot(saved_data['test_perf_rule_list'], label='testing perf for rule')
        ax[1].plot(saved_data['test_loss_list'], label='testing loss')
        ax[1].set_xlabel('testing #', fontsize=12)
        ax[1].legend(bbox_to_anchor=(1.3, 0.5))
        fig.tight_layout()
        plt.show()
        
#     print('totally takes {}s'.format(time.time()-start))
    
    return model, hp, hp_task, optim, saved_data





def disconnect_pfc_from(model):
    """ Disconnect projections from PFC to other regions """
    
#     model.rnn.w_rec.requires_grad = False
    with torch.no_grad():
        for cg1,cg2 in itertools.product(model.rnn.cell_group_list, model.rnn.cell_group_list):
            if 'pfc' in cg1 and 'pfc' not in cg2:
                model.rnn.w_rec[np.ix_(model.rnn.cg_idx[cg1], model.rnn.cg_idx[cg2])] = 0
            
    
def disconnect_pfc_to(model):
    """ Disconnect projections from other regions to PFC """
    
#     model.rnn.w_rec.requires_grad = False
    with torch.no_grad():
        for cg1,cg2 in itertools.product(model.rnn.cell_group_list, model.rnn.cell_group_list):
            if 'pfc' in cg2 and 'pfc' not in cg1:
                model.rnn.w_rec[np.ix_(model.rnn.cg_idx[cg1], model.rnn.cg_idx[cg2])] = 0

            
def gen_rdm_probes(n_probes=1000, n_dim=100, min=0, max=5):
    """ Generate random probes. Each element uniformly distributed between min and max.
        Return a n_probes*n_dim array 
    """
    
    probes = np.random.uniform(min, max, [n_probes, n_dim])
    
    return probes


def probe_net(model, probes, hp_task, hp, n_timesteps, noise=0):
    """ Probe the dynamical landscape of the network by starting at some location
        in the phase space and letting the network freely evolve
    """
    
    hp_copy = copy.deepcopy(hp)    # such that hp is not overwritten. use this hp for all the subsequent computations within this function
    hp_copy['network_noise'] = noise
    n_probes = probes.shape[0]
    model.rnn.batch_size = n_probes
    model.eval()
    device = model.rnn.w_rec.device
    print('model device: {}'.format(device))
#     disconnect_pfc_from(model)
#     disconnect_pfc_to(model)

    with torch.no_grad():
        h_init = torch.Tensor(probes).to(device)     # set the initial state for the entire network
        input_period = np.arange(0,1)    # input period for the trial history info (no input here)
#         n_steps = (hp_task['trial_end'] - hp_task['trial_start'])//hp['dt']
    
        I_prev_rew, I_prev_stim, I_prev_choice = compute_trial_history(model=model, last_rew=None, prev_stim=None, prev_choice=None, input_period=input_period, batch_size=model.rnn.batch_size, n_steps=n_timesteps,
                                                                       input_dim=model.rnn.n['input'], hp_task=hp_task, hp=hp_copy)    # trial history input is all 0 here
        
        print('h_init device: {}'.format(h_init.device))
        empty_input = torch.zeros(n_probes, model.rnn.n['input'], n_timesteps)    # no external input during the probing
        empty_input = empty_input.to(device)
        y_probe, y_rule_probe, _, _, _, data = model(input=empty_input, h_init=h_init, I_prev_rew=I_prev_rew, I_prev_stim=I_prev_stim, I_prev_choice=I_prev_choice, 
                                                     hp_task=hp_task, hp=hp_copy)
        
        rnn_activity = data['activity'].detach().cpu().numpy()
        total_input = data['total_inp'].detach().cpu().numpy()
        
        return rnn_activity, total_input

    

def plot_spd_trajectory(activity):
    """ Plot the speed of neural trajectory 
    
        activity - n_trials*n_neurons*n_timesteps
    """
    n_probes = activity.shape[0]
    n_timesteps = activity.shape[-1]
    speed = np.zeros([n_probes, n_timesteps])
    
    for t in range(1, n_timesteps):
        speed[:, t] = np.linalg.norm(activity[:,:,t] - activity[:,:,t-1], axis=1)

    fig = plt.figure(figsize=[7,4])
    fig.patch.set_facecolor('white')
    plt.style.use('classic')
    plt.title('Speed of neural trajectories')
    plt.xlabel('Timestep')
    plt.ylabel('Speed(a.u.)')
#     plot_task_epochs(hp_task=hp_task, hp=hp)
    for i in range(0,n_probes,10):
        plt.plot(speed[i,:], alpha=0.2, color='k')
    plt.plot(np.mean(speed, axis=0), color='green')
    fig.tight_layout()
    plt.show()
    
    return speed
    
    
    
def visualize_stable_states(activity, method='pca'):
    """ Visualize the stable states of the network using dim reduction """
    
    stable_states = np.mean(activity[:,:,-5:], axis=2)
    stable_states[np.where(np.isnan(stable_states))] = 1e6
    
    if method=='mds':
        dim_reduction = MDS(n_components=2)
    elif method=='pca':
        dim_reduction = PCA(n_components=2).fit(stable_states)
        print('explained variance %: {}'.format(dim_reduction.explained_variance_ratio_), flush=True)
        
    stable_states_lowD = dim_reduction.fit_transform(stable_states.astype(np.float64))
    stable_states_lowD += 0.0*np.random.normal(size=stable_states_lowD.shape)    # add some jiggles to visualize better
    
    ## plotting
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    plt.style.use('classic')
    plt.title('Low-D visualization of all steady states')
    n_states = stable_states_lowD.shape[0]
    # resting states
    plt.scatter(stable_states_lowD[:,0], stable_states_lowD[:,1], s=5, color='k', marker='.', alpha=1)
    plt.show()
    
    return stable_states, stable_states_lowD, dim_reduction


def concat_trials(rnn_activity):
    """ Input: n_trials * batch_size * n_neurons * n_timesteps(within a trial)
        Output: n_batches * n_neurons * n_timesteps_total
        
        Update 2/28/2022:
        Input: n_trials * n_timesteps(within a trial) * batch_size * n_neurons * 
        Output: n_timesteps_total * n_batches * n_neurons
    """
    
    for tr in range(rnn_activity.shape[0]):
        if tr==0:
            rnn_activity_concat = rnn_activity[tr,:,:,:]
        else:
            rnn_activity_concat = torch.cat((rnn_activity_concat, rnn_activity[tr,:,:,:]), dim=0)
            
    return rnn_activity_concat


def plot_weights_btw_pops():
    """ Plot the weights between every cell groups, as well as the entire connectivity matrix """
    
    full_conn = model.rnn.effective_weight(w=model.rnn.w_rec, mask=model.rnn.mask, w_fix=model.rnn.w_fix).detach().numpy()

    fig = plt.figure()
    plt.title('W_rec')
    full_conn[np.where(full_conn==0)] = np.nan
    sns.heatmap(full_conn, square=True, center=0)
    plt.xlabel('To')
    plt.ylabel('From')
    plt.show()

    fig = plt.figure()
    plt.title('W_rec distribution')
    plt.hist(full_conn.flatten())
    plt.show()    

    for (cg1, cg2) in itertools.product(model.rnn.cell_group_list, model.rnn.cell_group_list):
        if model.rnn.is_connected(cg2, cg1) is False:    # if cg1 is not connected to cg2
            continue 
        submatrix = full_conn[np.ix_(model.rnn.cg_idx[cg1], model.rnn.cg_idx[cg2])]

        fig = plt.figure()
        plt.title('Weights from {} to {}'.format(cg1, cg2))
        sns.heatmap(submatrix, square=True, center=0)
        plt.xlabel('To {}'.format(cg2))
        plt.ylabel('From {}'.format(cg1))
        plt.show()

        fig = plt.figure()
        plt.title('Distribution of weights from {} to {}'.format(cg1, cg2))
        plt.hist(submatrix.flatten())
        plt.show()

    w_in_eff = model.rnn.effective_weight(w=model.rnn.w_in, mask=model.rnn.mask_in)
    fig = plt.figure()
    plt.title('Input weight')
    sns.heatmap(w_in_eff.detach().numpy(), square=True, center=0)
    plt.show()

    w_rew_eff = model.rnn.effective_weight(w=model.rnn.w_rew, mask=model.rnn.mask_rew)
    fig = plt.figure()
    plt.title('Input reward weight')
    sns.heatmap(w_rew_eff.detach().numpy(), square=True, center=0)
    plt.show()

    fig = plt.figure()
    plt.title('Readout weight')
    sns.heatmap(model.readout.weight.detach().numpy(), square=True, center=0)

    fig = plt.figure()
    plt.title('Readout weight for rule')
    sns.heatmap(model.readout_rule.weight.detach().numpy(), square=True, center=0)
    plt.show()
    
    
    
def plot_inp_weights(model):
    """ TODO: Some bug with this function """
    fig = plt.figure()
    plt.title('Input weights onto different SR dendrites')
    for i in np.arange(1,5):
        if i==1 or i==2:
            label='color coherence'
        elif i==3 or i==4:
            label='motion coherence'
        print(i)
        plt.plot(model.rnn.w_in_eff.detach().numpy()[i,model.rnn.cg_idx['sr_edend']], label=label)    # TODO: compute w_in_eff 
    plt.legend()
    plt.xlabel('SR Dendrite ID')
    plt.ylabel('Input weights')
    fig.tight_layout()
    plt.show()
    
    
    from sklearn.metrics.pairwise import cosine_similarity
    input_weights = model.rnn.w_in_eff.detach().numpy()[1:5,model.rnn.cg_idx['sr_edend']]
    correlation = input_weights@np.transpose(input_weights)

    fig = plt.figure()
    plt.title('Dot product between input weights for different features')
    sns.heatmap(correlation, square=True)
    plt.xlabel('Feature')
    plt.xticks(np.arange(4)+0.5, ['redness', 'greeness', 'leftness', 'rightness'], rotation=45)
    plt.ylabel('Feature')
    plt.yticks(np.arange(4)+0.5, ['redness', 'greeness', 'leftness', 'rightness'], rotation=45)
    plt.show()

    fig = plt.figure()
    plt.title('Cosine similarity between input weights for different features')
    sns.heatmap(cosine_similarity(input_weights, input_weights), square=True)
    plt.xlabel('Feature')
    plt.xticks(np.arange(4)+0.5, ['redness', 'greeness', 'leftness', 'rightness'], rotation=45)
    plt.ylabel('Feature')
    plt.yticks(np.arange(4)+0.5, ['redness', 'greeness', 'leftness', 'rightness'], rotation=45)
    plt.show()


    
    
def display_connectivity(model, plot=False):
    """ show the connectivity of the model """
    
    w_rec_eff = model.rnn.effective_weight(w=model.rnn.w_rec, mask=model.rnn.mask, w_fix=model.rnn.w_fix).detach().cpu().numpy()

    for sender in model.rnn.cell_group_list: 
        for receiver in model.rnn.cell_group_list:
            if model.rnn.is_connected(receiver, sender)==False:
                print('{} is not connected to {}'.format(sender, receiver))
                continue
            elif model.rnn.is_connected(receiver, sender)==True:
                print('{} is connected to {}'.format(sender, receiver))
            # maybe use model.rnn.mask instead
            
            if plot==True:
                fig = plt.figure()
                plt.rc('font', size=30)
                plt.style.use('classic')
                fig.patch.set_facecolor('white')
                plt.title('Weights from {} to {}'.format(sender, receiver))
                sns.heatmap(w_rec_eff[np.ix_(model.rnn.cg_idx[sender], model.rnn.cg_idx[receiver])], square=True, center=0)
                plt.xlabel('To {}'.format(receiver))
                plt.ylabel('From {}'.format(sender))
                plt.show()
                
                

def plot_single_cell(ax, cg, n, var_name, sel, rnn_activity, plot_info, hp_task, hp):
    """ plot_info = a list of {'name': 'x', 'trials': [1,3,5], 'color': 'blue'}
    """
    
    plt.rc('font', size=12)
    
    ax.set_title('{} {}\ncolored by {}\n{} sel={:0.4f}'.format(cg, n, var_name, var_name, sel))
    for p in plot_info:
        name = p['name']
        trials = p['trials']
        color = p['color']
        ax.plot(np.mean(rnn_activity.detach().cpu().numpy()[trials, :, 0, n], axis=0), alpha=0.3, linewidth=10, color=color, label=name)
        for tr in trials:
#             if tr in trials:
            ax.plot(rnn_activity[tr, :, 0, n], color=color, linewidth=0.1, alpha=1)
    ax.legend()
    plot_task_epochs(hp_task=hp_task, hp=hp, ax=ax)
    xticks = np.arange(0, rnn_activity.shape[1], step=2)
    xticklabels = [hp['dt']*x for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Unit activity (a.u.)')
    

def plot_current(ax, sender, receiver, n, var_name, current_matrix, plot_info, hp_task, hp, model):
    """ plot_info = a list of {'name': 'x', 'trials': [1,3,5], 'color': 'blue'}
        n: index for the receiver unit
    """
    
    plt.rc('font', size=12)
    
    ax.set_title('current from {} to {} {}\ncolored by {}'.format(sender, receiver, n, var_name))
    for p in plot_info:
        name = p['name']
        trials = p['trials']
        color = p['color']
        ax.plot(np.mean(np.sum(current_matrix[trials, :, :, n][:, :, model.rnn.cg_idx[sender]], axis=-1), axis=0), alpha=0.3, linewidth=10, color=color, label=name)    # sum over senders, average over trials
        for tr in trials:
#             if tr in trials:
            ax.plot(np.sum(current_matrix[tr, :, model.rnn.cg_idx[sender], n], axis=0), color=color, linewidth=0.1, alpha=1)
    ax.legend()
    plot_task_epochs(hp_task=hp_task, hp=hp, ax=ax)
    xticks = np.arange(0, current_matrix.shape[1], step=2)
    xticklabels = [hp['dt']*x for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Unit activity (a.u.)')
    
    
    
    
def decoding(variable_name, neural_activity, target, cgs, decoder, model, chance, hp_task, hp, n_cv=5, plot=False):
    """ Decode task variable from neural activity 
        Args:
            - neural_activity: trial*neuron*timestep (update 3-16-2022: trial * timestep * neuron)
    """
    
    start = time.time()
    
    if decoder=='svc':
        decoder = svm.SVC()
    elif decoder=='lda':
        decoder = LinearDiscriminantAnalysis()
    
    # compute decoding accuracy
    test_accuracy = {}
    for cg in cgs:
        print(cg, neural_activity.shape, len(target), decoder, n_cv)
        _test_accuracy = np.array([cross_validate(decoder, X=neural_activity[:, t, model.rnn.cg_idx[cg]], y=target, cv=n_cv)['test_score'] for t in range(neural_activity.shape[1])])
        _test_accuracy = np.transpose(_test_accuracy)    # number of folds * number of timesteps
        print('_test_accuracy shape: {}'.format(_test_accuracy.shape))
        test_accuracy[cg] = _test_accuracy
    
    # plot decoding accuracy
    if plot==True:
        fig, ax = plt.subplots(figsize=[10,5])
        plt.style.use('classic')
        fig.patch.set_facecolor('white')
        
        ax.set_title('{} decoding'.format(variable_name))
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Decoding accuracy')
        ax.set_ylim([-0.1, 1.1])
        xticks = np.arange(0,neural_activity.shape[-1], step=2)
        xticklabels = [hp['dt']*x for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.axhline(y=chance, linestyle='--', color='k')
        plot_task_epochs(hp_task=hp_task, hp=hp, ax=ax)

        for cg in cgs:
#             print(test_accuracy[cg].shape)
            x = np.arange(test_accuracy[cg].shape[-1])
            y = np.mean(test_accuracy[cg], axis=0)
            y_error = stats.sem(test_accuracy[cg], axis=0)
            if 'sr' in cg:
                plt.plot(x, y, linestyle='-', label=cg)
            elif 'pfc' in cg:
                plt.plot(x, y, linestyle='-', label=cg)
            plt.fill_between(x, y-y_error, y+y_error, color='k', alpha=0.1)

        ax.legend(prop={'size': 12})
#         plt.rc('font', size=15)
        fig.tight_layout()
        plt.show() 

    print('Elapsed time: {}s'.format(time.time()-start))
          
    return test_accuracy, fig




def generate_neural_data_test(model, n_trials_test, switch_every_test, hp_test, hp_task_test, batch_size=1, to_plot=False, concat_activity=False, compute_current=False, compute_ime=False):
    """ generate some neural data for testing """
    
    model.rnn.batch_size = batch_size    # set batch size to 1 for testing

#     model.rnn.prev_rew_mag = 1
#     model.rnn.prev_choice_mag = 1
#     model.rnn.prev_stim_mag = 1
    
    _, _, _, test_data = test_frozen_weights(model=model, n_trials_test=n_trials_test, switch_every_test=switch_every_test, 
                                             init_rule=random.choice(['color', 'motion']), hp=hp_test, task='cxtdm',
                                             loss_fnc=nn.MSELoss(), hp_task=hp_task_test,
                                             delay_var=0, 
                                             give_prev_choice=False, give_prev_stim=False, give_prev_rew=True, plot=to_plot)




    rnn_activity = torch.stack(test_data['rnn_activity'], dim=0)    # n_trials * seq_len * batch * neuron
#     print('rnn_activity shape right after test_frozen_weights: {}'.format(rnn_activity.shape))
    rnn_activity = rnn_activity[:,:,0,:].unsqueeze(2)    # take the 1st sample in the batch
#     print('rnn_activity shape: {}'.format(rnn_activity.shape))


    # exclude the first few trials
    startfrom_tr = 20
    trial_duration = (hp_task_test['trial_end'] - hp_task_test['trial_start'])/hp_test['dt']
    startfrom_ts = int(startfrom_tr * trial_duration)    
    rnn_activity = rnn_activity[startfrom_tr:,:,:,:]    # get rid of the initial transient
    i_mes = test_data['i_mes']
#     print(i_mes)
    i_mes = np.stack(i_mes, axis=0)
#     print(i_mes.shape)
    i_mes = i_mes[startfrom_tr:,:,:,:]    # get rid of the initial transient
    for key in ['rules', 'stims', 'resps', 'perfs', 'perf_rules']:
        test_data[key] = test_data[key][startfrom_tr:]    # get rid of the initial transient


    # concatenate activity across trials
    start = time.time()
    if concat_activity==True:
        rnn_activity_concat = concat_trials(rnn_activity)
    else:
        rnn_activity_concat = 'NA'
#     print('concat takes {} seconds'.format(time.time()-start))
#     print('shape of rnn_activity_concat: {}'.format(rnn_activity_concat.shape))
#     print('mean of rnn_activity={}'.format(torch.mean(rnn_activity)))




    # compute current
    if compute_current==True:
    #     rnn_activity_moved = torch.movedim(rnn_activity.squeeze(1), 1, 2)    # n_trials * n_ts * n_neurons
        rnn_activity_moved = rnn_activity.squeeze(2)    # when rnn_activity is n_trials *n_timesteps * n_batches * n_neurons
    #     print('rnn_activity shape: {}'.format(rnn_activity.shape))

        n_neurons = rnn_activity.shape[-1]
        n_timesteps = rnn_activity.shape[1]
        current_matrix = torch.zeros([rnn_activity.shape[0], n_timesteps, n_neurons, n_neurons])    # trial x time x neuron x neuron
        w_rec_eff = model.rnn.effective_weight(w=model.rnn.w_rec, mask=model.rnn.mask, w_fix=model.rnn.w_fix)

        for n_sender in range(n_neurons):
            for n_receiver in range(n_neurons):
    #             print('rnn_activity_moved shape: {}'.format(rnn_activity_moved.shape))
    #             print('current_matrix shape: {}'.format(current_matrix.shape))
                current_matrix[:, :, n_sender, n_receiver] = rnn_activity_moved[:,:,n_sender] * w_rec_eff[n_sender, n_receiver]

    #     current_matrix = torch.movedim(current_matrix, (-2,-1), (-3,-2))    # n_trials*n_neurons(send)*n_neurons(receive)*n_timesteps

        current_matrix = current_matrix.detach().cpu().numpy()

    #     print('current_matrix shape: {}'.format(current_matrix.shape))
    else:
        current_matrix = 'NA'




    # plot outputs
    
    if to_plot==True:
        print(rnn_activity_concat.shape)
#         _rnn_activity_concat = torch.moveaxis(rnn_activity_concat, 1, 2)
        _rnn_activity_concat = rnn_activity_concat
        w_out_eff = (model.rnn.w_out*model.mask_out).detach().cpu().numpy()
        y = _rnn_activity_concat@w_out_eff

        fig = plt.figure()
        plt.rc('font', size=12)
        plt.title('output units')
        fig.patch.set_facecolor('white')
        plt.style.use('classic')
        plt.plot(y[:, 0, 0], color='red', label='left')    # take the 1st sample in a batch
        plt.plot(y[:, 0, 1], color='blue', label='right')
        plt.plot(y[:, 0, 2], color='green', label='fixation')
        plt.plot(test_data['yhat'][startfrom_ts:, 0, 0], color='red', linestyle='dashed')
        plt.plot(test_data['yhat'][startfrom_ts:, 0, 1], color='blue', linestyle='dashed')
        plt.plot(test_data['yhat'][startfrom_ts:, 0, 2], color='green', linestyle='dashed')
        plt.xlim([150,250])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Timestep')
        plt.ylabel('Unit activity')
        plt.legend()
        plt.show()

        w_out_rule = (model.rnn.w_out_rule*model.mask_out_rule).detach().numpy()
        y_rule = _rnn_activity_concat@w_out_rule
        fig = plt.figure()
        plt.title('rule outputs')
        fig.patch.set_facecolor('white')
        plt.style.use('classic')
        plt.plot(y_rule[:, 0, 0], color='red', label='rule 1')    # take the 1st sample in a batch
        plt.plot(y_rule[:, 0, 1], color='blue', label='rule 2')
        plt.plot(test_data['yhat_rule'][startfrom_ts:, 0, 0], color='red', linestyle='dashed')
        plt.plot(test_data['yhat_rule'][startfrom_ts:, 0, 1], color='blue', linestyle='dashed')
        # plt.xlim([150,250])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Timestep')
        plt.ylabel('Unit activity')
        plt.legend()
        plt.show()
    
    # to save memory
    del test_data['rnn_activity']
    del test_data['i_mes']
    
    if compute_ime==False:
        return {'rnn_activity': rnn_activity, 'current_matrix': current_matrix, 'test_data': test_data}
    elif compute_ime==True:
        return {'rnn_activity': rnn_activity, 'current_matrix': current_matrix, 'test_data': test_data, 'i_mes': i_mes}




def label_trials(test_data, to_plot=False, to_print=False):
    trial_labels = {}

    # perfs = test_data['perfs']
    perfs = [p[0] for p in test_data['perfs']]    
    # perf_rules = test_data['perf_rules']
    perf_rules = [p[0] for p in test_data['perf_rules']]

    error_trials = [i for i in range(len(perfs)) if perfs[i]==0]
    error_trials_rule = [i for i in range(len(perf_rules)) if perf_rules[i]==0]
    correct_trials = [i for i in range(len(perfs)) if perfs[i]==1]
    correct_trials_rule = [i for i in range(len(perf_rules)) if perf_rules[i]==1]

    stable_trs = [tr for tr in range(len(perfs)) if (tr-1 not in error_trials and tr not in error_trials)]

    
    # label trials by rule
    _rules = test_data['rules']
    rules = [_r[0] for _r in _rules]    # the rules for all the samples in a trial is the same
    rule1_trs = [i for i in range(len(rules)) if rules[i]=='color']
    rule2_trs = [i for i in range(len(rules)) if rules[i]=='motion']
    rule1_trs_stable = [tr for tr in rule1_trs if tr in stable_trs]    
    rule2_trs_stable = [tr for tr in rule2_trs if tr in stable_trs]
    rule1_after_error_trs = [tr for tr in rule1_trs if tr-1 in error_trials]
    rule2_after_error_trs = [tr for tr in rule2_trs if tr-1 in error_trials]
    

    # define switch trials
    switch_trs = [tr for tr in range(1, len(rules)) if rules[tr-1]!=rules[tr]]    # the index of the 1st trial of each rule block (the network does not know that rule has switched)


    # label trials by response

    _resp_list = test_data['resps']
    resp_list = []
    for r in _resp_list:
        if r[0].tolist()==[1,0]:    # take sample 0 from a batch
            resp_list.append('L')
        elif r[0].tolist()==[0,1]:
            resp_list.append('R')

    # extract the mean response state
    left_trs = [i for i in range(len(resp_list)) if resp_list[i]=='L']
    left_trs_stable = [tr for tr in left_trs if tr in stable_trs]    # left trials excluding the trial that the network does rule switch
#     left_states = torch.mean(rnn_activity[left_trs, 0, :, hp_task_test['resp_start']//hp_test['dt']:hp_task_test['resp_end']//hp_test['dt']], axis=-1)    # take sample 0 from the batch
    right_trs = [i for i in range(len(resp_list)) if resp_list[i]=='R']
    right_trs_stable = [tr for tr in right_trs if tr in stable_trs]    # left trials excluding the trial that the network does rule switch
#     right_states = torch.mean(rnn_activity[right_trs, 0, :, hp_task_test['resp_start']//hp_test['dt']:hp_task_test['resp_end']//hp_test['dt']], axis=-1)
    

    # plot the mean response state
#     if to_plot==True:
#         fig = plt.figure()
#         plt.style.use('classic')
#         fig.patch.set_facecolor('white')
#         for i in range(1, left_states.shape[0]):
#             plt.plot(left_states[i, model.rnn.cg_idx['sr_esoma']], color='blue')    # plot all left states
#         for i in range(1, right_states.shape[0]):
#             plt.plot(right_states[i, model.rnn.cg_idx['sr_esoma']], color='red')
#         # plt.legend()
#         plt.title('Response states')
#         plt.xlabel('neuron ID in SR Esoma')
#         plt.ylabel('activity')





    # label trials by color and motion

    allcolors = sorted(list(set([s[0][0] for s in test_data['stims']])))
    allmotions = sorted(list(set([s[0][1] for s in test_data['stims']])))
    trs_stable_color = {}
    trs_stable_motion = {}

    for c in allcolors:
        trs_stable_color[c] = [tr for tr in range(len(test_data['stims'])) if test_data['stims'][tr][0][0]==c and tr in stable_trs]

    for m in allmotions:
        trs_stable_motion[m] = [tr for tr in range(len(test_data['stims'])) if test_data['stims'][tr][0][1]==m and tr in stable_trs]

    
    if to_print==True:
        print('# correct/error trials: {}/{}'.format(len(correct_trials), len(error_trials)))
        print('# correct/error trials for rule: {}/{}'.format(len(correct_trials_rule), len(error_trials_rule)))
        print('# stable trials: {}'.format(len(stable_trs)))
        print('# rule1/2 trials: {}/{}'.format(len(rule1_trs), len(rule2_trs)))
        print('# rule1/2 stable trials: {}/{}'.format(len(rule1_trs_stable), len(rule2_trs_stable)))
        print('# left/right stable trials: {}/{}'.format(len(left_trs_stable), len(right_trs_stable)))
        for key in trs_stable_color.keys():
            print('color coh={}, # trials={}'.format(key, len(trs_stable_color[key])))
        for key in trs_stable_motion.keys():
            print('color motion={}, # trials={}'.format(key, len(trs_stable_motion[key])))



    trial_labels['error_trials'] = error_trials
    trial_labels['correct_trials'] = correct_trials
    trial_labels['error_trials_rule'] = error_trials_rule
    trial_labels['correct_trials_rule'] = correct_trials_rule
    trial_labels['rule1_trs_after_error'] = rule1_after_error_trs
    trial_labels['rule1_trs_stable'] = rule1_trs_stable
    trial_labels['rule2_trs_after_error'] = rule2_after_error_trs
    trial_labels['rule2_trs_stable'] = rule2_trs_stable
    trial_labels['switch_trs'] = switch_trs
    trial_labels['left_trs_stable'] = left_trs_stable
    trial_labels['right_trs_stable'] = right_trs_stable
    trial_labels['trs_stable_color'] = trs_stable_color
    trial_labels['trs_stable_motion'] = trs_stable_motion
    
    return trial_labels



def compute_sel_cxtdm(rnn_activity, hp, hp_task, rule1_trs_stable, rule2_trs_stable, left_trs_stable, right_trs_stable, error_trials, stims):
    """ compute the cell selectivity in the PFC """
    # compute cell selectivity
    
    all_sels = {}
    
    rule_sel = {}    # a dict
    rule_sel_normalized = {}    # a dict
    rule_sel_activity = {}    # rule seletivity using the mean activity during a trial
    rule_sel_normalized_activity = {}
    rule_sel_stim = {}    # rule selectivity using the activity during the stimulus period
    resp_sel = {}    # response selectivity 
    resp_sel_normalized = {}
    resp_sel_wout = {}
    # stim_sel = {}    # stimulus selectivity
    # cat_sel = {}    # category selectivity
    # cat_sel_normalized = {}
    color_sel = {}
    color_sel_normalized = {}
    motion_sel = {}
    motion_sel_normalized = {}
    error_selectivity = {}
    error_selectivity_normalized = {}
    

    trial_history_start_ts = hp_task['trial_history_start']//hp['dt']
    trial_history_end_ts = hp_task['trial_history_end']//hp['dt']
    stim_start_ts = hp_task['stim_start']//hp['dt']
    stim_end_ts = hp_task['stim_end']//hp['dt']
    resp_start_ts = hp_task['resp_start']//hp['dt']
    resp_end_ts = hp_task['resp_end']//hp['dt']
    iti_ts = np.arange(trial_history_end_ts, stim_start_ts)
    delay_ts = np.arange(stim_end_ts, resp_start_ts)

    n_neurons = rnn_activity.shape[-1]

    for n in range(n_neurons):
    #     if n in model.rnn.dend_idx:
    #         continue


    #     # rule selectivity using fixed point activity
    #     act_rule1_fp = attractors[0,n]
    #     act_rule2_fp = attractors[1,n]    # alternatively, use the fixed point activity

    #     # normalized
    #     if act_rule1_fp==0 and act_rule2_fp==0:
    #         rule_sel_normalized[n] = 0
    #     else:
    #         rule_sel_normalized[n] = (act_rule1_fp - act_rule2_fp)/(np.abs(act_rule1_fp) + np.abs(act_rule2_fp))

    #     # unnormalized
    #     rule_sel[n] = (act_rule1_fp - act_rule2_fp)    # such that low firing rate neurons have low rule selectivityrule1_trs_stable


        # rule selectivity using the average activity during a trial
    #     act_rule1 = np.mean(rnn_activity[rule1_trs_stable, 0, n, :].detach().cpu().numpy())
    #     act_rule2 = np.mean(rnn_activity[rule2_trs_stable, 0, n, :].detach().cpu().numpy())
        act_rule1 = np.mean(rnn_activity[rule1_trs_stable, :, 0, n][:, delay_ts].detach().cpu().numpy())    # only look at delay period
        act_rule2 = np.mean(rnn_activity[rule2_trs_stable, :, 0, n][:, delay_ts].detach().cpu().numpy())
        rule_sel_activity[n] = act_rule1 - act_rule2

        # normalized
        if act_rule1==0 and act_rule2==0:
            rule_sel_normalized_activity[n] = 0
        else:
            rule_sel_normalized_activity[n] = (act_rule1 - act_rule2)/(np.abs(act_rule1) + np.abs(act_rule2))

        # rule seletivity using the stimulus period activity
        act_rule1_stim = np.mean(rnn_activity[rule1_trs_stable, stim_start_ts:stim_end_ts, 0, n].detach().cpu().numpy())
        act_rule2_stim = np.mean(rnn_activity[rule2_trs_stable, stim_start_ts:stim_end_ts, 0, n].detach().cpu().numpy())
        rule_sel_stim[n] = act_rule1_stim - act_rule2_stim

        # respose selectivity
        resp_start_ts = hp_task['resp_start']//hp['dt']
        resp_end_ts = hp_task['resp_end']//hp['dt']
        mean_left_state = np.mean(rnn_activity[left_trs_stable, resp_start_ts:resp_end_ts, 0, :].detach().cpu().numpy(), axis=(0, 1))
        mean_right_state = np.mean(rnn_activity[right_trs_stable, resp_start_ts:resp_end_ts, 0, :].detach().cpu().numpy(), axis=(0, 1))
        if mean_left_state[n]==0 and mean_right_state[n]==0:
            resp_sel[n] = 0
            resp_sel_normalized[n] = 0
        else:
            resp_sel_normalized[n] = (mean_left_state[n] - mean_right_state[n])/(np.abs(mean_left_state[n]) + np.abs(mean_right_state[n]))
            resp_sel[n] = mean_left_state[n] - mean_right_state[n]
        
        # TODO: response selectivity based on the readout weight: sel = (w_left - w_right)/(np.abs(w_left) + np.abs(w_right))
        # Or maybe just define it in the analysis code
        



        # stimulus selectivity
        stim_period = np.arange(stim_start_ts, stim_end_ts)
        allcolors = sorted(list(set([s[0][0] for s in stims])))
        allmotions = sorted(list(set([s[0][1] for s in stims])))
        mean_act_color = {}    # the mean activity of each color coherence
        for color in allcolors:
            trs = [tr for tr in range(len(stims)) if stims[tr][0][0]==color]
            mean_act_color[color] = np.mean(rnn_activity.detach().cpu().numpy()[trs, :, 0, n], axis=(0, 1))
        color_sel[n] = np.mean([np.abs(x-y) for x, y in itertools.combinations(mean_act_color.values(), 2)])
        color_sel_normalized[n] = np.mean([np.abs(x-y)/(x+y) if x!=0 and y!=0 else 0 for x, y in itertools.combinations(mean_act_color.values(), 2)])

        mean_act_motion = {}    # the mean activity of each motion coherence
        for motion in allmotions:
            trs = [tr for tr in range(len(stims)) if stims[tr][0][1]==motion]
            mean_act_motion[motion] = np.mean(rnn_activity.detach().cpu().numpy()[trs, :, 0, n], axis=(0, 1))
        motion_sel[n] = np.mean([np.abs(x-y) for x, y in itertools.combinations(mean_act_motion.values(), 2)])
        motion_sel_normalized[n] = np.mean([np.abs(x-y)/(x+y) if x!=0 and y!=0 else 0 for x, y in itertools.combinations(mean_act_motion.values(), 2)])



        # category selectivity
    #     AC_trs_prev_corr = A_trs_prev_corr + C_trs_prev_corr
    #     BD_trs_prev_corr = B_trs_prev_corr + D_trs_prev_corr
    #     mean_act_AC = np.mean(rnn_activity.detach().cpu().numpy()[AC_trs_prev_corr, 0, n, :], axis=(0,-1))
    #     mean_act_BD = np.mean(rnn_activity.detach().cpu().numpy()[BD_trs_prev_corr, 0, n, :], axis=(0,-1))
    #     if mean_act_AC==0 and mean_act_BD==0:
    #         cat_sel[n] = 0
    #         cat_sel_normalized[n] = 0
    #     else:
    #         cat_sel[n] = mean_act_AC - mean_act_BD
    #         cat_sel_normalized[n] = (mean_act_AC-mean_act_BD)/(mean_act_AC+mean_act_BD)


    ## compute error signal    
#     w_rew_eff = model.rnn.effective_weight(w=model.rnn.w_rew, mask=model.rnn.mask_rew).detach().cpu().numpy()
#     error_signal_dict = {n: w_rew_eff[1, n] for n in range(model.rnn.total_n_neurons)}
#     correct_signal_dict = {n: w_rew_eff[0, n] for n in range(model.rnn.total_n_neurons)}

    ## compute error selectivity
    # time_period = np.arange(hp_task_test['trial_start']//hp_test['dt'], hp_task_test['trial_end']//hp_test['dt'])    # time period used for calculating neural activity
    time_period = iti_ts    # only look at ITI
    error_fdbk_trs = [tr+1 for tr in error_trials if tr!=rnn_activity.shape[0]-1]    # 1 trial after the error trial (when the network receives the error feedback)
    correct_fdbk_trs = [tr for tr in range(rnn_activity.shape[0]) if tr not in error_fdbk_trs]

    FR_error = torch.mean(rnn_activity[error_fdbk_trs, :, 0, :], axis=0)    # avg over trials
    meanFR_error = torch.mean(FR_error[time_period, :], axis=0)    # avg over time
    FR_correct = torch.mean(rnn_activity[correct_fdbk_trs, :, 0, :], axis=0)
    meanFR_correct = torch.mean(FR_correct[time_period, :], axis=0)

    for n in range(n_neurons):
        # error_selectivity = (act_error - act_correct)/(act_error + act_correct)
        error_selectivity[n] = (meanFR_error[n] - meanFR_correct[n]).numpy()
        if meanFR_error[n]==0 and meanFR_correct[n]==0:
            error_selectivity_normalized[n] = 0
        else:
            error_selectivity_normalized[n] = (meanFR_error[n] - meanFR_correct[n])/(meanFR_error[n] + meanFR_correct[n]).numpy()

            
    ## sort the selectivity
    rule_sel_sorted = {k: v for k, v in sorted(rule_sel.items(), key=lambda item: item[1])}
    rule_sel_normalized_sorted = {k: v for k, v in sorted(rule_sel_normalized.items(), key=lambda item: item[1])}
    rule_sel_normalized_activity_sorted = {k: v for k, v in sorted(rule_sel_normalized_activity.items(), key=lambda item: item[1])}
    rule_sel_activity_sorted = {k: v for k, v in sorted(rule_sel_activity.items(), key=lambda item: item[1])}
    rule_sel_stim_sorted = {k: v for k, v in sorted(rule_sel_stim.items(), key=lambda item: item[1])}
    resp_sel_sorted = {k: v for k, v in sorted(resp_sel.items(), key=lambda item: item[1])}
    color_sel_sorted = {k: v for k, v in sorted(color_sel.items(), key=lambda item: item[1], reverse=True)}
    motion_sel_sorted = {k: v for k, v in sorted(motion_sel.items(), key=lambda item: item[1], reverse=True)}
    color_sel_normalized_sorted = {k: v for k, v in sorted(color_sel_normalized.items(), key=lambda item: item[1], reverse=True)}
    motion_sel_normalized_sorted = {k: v for k, v in sorted(motion_sel_normalized.items(), key=lambda item: item[1], reverse=True)}
    error_selectivity_sorted = {k: v for k, v in sorted(error_selectivity.items(), key=lambda item: item[1], reverse=True)}
    error_selectivity_normalized_sorted = {k: v for k, v in sorted(error_selectivity_normalized.items(), key=lambda item: item[1], reverse=True)}
    
    
    # summarize into one big dict
    all_sels['rule'] = rule_sel
    all_sels['rule_normalized'] = rule_sel_normalized
    all_sels['rule_activity'] = rule_sel_activity
    all_sels['rule_normalized_activity'] = rule_sel_normalized_activity
    all_sels['resp'] = resp_sel
    all_sels['resp_normalized'] = resp_sel_normalized
    all_sels['color'] = color_sel
    all_sels['color_normalized'] = color_sel_normalized
    all_sels['motion'] = motion_sel
    all_sels['motion_normalized'] = motion_sel_normalized
    all_sels['error'] = error_selectivity
    all_sels['error_normalized'] = error_selectivity_normalized
    
    return all_sels

    
    
def define_subpop_pfc(model, rnn_activity, hp_task, hp, rule_sel, err_sel, rule1_trs_stable, rule2_trs_stable, rule1_after_error_trs, rule2_after_error_trs, rule_threshold=0.5, err_threshold=0.5, toprint=False):
    """ define the subpopulations within PFC """
    
    
    cell_types_func = ['rule1', 'rule2', 'mix_err_rule1', 'mix_err_rule2', 'mix_corr_rule1', 'mix_corr_rule2', 'unclassified']
    cell_types = ['esoma', 'edend', 'pv', 'sst', 'vip']
    subcg_pfc = [x+'_'+y for (x,y) in itertools.product(cell_types_func, cell_types)]

    subcg_pfc_idx = {}
    for subcg in subcg_pfc:
        subcg_pfc_idx[subcg] = []
    
#     err_fdbk_trs = [n for n in range(rnn_activity.shape[0]) if n-1 in error_trials]
#     stable_trs = [n for n in range(rnn_activity.shape[0]) if n not in error_trials and n-1 not in error_trials]

    trial_history_end_ts = hp_task['trial_history_end']//hp['dt']
    stim_start_ts = hp_task['stim_start']//hp['dt']
    iti_ts = np.arange(trial_history_end_ts, stim_start_ts)

    for n in model.rnn.pfc_idx:
        if n in model.rnn.dend_idx:
            continue    # classify dendrites based on soma activity

        if np.abs(rule_sel[n])>rule_threshold and np.abs(err_sel[n])<err_threshold:    # rule neurons
            # rule 1 neurons
            if rule_sel[n]>0:
                for cg in cell_types:
                    if n in model.rnn.cg_idx['pfc_'+cg]:
                        subcg_pfc_idx['rule1_'+cg].append(n)
                        if cg=='esoma':
                            for b in range(model.rnn.n_branches):
                                subcg_pfc_idx['rule1_edend'].append(n + (b+1)*len(model.rnn.cg_idx['pfc_esoma']))
                                
            # rule 2 neurons
            elif rule_sel[n]<=0:
                for cg in cell_types:
                    if n in model.rnn.cg_idx['pfc_'+cg]:
                        subcg_pfc_idx['rule2_'+cg].append(n)
                        if cg=='esoma':
                            for b in range(model.rnn.n_branches):
                                subcg_pfc_idx['rule2_edend'].append(n + (b+1)*len(model.rnn.cg_idx['pfc_esoma']))


        # mix error neurons
        elif err_sel[n]>=err_threshold:
    #     elif np.mean(rnn_activity.cpu().numpy()[err_fdbk_trs, 0, n, :])>np.mean(rnn_activity.cpu().numpy()[stable_trs, 0, n, :]):
            if np.mean(rnn_activity.cpu().numpy()[rule1_after_error_trs, :, 0, n][:, iti_ts])>np.mean(rnn_activity.cpu().numpy()[rule2_after_error_trs, :, 0, n][:, iti_ts]):    # errorxrule 1 
                for cg in cell_types:
                    if n in model.rnn.cg_idx['pfc_'+cg]:
                        subcg_pfc_idx['mix_err_rule1_'+cg].append(n)
                        if cg=='esoma':
                            for b in range(model.rnn.n_branches):
                                subcg_pfc_idx['mix_err_rule1_edend'].append(n + (b+1)*len(model.rnn.cg_idx['pfc_esoma']))
            else:
                for cg in cell_types:
                    if n in model.rnn.cg_idx['pfc_'+cg]:
                        subcg_pfc_idx['mix_err_rule2_'+cg].append(n)
                        if cg=='esoma':
                            for b in range(model.rnn.n_branches):
                                subcg_pfc_idx['mix_err_rule2_edend'].append(n + (b+1)*len(model.rnn.cg_idx['pfc_esoma']))
        # mix correct neurons
        elif err_sel[n]<=-err_threshold:
    #     elif np.mean(rnn_activity.cpu().numpy()[err_fdbk_trs, 0, n, :])<=np.mean(rnn_activity.cpu().numpy()[stable_trs, 0, n, :]):
            if np.mean(rnn_activity.cpu().numpy()[rule1_trs_stable, :, 0, n][:, iti_ts])>np.mean(rnn_activity.cpu().numpy()[rule2_trs_stable, :, 0, n][:, iti_ts]):
                for cg in cell_types:
                    if n in model.rnn.cg_idx['pfc_'+cg]:
                        subcg_pfc_idx['mix_corr_rule1_'+cg].append(n)
                        if cg=='esoma':
                            for b in range(model.rnn.n_branches):
                                subcg_pfc_idx['mix_corr_rule1_edend'].append(n + (b+1)*len(model.rnn.cg_idx['pfc_esoma']))
            else:
                for cg in cell_types:
                    if n in model.rnn.cg_idx['pfc_'+cg]:
                        subcg_pfc_idx['mix_corr_rule2_'+cg].append(n)
                        if cg=='esoma':
                            for b in range(model.rnn.n_branches):
                                subcg_pfc_idx['mix_corr_rule2_edend'].append(n + (b+1)*len(model.rnn.cg_idx['pfc_esoma']))
        else:
            for cg in cell_types:
                    if n in model.rnn.cg_idx['pfc_'+cg]:
                        subcg_pfc_idx['unclassified_'+cg].append(n)
                        if cg=='esoma':
                            for b in range(model.rnn.n_branches):
                                subcg_pfc_idx['unclassified_edend'].append(n + (b+1)*len(model.rnn.cg_idx['pfc_esoma']))
                                

    # show the number of neurons for each subpopulation
    if toprint==True:
        sum = 0
        for subcg in subcg_pfc:
            print(subcg, len(subcg_pfc_idx[subcg]))
            sum += len(subcg_pfc_idx[subcg])
        print(sum)
    
    return subcg_pfc_idx




def define_subpop_sr(model, rnn_activity, hp_task, hp, rule_sel, resp_sel, rule1_trs_stable, rule2_trs_stable, rule_threshold=0, resp_threshold=0, toprint=False):
    """ define subpopulations within SR """

    cell_types_func = ['rule1', 'rule2', 'left', 'right']
    cell_types = ['esoma', 'edend', 'pv', 'sst', 'vip']
    subcg_sr = [x+'_'+y for (x,y) in itertools.product(cell_types_func, cell_types)]

    subcg_sr_idx = {}
    for subcg in subcg_sr:
        subcg_sr_idx[subcg] = []    

    for n in model.rnn.sr_idx:
        # rule 1 neurons
        if rule_sel[n]>rule_threshold:
            for cg in cell_types:
                if n in model.rnn.cg_idx['sr_'+cg]:
                    subcg_sr_idx['rule1_'+cg].append(n)
#                     if cg=='esoma':
#                         for b in range(model.rnn.n_branches):
#                             subcg_pfc_idx['rule1_edend'].append(n + (b+1)*len(model.rnn.cg_idx['pfc_esoma']))
        # rule 2 neurons                        
        if rule_sel[n]<=-rule_threshold:
            for cg in cell_types:
                if n in model.rnn.cg_idx['sr_'+cg]:
                    subcg_sr_idx['rule2_'+cg].append(n)
#                     if cg=='esoma':
#                         for b in range(model.rnn.n_branches):
#                             subcg_pfc_idx['rule1_edend'].append(n + (b+1)*len(model.rnn.cg_idx['pfc_esoma'])) 
        # response left neurons
        if resp_sel[n]>resp_threshold:
            for cg in cell_types:
                if n in model.rnn.cg_idx['sr_'+cg]:
                    subcg_sr_idx['left_'+cg].append(n)
        # response right neurons
        if resp_sel[n]<=-resp_threshold:
            for cg in cell_types:
                if n in model.rnn.cg_idx['sr_'+cg]:
                    subcg_sr_idx['right_'+cg].append(n)
                         

    # show the number of neurons for each subpopulation
    if toprint==True:
        for subcg in sorted(subcg_sr):
            print(subcg, len(subcg_sr_idx[subcg]))
            
    return subcg_sr_idx
    
    
    
    

def plot_conn_subpop(weight, subcg_to_plot_sender, subcg_to_plot_receiver, cg_idx, plot=True):
    """ plot the connectivity with indices sorted according to their subpopulation assignment """
        
    neuron_id_aggr_sender = []    # aggregated neuron id
    for subcg in subcg_to_plot_sender:
        neuron_id_aggr_sender += cg_idx[subcg]
    neuron_id_aggr_receiver = []    # aggregated neuron id
    for subcg in subcg_to_plot_receiver:
        neuron_id_aggr_receiver += cg_idx[subcg]

    fig, ax = plt.subplots(figsize=[10,8])
    fig.patch.set_facecolor('white')
    plt.style.use('classic')
    ax.set_title('Connectivity between different subpopulations\n within the PFC')
    ax = sns.heatmap(weight[np.ix_(neuron_id_aggr_sender, neuron_id_aggr_receiver)], square=True, center=0, cbar_kws={"shrink": .5})

    # print(w_rec_eff)

    vlines = []    # the lines on the heatmap separating different sub-populations
    x = 0
    for subcg in subcg_to_plot_receiver:
        x+=len(cg_idx[subcg])
        vlines.append(x)
    hlines = []    # the lines on the heatmap separating different sub-populations
    y = 0
    for subcg in subcg_to_plot_sender:
        y+=len(cg_idx[subcg])
        hlines.append(y)
    for x in vlines:
        ax.axvline(x=x, color='white', linewidth=2)
    for y in hlines:
        ax.axhline(y=y, color='white', linewidth=2)
    ax.set_xlabel('To')
    ax.set_ylabel('From')

    xticks = []
    xticklabels = []
    x_up = 0
    for subcg in subcg_to_plot_receiver:
        if len(cg_idx[subcg])==0:
            continue
        x_up += len(cg_idx[subcg])
        xt = x_up - len(cg_idx[subcg])//2
        xticks.append(xt)
        xticklabels.append(subcg)
    yticks = []
    yticklabels = []
    y_up = 0
    for subcg in subcg_to_plot_sender:
        if len(cg_idx[subcg])==0:
            continue
        y_up += len(cg_idx[subcg])
        yt = y_up - len(cg_idx[subcg])//2
        yticks.append(yt)
        yticklabels.append(subcg)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation=0)
    fig.tight_layout()
    if plot==True:
        plt.show()

    # print(w_rec_eff)
    for subcg_sender in subcg_to_plot_sender:
        for subcg_receiver in subcg_to_plot_receiver:
            sender_idx = cg_idx[subcg_sender]
            receiver_idx = cg_idx[subcg_receiver]
            print('mean weight from {} to {}: {}'.format(subcg_sender, subcg_receiver, np.mean(weight[np.ix_(sender_idx, receiver_idx)])))
            
            
    return fig, ax



def criteria_explode(data, threshold=1e2):
    """ the criteria for determining if the activity is exploding """

    explode = np.isnan(data).any()==True or (data>=threshold).any()==True  
    
    return explode





class HiddenPrints:
    """ block all the print """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout