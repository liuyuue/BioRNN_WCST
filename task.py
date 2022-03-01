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


def get_default_hp_cxtdm():
    """ Default parameters for the task """
    hp_cxtdm = {
                # trial start time (in ms)
                'trial_start': 0,
                # fixation start time
                'fix_start': 0,
                # fixation end time
                'fix_end': 0,
                # rule input start time
                'rule_start': 0,
                # rule input end time
                'rule_end': 0,
                # stimulus start time
                'stim_start': 300,
                # stimulus end time
                'stim_end': 400,
                # response start time
                'resp_start': 700,
                # response end time
                'resp_end': 800,
                # trial end time
                'trial_end': 800,
                'trial_history_start': 0,
                'trial_history_end': 100
                }
#     hp_cxtdm = {
#                 # trial start time
#                 'trial_start': 0,
#                 # fixation start time
#                 'fix_start': 0,
#                 # fixation end time
#                 'fix_end': 1000,
#                 # rule start time
#                 'rule_start': 0,
#                 # rule end time
#                 'rule_end': 0,
#                 # stimulus start time
#                 'stim_start': 500,
#                 # stimulus end time
#                 'stim_end': 1000,
#                 # response start time
#                 'resp_start': 1500,
#                 # response end time
#                 'resp_end': 2500,
#                 # trial end time
#                 'trial_end': 2500
#                 }
    
    return hp_cxtdm





def make_task_siegel(hp, rule, n_trials, hp_cxtdm, trial_type='no_constraint'):
    """ 
        The task in Siegel et al., Science(2015). Context-dependent decision making with 2 cues signaling
        each rule.
        
        Input: 
        - hp: hyperparameters for the neural network
        - rule: 'color' or 'motion' or 'random'
        - trial_start: start time for the trial (ms)
        - fix_start: start time for fixation (ms)
        - fix_end: end time for fixation (ms)
        - stim_start: start time for stimulus (ms)
        - stim_end: end time for stimulus (ms)
        - resp_start: start time of the response period (ms)
        - resp_end: end time of the response period (ms)
        - trial_end: end time of the trial (ms)
        - explicit_rule: if the rule (context) is explicitly cued
        - train_pfc: if output of pfc needs to encode the rule info
        
        Output:
        - x: sensory input: step*batch*feature
        - yhat: target output: step*batch*feature
                
    """
    dt = hp['dt']
    n_timesteps = int((hp_cxtdm['trial_end'] - hp_cxtdm['trial_start'])/dt)
    n_in = hp['n_input']
    n_in_rule = hp['n_input_rule_cue']
    n_out = hp['n_output']
    n_out_rule  = hp['n_output_rule']
    explicit_rule = hp['explicit_rule']
    train_rule = hp['train_rule']
#     batch_size = hp['batch_size']
    
    # the timing of different task event in unit of timesteps
    trial_start_ts = int(hp_cxtdm['trial_start']/dt)    
    fix_start_ts = int(hp_cxtdm['fix_start']/dt)   
    fix_end_ts = int(hp_cxtdm['fix_end']/dt)  
    rule_start_ts = int(hp_cxtdm['rule_start']/dt)
    rule_end_ts = int(hp_cxtdm['rule_end']/dt)
    stim_start_ts = int(hp_cxtdm['stim_start']/dt)
    stim_end_ts = int(hp_cxtdm['stim_end']/dt)
    resp_start_ts = int(hp_cxtdm['resp_start']/dt)
    resp_end_ts = int(hp_cxtdm['resp_end']/dt)
    trial_end_ts = int(hp_cxtdm['trial_end']/dt)  
            
    # check input dim
    if (explicit_rule==True and n_in!=9) or (explicit_rule==False and n_in!=5):
        raise ValueError('check input dimensionality!')

    # generate coherences
    coh_colors = random.choices([-0.5, -0.25, -0.1, 0.1, 0.25, 0.5], k=n_trials)    # color coherence. positive for red, negative for green
    coh_motions = random.choices([-0.5, -0.25, -0.1, 0.1, 0.25, 0.5], k=n_trials)    # motion coherence. positive for left, negative for right 
    # simpler version
#     coh_colors = random.choices([-0.5, 0.5], k=n_trials)    # color coherence. positive for red(left), negative for green(right)
#     coh_motions = random.choices([-0.5, 0.5], k=n_trials)    # motion coherence. positive for left, negative for right 
    
#     print('coh_color={}, coh_motion={}'.format(coh_colors, coh_motions))
    
    
    # modify the stimuli to make trials congruent or incongruent
    for tr in range(n_trials):
        if trial_type=='incongruent':
            if coh_colors[tr]*coh_motions[tr]>0:
                coh_motions[tr] = -coh_motions[tr]
            else:
                continue
        elif trial_type=='congruent':
            if coh_colors[tr]*coh_motions[tr]<0:
                coh_motions[tr] = -coh_motions[tr]
            else:
                continue
        elif trial_type=='no_constraint':
            continue
        else:
            raise NotImplementedError
    
    
    
    # generate rules for each trial
    if rule=='color':
        rules = random.choices(['color'], k=n_trials)
    elif rule=='motion':
        rules = random.choices(['motion'], k=n_trials)
    elif rule=='random':
        rules = random.choices(['color', 'motion'], k=n_trials)
    else:
        raise ValueError('rule name "{}" does not exist!'.format(rule))
    color_rule_trs = [i for i in range(len(rules)) if rules[i]=='color']
    motion_rule_trs = [i for i in range(len(rules)) if rules[i]=='motion']
    
    rule_cues = np.zeros(n_trials)    # non-zero only for explicit rule condition
    
    if explicit_rule==True:
        color_rule_cue1_trs = random.sample(color_rule_trs,int(len(color_rule_trs)/2))    # there are 2 cues for each rule
        color_rule_cue2_trs = [i for i in color_rule_trs if i not in color_rule_cue1_trs]
        motion_rule_cue1_trs = random.sample(motion_rule_trs,int(len(motion_rule_trs)/2))
        motion_rule_cue2_trs = [i for i in motion_rule_trs if i not in motion_rule_cue1_trs]
        
        rule_cues[color_rule_cue1_trs] = 1
        rule_cues[color_rule_cue2_trs] = 2
        rule_cues[motion_rule_cue1_trs] = 3
        rule_cues[motion_rule_cue2_trs] = 4
    
    left_trs = [(r=='color' and coh_c>0) or (r=='motion' and coh_m>0) 
                             for (r, coh_c, coh_m) in zip(rules, coh_colors, coh_motions)]
    right_trs = [(r=='color' and coh_c<0) or (r=='motion' and coh_m<0) 
                             for (r, coh_c, coh_m) in zip(rules, coh_colors, coh_motions)]
    
    # generate training data
#     yhat = torch.zeros(size=[n_trials, n_out, n_timesteps])
    yhat = torch.zeros(size=[n_timesteps, n_trials, n_out])
#     yhat_rule = torch.zeros(size=[n_trials, n_out_rule, n_timesteps])
    yhat_rule = torch.zeros(size=[n_timesteps, n_trials, n_out_rule])
#     x = torch.zeros(size=[n_trials, n_in, n_timesteps])
    x = torch.zeros(size=[n_timesteps, n_trials, n_in])
#     x_rule = torch.zeros(size=[n_trials, n_in_rule, n_timesteps])
    x_rule = torch.zeros(size=[n_timesteps, n_trials, n_in_rule])
    
    # compute x
#     x[:, 0, fix_start_ts:fix_end_ts] = 1    # the fixation input
    x[fix_start_ts:fix_end_ts, :, 0] = 1
#     x[:, 1, stim_start_ts:stim_end_ts] = torch.transpose((0.5 + torch.Tensor(coh_colors)).repeat(int(stim_end_ts-stim_start_ts), 1), 0, 1)    # redness
    x[stim_start_ts:stim_end_ts, :, 1] = (0.5 + torch.Tensor(coh_colors)).repeat(int(stim_end_ts-stim_start_ts), 1)    # redness
#     x[:, 2, stim_start_ts:stim_end_ts] = torch.transpose((0.5 - torch.Tensor(coh_colors)).repeat(int(stim_end_ts-stim_start_ts), 1), 0, 1)    # greenness
    x[stim_start_ts:stim_end_ts, :, 2] = (0.5 - torch.Tensor(coh_colors)).repeat(int(stim_end_ts-stim_start_ts), 1)    # greenness
#     x[:, 3, stim_start_ts:stim_end_ts] = torch.transpose((0.5 + torch.Tensor(coh_motions)).repeat(int(stim_end_ts-stim_start_ts), 1), 0, 1)   # leftness
    x[stim_start_ts:stim_end_ts, :, 3] = (0.5 + torch.Tensor(coh_motions)).repeat(int(stim_end_ts-stim_start_ts), 1)   # leftness
#     x[:, 4, stim_start_ts:stim_end_ts] = torch.transpose((0.5 - torch.Tensor(coh_motions)).repeat(int(stim_end_ts-stim_start_ts), 1), 0, 1)    # rightness
    x[stim_start_ts:stim_end_ts, :, 4] = (0.5 - torch.Tensor(coh_motions)).repeat(int(stim_end_ts-stim_start_ts), 1)    # rightness
    if explicit_rule==True:
#         x_rule[color_rule_cue1_trs, 0, rule_start_ts:rule_end_ts] = 1
        x_rule[rule_start_ts:rule_end_ts, color_rule_cue1_trs, 0] = 1
#         x_rule[color_rule_cue2_trs, 1, rule_start_ts:rule_end_ts] = 1
        x_rule[rule_start_ts:rule_end_ts, color_rule_cue2_trs, 1] = 1
#         x_rule[motion_rule_cue1_trs, 2, rule_start_ts:rule_end_ts] = 1
        x_rule[rule_start_ts:rule_end_ts, motion_rule_cue1_trs, 2] = 1
#         x_rule[motion_rule_cue2_trs, 3, rule_start_ts:rule_end_ts] = 1
        x_rule[rule_start_ts:rule_end_ts, motion_rule_cue2_trs, 3] = 1
    # add noise to sensory input
#     x[:, 1:5, stim_start_ts:stim_end_ts] += hp['input_noise_perceptual'] * torch.normal(mean=torch.zeros(n_trials, 4, int(stim_end_ts-stim_start_ts)), std=1)
    x[stim_start_ts:stim_end_ts, :, 1:5] += hp['input_noise_perceptual'] * torch.normal(mean=torch.zeros(int(stim_end_ts-stim_start_ts), n_trials, 4), std=1)
    # add noise to rule input
    if explicit_rule==True:
#         x_rule[:, :, rule_start_ts:rule_end_ts] += hp['input_noise_rule'] * torch.normal(mean=torch.zeros(n_trials, 4, int(rule_end_ts-rule_start_ts)), std=1)
        x_rule[rule_start_ts:rule_end_ts, :, :] += hp['input_noise_rule'] * torch.normal(mean=torch.zeros(int(rule_end_ts-rule_start_ts), n_trials, 4), std=1)

    # compute yhat
#     yhat[:, 2, fix_start_ts:fix_end_ts] = 1    # fixation output
    yhat[fix_start_ts:fix_end_ts, :, 2] = 1    # fixation output
#     yhat[left_trs, 0, resp_start_ts:resp_end_ts] = 1
    yhat[resp_start_ts:resp_end_ts, left_trs, 0] = 1
#     yhat[right_trs, 1, resp_start_ts:resp_end_ts] = 1
    yhat[resp_start_ts:resp_end_ts, right_trs, 1] = 1
    
    # compute yhat_rule if needed 
    if train_rule==True:
#         yhat_rule[color_rule_trs, 0, rule_start_ts:trial_end_ts] = 1    # auxillary rule output
        yhat_rule[rule_start_ts:trial_end_ts, color_rule_trs, 0] = 1    # auxillary rule output
#         yhat_rule[motion_rule_trs, 1, rule_start_ts:trial_end_ts] = 1
        yhat_rule[rule_start_ts:trial_end_ts, motion_rule_trs, 1] = 1
    elif train_rule==False:
        yhat_rule = None
    
    stims = [(c,m) for (c,m) in zip(coh_colors, coh_motions)]
    
    task_data = {'rules': rules, 'rule_cues': rule_cues, 'stims': stims, 'coh_colors': coh_colors, 'coh_motions': coh_motions, 'left_trs': left_trs, 'right_trs': right_trs}
    
    return x, x_rule, yhat, yhat_rule, task_data







def get_default_hp_fusi():
    """ Default parameters for the task in Bernadi et al., Cell (2020) """
    
    hp_fusi = {
                # trial start time
                'trial_start': 0,
                # fixation start time
                'fix_start': 0,
                # fixation end time
                'fix_end': 0,
                # rule start time
                'rule_start': 0,
                # rule end time
                'rule_end': 0,
                # stimulus start time
                'stim_start': 500,
                # stimulus end time
                'stim_end': 1000,
                # response start time
                'resp_start': 1500,
                # response end time
                'resp_end': 2500,
                # trial end time
                'trial_end': 2500
                }
    
    return hp_fusi





def make_task_fusi(hp, rule, n_trials, hp_fusi):
    """ 
        The task in Bernadi et al., Cell (2020). 
        
        Input: 
        - hp: hyperparameters for the neural network
        - rule: 'color' or 'motion' or 'random'
        - trial_start: start time for the trial (ms)
        - fix_start: start time for fixation (ms)
        - fix_end: end time for fixation (ms)
        - stim_start: start time for stimulus (ms)
        - stim_end: end time for stimulus (ms)
        - resp_start: start time of the response period (ms)
        - resp_end: end time of the response period (ms)
        - trial_end: end time of the trial (ms)
        - explicit_rule: if the rule (context) is explicitly cued
        - train_pfc: if output of pfc needs to encode the rule info
                
    """
    dt = hp['dt']
    n_timesteps = int((hp_fusi['trial_end'] - hp_fusi['trial_start'])/dt)
    n_in = hp['n_input']
    n_in_rule = hp['n_input_rule_cue']
    n_out = hp['n_output']
    n_out_rule  = hp['n_output_rule']
    explicit_rule = hp['explicit_rule']
    train_rule = hp['train_rule']
#     batch_size = hp['batch_size']
    
    # the timing of different task event in unit of timesteps
    trial_start_ts = int(hp_fusi['trial_start']/dt)    
    fix_start_ts = int(hp_fusi['fix_start']/dt)   
    fix_end_ts = int(hp_fusi['fix_end']/dt)  
    rule_start_ts = int(hp_fusi['rule_start']/dt)
    rule_end_ts = int(hp_fusi['rule_end']/dt)
    stim_start_ts = int(hp_fusi['stim_start']/dt)
    stim_end_ts = int(hp_fusi['stim_end']/dt)
    resp_start_ts = int(hp_fusi['resp_start']/dt)
    resp_end_ts = int(hp_fusi['resp_end']/dt)
    trial_end_ts = int(hp_fusi['trial_end']/dt)  
            
    # check input dim
    if (explicit_rule==True and n_in!=9) or (explicit_rule==False and n_in!=5):
        raise ValueError('check input dimensionality!')

    # generate coherences
    stim = random.choices(['A', 'B', 'C', 'D'], k=n_trials)    # color coherence. positive for red, negative for green
    A_trs = [i for i in range(len(stim)) if stim[i]=='A']
    B_trs = [i for i in range(len(stim)) if stim[i]=='B']
    C_trs = [i for i in range(len(stim)) if stim[i]=='C']
    D_trs = [i for i in range(len(stim)) if stim[i]=='D']
    
    # generate rules for each trial
    if rule=='cxt1':
        rules = random.choices(['cxt1'], k=n_trials)
    elif rule=='cxt2':
        rules = random.choices(['cxt2'], k=n_trials)
    elif rule=='random':
        rules = random.choices(['cxt1', 'cxt2'], k=n_trials)
    else:
        raise ValueError('rule name "{}" does not exist!'.format(rule))
#     print(rules)
    cxt1_trs = [i for i in range(len(rules)) if rules[i]=='cxt1']
    cxt2_trs = [i for i in range(len(rules)) if rules[i]=='cxt2']
    
    rule_cues = np.zeros(n_trials)    # non-zero only for explicit rule condition
    
    if explicit_rule==True:   # TODO: check if this part is correct
        cxt1_cue1_trs = random.sample(cxt1_trs,int(len(cxt1_trs)/2))    # there are 2 cues for each rule
        cxt1_cue2_trs = [i for i in cxt1_trs if i not in cxt1_cue1_trs]
        cxt2_cue1_trs = random.sample(cxt2_trs,int(len(cxt2_trs)/2))
        cxt2_cue2_trs = [i for i in cxt2_trs if i not in cxt2_cue1_trs]
        
        rule_cues[cxt1_cue1_trs] = 1
        rule_cues[cxt1_cue2_trs] = 2
        rule_cues[cxt2_cue1_trs] = 3
        rule_cues[cxt2_cue2_trs] = 4
    
    left_trs = [(r=='cxt1' and (st=='A' or st=='C')) or (r=='cxt2' and (st=='B' or st=='D')) 
                             for (r, st) in zip(rules, stim)]
    right_trs = [(r=='cxt1' and (st=='B' or st=='D')) or (r=='cxt2' and (st=='A' or st=='C'))
                             for (r, st) in zip(rules, stim)]
    
    # generate training data
    yhat = torch.zeros(size=[n_trials, n_out, n_timesteps])
    yhat_rule = torch.zeros(size=[n_trials, n_out_rule, n_timesteps])
    x = torch.zeros(size=[n_trials, n_in, n_timesteps])
    x_rule = torch.zeros(size=[n_trials, n_in_rule, n_timesteps])
    
    # compute x
    x[:, 0, fix_start_ts:fix_end_ts] = 1    # the fixation input
#     print(A_trs, B_trs, C_trs, D_trs)
    x[A_trs, 1, stim_start_ts:stim_end_ts] = 1    
    x[B_trs, 2, stim_start_ts:stim_end_ts] = 1    
    x[C_trs, 3, stim_start_ts:stim_end_ts] = 1  
    x[D_trs, 4, stim_start_ts:stim_end_ts] = 1    
    if explicit_rule==True:
        x_rule[cxt1_cue1_trs, 0, rule_start_ts:rule_end_ts] = 1
        x_rule[cxt1_cue2_trs, 1, rule_start_ts:rule_end_ts] = 1
        x_rule[cxt2_cue1_trs, 2, rule_start_ts:rule_end_ts] = 1
        x_rule[cxt2_cue2_trs, 3, rule_start_ts:rule_end_ts] = 1
    # add noise to sensory input
    x[:, 1:5, stim_start_ts:stim_end_ts] += hp['input_noise_perceptual'] * torch.normal(mean=torch.zeros(n_trials, 4, int(stim_end_ts-stim_start_ts)), std=1)
    # add noise to rule input
    if explicit_rule==True:
        x_rule[:, :, rule_start_ts:rule_end_ts] += hp['input_noise_rule'] * torch.normal(mean=torch.zeros(n_trials, 4, int(rule_end_ts-rule_start_ts)), std=1)

    # compute yhat
    yhat[:, 2, fix_start_ts:fix_end_ts] = 1    # fixation output
    yhat[left_trs, 0, resp_start_ts:resp_end_ts] = 1
    yhat[right_trs, 1, resp_start_ts:resp_end_ts] = 1
    
    # compute yhat_rule if needed 
    if train_rule==True:
        yhat_rule[cxt1_trs, 0, rule_start_ts:trial_end_ts] = 1    # auxillary rule output
        yhat_rule[cxt2_trs, 1, rule_start_ts:trial_end_ts] = 1
    elif train_rule==False:
        yhat_rule = None
    
    task_data = {'rules': rules, 'rule_cues': rule_cues, 'stims': stim, 'left_trs': left_trs, 'right_trs': right_trs}
    
    return x, x_rule, yhat, yhat_rule, task_data
