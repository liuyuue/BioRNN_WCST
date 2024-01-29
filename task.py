# Copyright 2024 Yue Liu

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and limitations under the License


from runpy import run_module
import numpy as np; np.set_printoptions(precision=2); np.random.seed(0)
import torch; torch.set_printoptions(precision=2)
import torch.nn as nn
import matplotlib.pyplot as plt
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


def get_default_hp_wcst():
    """ Default parameters for the task """
    
    hp_wcst = {
                # trial start time (in ms)
                'trial_start': 0,
                # fixation start time
                'fix_start': 0,
                # fixation end time
                'fix_end': 0,
                # trial history input start time
                'trial_history_start': 0,
                # trial history input end time
                'trial_history_end': 100,
                # stimulus start time
                'center_card_on': 1100,
                # stimulus end time
                'center_card_off': 2100,
                # response start time
                'test_cards_on': 1600,
                # response end time
                'test_cards_off': 2100,
                # response cue start time
                'resp_start': 1600,
                # response cue end time
                'resp_end': 2100,
                # trial end time
                'trial_end': 2100
                }

    return hp_wcst


    
class WCST():
    """ The Wisconsin card sorting task 
        rule list: the rules (color, shape, etc.)
        n_features_per_rule: number of features per rule (e.g. how many colors are there)
        n_test_cards: number of cards to match to the center card
    """

    def __init__(self, hp, hp_wcst, rule, rule_list, n_features_per_rule, n_test_cards):
        self.rule = rule
        self.rule_list = rule_list
        self.n_features_per_rule = n_features_per_rule
        self.n_test_cards = n_test_cards
        self.dt = hp['dt']
        self.timestamps = hp_wcst
        self.n_ts = (self.timestamps['trial_end'] - self.timestamps['trial_start'])//self.dt
        self.n_ts = int(self.n_ts)
    
    def make_task_1tr(self):
        # generate the center card
#         print('self.rule_list = {}'.format(self.rule_list), flush=True)
        center_card = dict.fromkeys(self.rule_list)    # the card to match all the other cards to
        for r in self.rule_list:
            center_card[r] = random.randrange(self.n_features_per_rule)
        
        # generate the test cards
        test_cards = dict.fromkeys(np.arange(self.n_test_cards))  
        for c in range(self.n_test_cards):
            test_cards[c] = dict.fromkeys(self.rule_list)    # each card is also a dict
        ## define the matched features. The match card as the same rule feature as the center card, whereas the nonmatch card has the same (random) non-rule feature as the center card
        match_card_id = np.random.choice(np.arange(self.n_test_cards))    # which card is the matched one
#         print('center card={}'.format(center_card), flush=True)
        test_cards[match_card_id][self.rule] = center_card[self.rule]    # the match card has the same rule feature as the center card
        ## define the irrelevant rule and the nonmatch card. The nonmatch card has the same feature for the irrelvant rule as the center card 
        irrel_rule = random.choice([r for r in self.rule_list if r!=self.rule])    # the irrelevant rule, or the rule that the nonmatch card has the same feature as the center card   
#         nonmatch_card_id = [c for c in np.arange(self.n_test_cards) if c!=match_card_id][0]    # the nonmatch card is a random card from the rest
        nonmatch_card_id = random.choice([c for c in np.arange(self.n_test_cards) if c!=match_card_id])    # the nonmatch card is a random card from the rest
        test_cards[nonmatch_card_id][irrel_rule] = center_card[irrel_rule]    # the nonmatch card has the same feature as the center card but for the invalid rule 
        ## define the other features
        for c in range(self.n_test_cards):
            for r in self.rule_list:
                if (c==match_card_id and r==self.rule) or (c==nonmatch_card_id and r==irrel_rule):
                    continue
                elif (c!=match_card_id and r==self.rule):
                    test_cards[c][r] = random.choice([f for f in range(self.n_features_per_rule) if f!=center_card[self.rule]])
                elif (c!=nonmatch_card_id and r==irrel_rule):
                    test_cards[c][r] = random.choice([f for f in range(self.n_features_per_rule) if f!=center_card[irrel_rule]])
#                 else:
#                     test_cards[c][r] = random.randrange(self.n_features_per_rule)    # other features can be random
#                     print('test card {}, {}={}\n'.format(c, r, test_cards[c][r]), flush=True)
        
        
#         print('center card: {}\ntest cards: {}'.format(center_card, test_cards), flush=True)
#         print('rule={}, match_card_id={}, nonmatch_card_id={}\n'.format(self.rule, match_card_id, nonmatch_card_id), flush=True)
        
        
        # generate the input and target currents
        x = torch.zeros([self.n_ts, self.n_features_per_rule*len(self.rule_list)*(self.n_test_cards+1)])    # along the last dimension, the first few entries are the properties of the center card, the rest are the test cards
#         print('n_ts = {}'.format(self.n_ts), flush=True)
        x_rule = torch.zeros([self.n_ts, len(self.rule_list)])
        yhat = torch.zeros([self.n_ts, self.n_test_cards])
        yhat_rule = torch.zeros([self.n_ts, len(self.rule_list)])
        
        ## center card
        x_idx = 0     # the current index for the input
#         print('center card', flush=True)
        for r in self.rule_list:
#             print('rule {}'.format(r), flush=True)
#             print('x_idx={}'.format(x_idx), flush=True)
            x[int(self.timestamps['center_card_on']//self.dt):int(self.timestamps['center_card_off']//self.dt), x_idx+center_card[r]] = 1
            x_idx += self.n_features_per_rule
        ## test cards
        for c in range(self.n_test_cards):
#             print('test card {}'.format(c), flush=True)
            for r in self.rule_list:
#                 print('rule {}'.format(r), flush=True)
#                 print('x_idx={}'.format(x_idx), flush=True)
#                 print('x shape: {}'.format(x.shape), flush=True)
#                 print('test_cards[c][r]={}'.format(test_cards[c][r]), flush=True)
#                 print([self.timestamps['test_cards_on']//self.dt, self.timestamps['test_cards_off']//self.dt], flush=True)
                x[np.arange(self.timestamps['test_cards_on']//self.dt, self.timestamps['test_cards_off']//self.dt), x_idx+test_cards[c][r]] = 1
                x_idx += self.n_features_per_rule
                    
        ## target
        yhat[int(self.timestamps['resp_start']//self.dt):int(self.timestamps['resp_end']//self.dt), match_card_id] = 1    # the response output target
        for i in range(len(self.rule_list)):
            if self.rule_list[i]==self.rule:
                rule_idx = i
        yhat_rule[int(self.timestamps['trial_start']//self.dt):int(self.timestamps['trial_end']//self.dt), rule_idx] = 1    # the rule output target

        # other information about this trial
        task_data = {'center_card': center_card, 'test_cards': test_cards, 'correct_id': match_card_id}
        
#         print('x={}, x_rule={}, yhat={}, yhat_rule={}\n'.format(torch.mean(x, dim=0), torch.mean(x_rule, dim=0), torch.mean(yhat, dim=0), torch.mean(yhat_rule, dim=0)), flush=True)
        
        return x, x_rule, yhat, yhat_rule, task_data
        
    
    def make_task_batch(self, batch_size):
        x = []
        x_rule = []
        yhat = []
        yhat_rule = []
        task_data = []
        for ba in range(batch_size):
            _x, _x_rule, _yhat, _yhat_rule, _task_data = self.make_task_1tr()
            x.append(_x)
            yhat.append(_yhat)
            yhat_rule.append(_yhat_rule)
            task_data.append(_task_data)
        x = torch.stack(x, dim=1)
        x_rule = torch.tensor(x_rule)
        yhat = torch.stack(yhat, dim=1)
        yhat_rule = torch.stack(yhat_rule, dim=1)

        return x, x_rule, yhat, yhat_rule, task_data
    
    
    def get_perf(self, y, yhat):
        """ From the output and target, get the performance of the network 
            Args:
                y: batch_size*n_output*n_timesteps.
                yhat: batch_size*n_output*n_timesteps.
            Returns:
                resp_correct: length batch_size binary vector
        """
#         print(y.shape, yhat.shape, flush=True)
        if y.size()[-1]!=3 or yhat.size()[-1]!=3:
            raise ValueError('This function only works when there are 3 choices!')
            
        resp_start_ts = int(self.timestamps['resp_start']/self.dt)
        resp_end_ts = int(self.timestamps['resp_end']/self.dt)

#         softmax = nn.Softmax(dim=1)    # softmax would soften the difference a lot and worsen the performance...

        y_choice = torch.mean(y[resp_start_ts:resp_end_ts, :, :], dim=0)    # the mean network output during choice period
        choice_prob = y_choice    # convert output into choice probability        
        choice = torch.zeros([choice_prob.shape[0], 3]).to(choice_prob.device)    # compute choices from choice probabilities
        
        # winner take all
        for j in range(choice.shape[1]):
            choice[:,j] = torch.tensor([1 if choice_prob[i, j]==torch.max(choice_prob[i, :]) else 0 for i in range(choice_prob.shape[0])])    # the maximum of the three outputs is the choice
        
        # alternatively, sample a choice using output neurons' activity
#         for i in range(choice.shape[0]):
# #             choice_prob_norm = (choice_prob[i, :]/torch.sum(choice_prob[i, :])).numpy()
#             choice_prob_norm = torch.softmax(choice_prob[i, :]/0.4, dim=0).numpy()
#             if i==0:
#                 print('choice_prob_norm = {}'.format(choice_prob_norm))
#             j = np.random.choice([0, 1, 2], size=1, p=choice_prob_norm)
#             choice[i, j] = 1        
        
        
        target = torch.mean(yhat[resp_start_ts:resp_end_ts, :, :], dim=0)
        target_prob = target

    #     print('choice device: {}. target_prob device: {}'.format(choice.device, target_prob.device))
#         match = torch.abs(choice - target_prob) <= 0.5     # correct when the   
    #     match = torch.abs(choice_prob - target_prob) <= 0.5    # to prevent low activity for both output nodes
        match = (choice==target_prob)
        resp_correct = match[:,0] * match[:,1] * match[:,2]    # correct response if the probability from target is differed by less than threshold% for both choices

        return resp_correct, choice_prob, choice
    
    
    def get_perf_rule(self, y_rule, yhat_rule):
        """ Get the performance of the network for the rule output
            Args:
                y: batch_size*n_output*n_timesteps. default n_output=2
                yhat: batch_size*n_output*n_timesteps. default n_output=2
            Returns:
                resp_correct: length batch_size binary vector
        """
#         print(y.shape, yhat.shape, flush=True)
        if y_rule.size()[-1]!=2 or yhat_rule.size()[-1]!=2:
            raise ValueError('This function only works when there are 2 rules!')
            
        rule_start_ts = int(self.timestamps['trial_start']/self.dt)
        rule_end_ts = int(self.timestamps['trial_end']/self.dt)
        
#         softmax = nn.Softmax(dim=1)    # softmax would soften the difference a lot and worsen the performance...

        y_choice = torch.mean(y_rule[rule_start_ts:rule_end_ts, :, :], dim=0)    # the mean network output during choice period
        choice_prob = y_choice    # convert output into choice probability        
        choice = torch.zeros([choice_prob.shape[0], 2]).to(choice_prob.device)    # compute choices from choice probabilities
        for j in range(choice.shape[1]):
            choice[:, j] = torch.tensor([1 if choice_prob[i, j]==torch.max(choice_prob[i, :]) else 0 for i in range(choice_prob.shape[0])])    # the maximum of the outputs is the choice
        
        target = torch.mean(yhat_rule[rule_start_ts:rule_end_ts, :, :], dim=0)
    #     print('choice device: {}. target_prob device: {}'.format(choice.device, target_prob.device))
#         match = torch.abs(choice - target_prob) <= 0.5     # correct when the   
    #     match = torch.abs(choice_prob - target_prob) <= 0.5    # to prevent low activity for both output nodes
        _match = (choice==target)
        match = _match[:,0] * _match[:,1]    # correct response if the probability from target is differed by less than threshold% for both choices
        
        # alternatively
#         match = torch.tensor([choice[i, :]==target[i, :] for i in choice.shape[0]])

        return match, choice_prob, choice



    






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
                # response cue start time
                'resp_cue_start': 700,
                # response cue end time
                'resp_cue_end': 750,
                # trial end time
                'trial_end': 800,
                'trial_history_start': 0,
                'trial_history_end': 100,
                }

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
    resp_cue_start_ts = int(hp_cxtdm['resp_cue_start']/dt)
    resp_cue_end_ts = int(hp_cxtdm['resp_cue_end']/dt)
    trial_end_ts = int(hp_cxtdm['trial_end']/dt)  
            
    # check input dim
    if (explicit_rule==True and n_in!=9) or (explicit_rule==False and hp['resp_cue']==False and n_in!=5) or (explicit_rule==False and hp['resp_cue']==True and n_in!=6):
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
    
    # add a response cue
    if hp['resp_cue']==True:
        x[resp_cue_start_ts:resp_cue_end_ts, :, 5] = 1
    
    
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




