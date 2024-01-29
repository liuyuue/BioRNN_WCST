import numpy as np; np.set_printoptions(precision=4); np.random.seed(0)
import torch; torch.set_printoptions(sci_mode=False, precision=4)
import torch.nn as nn
import matplotlib.pyplot as plt; plt.rc('font', size=15)
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
import os

from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from task import *; from task_new import *
from functions import *
from model_working import *


# print(torch.__version__)
# print(sys.version)
                
# %matplotlib inline



def train_sbatch(jobname):
    hp, _, loss_fnc = get_default_hp()
    
    with open('/scratch/yl4317/two_module_rnn/sbatch/temp_hps/{}_hp.pickle'.format(jobname), 'rb') as handle:
        custom_hp = pickle.load(handle)
        
    for key in custom_hp.keys():
        hp[key] = custom_hp[key]    # use the customized hp
    print('Running job {}'.format(hp['jobname']))
    train_bpxtrials_v2_working(hp)
    
    os.remove('/scratch/yl4317/two_module_rnn/sbatch/temp_hps/{}_hp.pickle'.format(jobname))
    
    
def not_train_sbatch(jobname):
    hp, _, loss_fnc = get_default_hp()
    
    with open('/scratch/yl4317/two_module_rnn/sbatch/temp_hps/{}_hp.pickle'.format(jobname), 'rb') as handle:
        custom_hp = pickle.load(handle)
        
    for key in custom_hp.keys():
        hp[key] = custom_hp[key]    # use the customized hp
    print('Running job {}'.format(hp['jobname']))
    save_random_models(hp)
    
    os.remove('/scratch/yl4317/two_module_rnn/sbatch/temp_hps/{}_hp.pickle'.format(jobname))
    
    

def train_bpxtrials_v2_working(hp):
    ''' train a model that operates across trials '''
    
    start = time.time()
    times = {}
    times['forward'] = []
    
    # reset some of the hps 
    for area in ['sr', 'pfc']:
        hp['n_{}_edend'.format(area)] = hp['n_branches'] * hp['n_{}_esoma'.format(area)]
        
    # ensure reproducibility 
    torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms(True) 
    torch.backends.cudnn.deterministic = True 
    torch.manual_seed(hp['torch_seed'])
    np.random.seed(0)
    random.seed(0)
    
    # trace gradient
    torch.autograd.set_detect_anomaly(False)
    
    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}\n'.format(device), flush=True)
    if device=='cpu' and hp['batch_size']>100:
        raise ValueError('batch size too large for CPU!')
    
    # list of rules
    task = hp['task']    # cxtdm, salzman, wcst
    if task=='cxtdm': 
        rule_list = ['color', 'motion']    # for the cxtdm task
    elif task=='salzman':
        rule_list = ['cxt1', 'cxt2']    # for the Fusi&Salzman task
    elif task=='wcst':
        rule_list = ['color', 'shape']
        print('rule_list={}\n'.format(rule_list), flush=True)
    print('\ntask name: {}\n'.format(task))

    # add hook to the parameters to monitor gradients (optional)
    # for name, param in model.named_parameters():  
    #     if param.requires_grad==True:
    #         param.register_hook(lambda grad: print(name, grad))

    # save_name = '{}_init={}_l2h={}_torchseed={}_lr={}_optim={}'.format(hp['jobname'], hp['initialization'], hp['l2_h'], hp['torch_seed'], hp['lr'], hp['optimizer'].__name__)
    hp['save_name'] = hp['jobname']
    
    # if the model has reached convergence, exit
    saved_models = [f for f in os.listdir('/scratch/yl4317/two_module_rnn/saved_models/') if hp['save_name'] in f]
    if len(saved_models)!=0:
        print('found a model: {}, exit'.format(saved_models))
        return
    
    # check if there is a checkpoint
    chkpt_files = [f for f in os.listdir('/scratch/yl4317/two_module_rnn/saved_checkpoints/') if 'chkpt' in f and hp['save_name'] in f]
    if len(chkpt_files)==0:
        # no checkpoint
        print('no checkpoint, start a new training\n')
        ba_start = 0
        start_time = 0
        if hp['task']=='cxtdm':
            hp_task = get_default_hp_cxtdm() 
        elif hp['task']=='wcst':
            hp_task = get_default_hp_wcst()
        else:
            raise ValueError('{} task not implemented!'.format(hp['task']))
        model = Net_readoutSR_working(hp)
        model.to(device); model.rnn.to(device)    
       # model = SimpleNet_readoutSR(hp)    # simplified net
        if hp['optimizer']=='adam':  
            optimizer = torch.optim.Adam 
        elif hp['optimizer']=='SGD':  
            optimizer = torch.optim.SGD 
        elif hp['optimizer']=='Rprop':  
            optimizer = torch.optim.Rprop 
        elif hp['optimizer']=='RMSprop':  
            optimizer = torch.optim.RMSprop 
        else:
            raise NotImplementedError   # Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD
        optim = optimizer(params=model.parameters(), lr=hp['learning_rate'])    # instantiate the optimizer. amsgrad may help with convergence (test?)  
    elif len(chkpt_files)==1:
        # found a checkpoint
#         if torch.cuda.is_available():
#             checkpoint = torch.load('/scratch/yl4317/two_module_rnn/saved_checkpoints/'+chkpt_files[0], map_location=device)
#         else:
        checkpoint = torch.load('/scratch/yl4317/two_module_rnn/saved_checkpoints/'+chkpt_files[0], map_location=device)
#         checkpoint = chkpt_files[0]
        ba_start = int(checkpoint['step'])
        start_time = checkpoint['time']    # how long it has take
        hp_task = checkpoint['hp_task']
#         with HiddenPrints():
        model = Net_readoutSR_working(hp)
        model.to(device); model.rnn.to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if hp['optimizer']=='adam':  
            optimizer = torch.optim.Adam 
        elif hp['optimizer']=='SGD':  
            optimizer = torch.optim.SGD 
        elif hp['optimizer']=='Rprop':  
            optimizer = torch.optim.Rprop 
        elif hp['optimizer']=='RMSprop':  
            optimizer = torch.optim.RMSprop 
        else:
            raise NotImplementedError   # Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD
        optim = optimizer(params=model.parameters(), lr=hp['learning_rate'])    # instantiate the optimizer here
        optim.load_state_dict(checkpoint['optim_state_dict'])
#         # debug: move the optimizer to CUDA
#         for state in optimizer.state.values():
#             for k, v in state.items():
#                 if isinstance(v, torch.Tensor):
#                 if torch.is_tensor(v):
# #                     state[k] = v.cuda()
#                     state[k] = v.to(device)
        print('starting from checkpoint (step {})\n'.format(ba_start))
    elif len(chkpt_files)>1:
        raise ValueError('found more than 1 checkpoints: {}'.format(chkpt_files))
        
    # define loss function
#     _, _, loss_fnc = get_default_hp()
    if hp['loss_type']=='mse':
        loss_fnc = nn.MSELoss()
    else:
        raise NotImplementedError

    
    # delay variability
    delay_var = 200    # variability in the delay. 200 ms
    hp_task['delay_var'] = delay_var
    iti_var = 200    # variability in the ITI
    hp_task['iti_var'] = iti_var

    # print some hps (optional)
    for key in hp.keys():
        if key in ['mglur', 'divide_sr_sst_vip', 'sr_sst_high_bias', 'no_pfcesoma_to_srsst',
                  'no_pfcesoma_to_sredend', 'no_pfcesoma_to_srpv', 'no_pfcesoma_to_srvip',
                  'fdbk_to_vip', 'grad_remove_history', 'trainable_dend2soma',
                  'divisive_dend_inh', 'divisive_dend_ei', 'scale_down_init_wexc'] and hp[key]==True:
            print('{}={}\n'.format(key, hp[key]))
        if key in 'dend_nonlinearity':
            print('dendritic nonlinearity = {}\n'.format(hp[key]))
        if key in 'activation':
            print('somatic nonlinearity = {}\n'.format(hp[key]))
            
    # print some more
    print('Hyperparameters:\n{}\n'.format(hp), flush=True)
    print('Hyperparameters for the task:\n{}\n'.format(hp_task), flush=True)
    print(optim, flush=True)
    print('\n')
    print(model, flush=True)
    print('\n')

    # display connectivity
#     display_connectivity(model, plot=False)

    
    #============================START===========================#
    plot=hp['plot_during_training']      # whether to plot during training 
    
    if len(chkpt_files)==0:
        perf_list_big = []
        perf_rule_list_big = []
        loss_list_big = []
        test_perf_list_big = []
        test_perf_rule_list_big = []
        test_loss_list_big = []
        rnn_activity_list = []

        # resetting network + curriculum learning
        reset_network = False
        give_prev_stim = True
        give_prev_choice = True
        give_prev_rew = True
        curriculum = True    # curriculum learning: gradually remove trial history inputs
        prev_stim_mag = 1
        prev_choice_mag = 1
        prev_rew_mag = 1
        print('Reset network: {}, give_prev_stim: {}, give_prev_choice: {}, give_prev_rew: {}\n'.
              format(reset_network, give_prev_stim, give_prev_choice, give_prev_rew), flush=True)
        curriculum_t = False
        if curriculum_t==True:    # curriculum learning: gradually increase block length
            hp['block_len_n_swit_comb'] = [(20,2)]    # curriculum: gradually increase number of trials
            hp['block_len'], hp['n_switches'] = hp['block_len_n_swit_comb'][0]
            curr_t_progress = 0    # an index that tracks progress through curriculum learning
            print('Curriculum learning in time dimension: Block length = {}. Number of switches = {}\n'.format(hp['block_len'], hp['n_switches']), flush=True)
    elif len(chkpt_files)==1:
        # TODO
        if torch.cuda.is_available():
            checkpoint = torch.load('/scratch/yl4317/two_module_rnn/saved_checkpoints/'+chkpt_files[0])
        else:
            checkpoint = torch.load('/scratch/yl4317/two_module_rnn/saved_checkpoints/'+chkpt_files[0], map_location=torch.device('cpu'))
#         checkpoint = chkpt_files[0]
        perf_list_big = checkpoint['perf_list']
        perf_rule_list_big = checkpoint['perf_rule_list']
        loss_list_big = checkpoint['loss_list']
        test_perf_list_big = checkpoint['test_perf_list']
        test_perf_rule_list_big = checkpoint['test_perf_rule_list']
        test_loss_list_big = checkpoint['test_loss_list']
        rnn_activity_list = []

        # resetting network + curriculum learning
        reset_network = checkpoint['reset_network']    # should be False
        give_prev_stim = checkpoint['give_prev_stim']
        give_prev_choice = checkpoint['give_prev_choice']
        give_prev_rew = checkpoint['give_prev_rew']
        curriculum = checkpoint['curriculum']    # curriculum learning: gradually remove trial history inputs
        prev_stim_mag = checkpoint['prev_stim_mag']
        prev_choice_mag = checkpoint['prev_choice_mag']
        prev_rew_mag = checkpoint['prev_rew_mag']
        print('Reset network: {}, give_prev_stim: {}, give_prev_choice: {}, give_prev_rew: {}\n'.
              format(reset_network, give_prev_stim, give_prev_choice, give_prev_rew), flush=True)
        curriculum_t = [checkpoint['curriculum_t'] if 'curriculum_t' in checkpoint.keys() else False]
        if curriculum_t==True:    # curriculum learning: gradually increase block length (not used anymore)
            raise NotImplementedError
    elif len(chkpt_files)>1:
        raise ValueError('found more than 1 checkpoints: {}'.format(chkpt_files))


    #=== start training ===#
    for ba in range(ba_start, int(1e10)): 
        start_step = time.time()
        
#         current_rule = random.choice(rule_list)    # randomly choose a rule to start
        current_rule = rule_list[0]    # use a fixed rule
        perf_list = []
        perf_rule_list = []
        rnn_activity_over_trials = []
        loss = 0

        block_len = hp['block_len'] + random.randint(0,0)    # test: variable block length
        switches = random.sample(range(block_len-1), hp['n_switches'])

        if hp['grad_remove_history']==True:
            if give_prev_stim==False:
#                 prev_stim_mag -= 1e-3    # gradually remove it
                prev_stim_mag = 0     # suddenly remove it
                prev_stim_mag = max(prev_stim_mag, 0)
            if give_prev_rew==False:
#                 prev_rew_mag -= 1e-3    # gradually remove it
                prev_rew_mag = 0
                prev_rew_mag =  max(prev_rew_mag, 0)
            if give_prev_choice==False:
#                 prev_choice_mag -= 1e-3    # gradually remove it
                prev_choice_mag = 0
                prev_choice_mag = max(prev_choice_mag, 0)

        for tr in range(block_len):
#             print('tr {}'.format(tr), flush=True)
            # compute the trial history
            if tr==0:
                h_init = None
                i_me_init = None
                last_rew = None
                prev_stim = None
                prev_choice = None
            else:
                if reset_network==False:
                    if hp['bpx1tr']==True:
                        h_init = h_last.detach()    # for backprop after each trial
                        i_me_init = i_me_last.detach()
                    else:
                        h_init = h_last 
                        i_me_init = i_me_last
                else:
                    h_init = None
                    i_me_init = None
                if give_prev_rew==True or hp['grad_remove_history']==True:
                    last_rew = perf.detach()    # detach it from the graph
                else:
                    last_rew = None
                if give_prev_stim==True or hp['grad_remove_history']==True:
                    prev_stim = _x.detach()    # detach it from the graph
                else:
                    prev_stim = None
                if give_prev_choice==True or hp['grad_remove_history']==True:
                    prev_choice = choice.detach()    # detach it from the graph
                else:
                    prev_choice = None



            # implement variable trial duration
            hp_task_var_delay = copy.deepcopy(hp_task)
            iti = hp_task['center_card_on'] - hp_task['trial_history_end'] + int(np.random.uniform(low=-iti_var, high=iti_var))
            hp_task_var_delay['center_card_on'] = hp_task_var_delay['trial_history_end'] + iti
            hp_task_var_delay['center_card_off'] = hp_task_var_delay['center_card_on'] + (hp_task['center_card_off'] - hp_task['center_card_on'])
            hp_task_var_delay['test_cards_on'] = hp_task_var_delay['center_card_on'] + (hp_task['test_cards_on'] - hp_task['center_card_on'])
            hp_task_var_delay['test_cards_off'] = hp_task_var_delay['test_cards_on'] + (hp_task['test_cards_off'] - hp_task['test_cards_on'])
            hp_task_var_delay['resp_start'] = hp_task_var_delay['test_cards_on']
            hp_task_var_delay['resp_end'] =  hp_task_var_delay['test_cards_off']
            hp_task_var_delay['trial_end'] = hp_task_var_delay['resp_end']
#             print('gewg hp_task_var_delay', hp_task_var_delay)

#             hp_task_var_delay['resp_start'] = hp_task['resp_start'] + np.random.uniform(low=-delay_var, high=delay_var)    # adjust this 
#             hp_task_var_delay['resp_end'] = hp_task_var_delay['resp_start'] + (hp_task['resp_end']-hp_task['resp_start'])    # resp duration remains the same
#             hp_task_var_delay['trial_end'] = hp_task_var_delay['resp_end']
#             if hp['task']=='wcst':
#                 hp_task_var_delay['test_cards_on'] = hp_task_var_delay['resp_start']
#                 hp_task_var_delay['test_cards_off'] = hp_task_var_delay['resp_end']
#             if hp['resp_cue']==True:
#                 hp_task_var_delay['resp_cue_start'] = hp_task_var_delay['resp_start']    # change the start of the response cue
#                 hp_task_var_delay['resp_cue_end'] = hp_task_var_delay['resp_cue_start'] + (hp_task['resp_cue_end']-hp_task['resp_cue_start'])     # keep the duration of the response cue the same 


            # compute the trial history current
            start_trialhistory = time.time()
            input_period = np.arange(int(hp_task_var_delay['trial_history_start']/hp['dt']), int(hp_task_var_delay['trial_history_end']/hp['dt']))    # input period for the trial history info
            n_steps = (hp_task_var_delay['trial_end'] - hp_task_var_delay['trial_start'])//hp['dt']
            n_steps = int(n_steps)
            if hp['task']=='cxtdm':
                ts_prev_stim_start = hp_task_var_delay['stim_start']
                ts_prev_stim_end = hp_task_var_delay['stim_end']
                I_prev_rew, I_prev_stim, I_prev_choice = compute_trial_history(last_rew=last_rew, prev_stim=prev_stim, prev_choice=prev_choice, input_period=input_period, batch_size=hp['batch_size'], n_steps=n_steps, input_dim=model.rnn.n['input'], stim_start=ts_prev_stim_start, stim_end=ts_prev_stim_end, dt=hp['dt'], choice_dim=2)    # each current is time*batch*feature
            if hp['task']=='wcst':
                ts_prev_stim_start = hp_task_var_delay['test_cards_on']    # here the center card is still shown
                ts_prev_stim_end = hp_task_var_delay['test_cards_off']
                I_prev_rew, I_prev_stim, I_prev_choice = compute_trial_history(last_rew=last_rew, prev_stim=prev_stim, prev_choice=prev_choice, input_period=input_period, batch_size=hp['batch_size'], n_steps=n_steps, input_dim=model.rnn.n['input'], stim_start=ts_prev_stim_start, stim_end=ts_prev_stim_end, dt=hp['dt'], choice_dim=3)    # each current is time*batch*feature
            I_prev_rew, I_prev_stim, I_prev_choice = I_prev_rew.to(device), I_prev_stim.to(device), I_prev_choice.to(device)
            trial_history = {'i_prev_rew': prev_rew_mag*I_prev_rew, 'i_prev_choice': prev_choice_mag*I_prev_choice, 'i_prev_stim': prev_stim_mag*I_prev_stim}
#             print('compute trial history input takes {:.2e}s, total time for this step = {:.2e}s'.format(time.time()-start_trialhistory, time.time()-start_step))
    
#                 # test: plot the history currents
#                 fig, ax=plt.subplots()
#                 ax.set_title('I_prev_rew')
#                 for i in range(I_prev_rew.shape[-1]):
#                     ax.plot(I_prev_rew[:,0,i])
#                 ax.set_ylim([-0.1, 1.1])
#                 fig, ax=plt.subplots()
#                 ax.set_title('I_prev_stim')
#                 for i in range(I_prev_stim.shape[-1]):
#                     ax.plot(I_prev_stim[:,0,i])
#                 ax.set_ylim([-0.1, 1.1])
#                 fig, ax=plt.subplots()
#                 ax.set_title('I_prev_choice')
#                 for i in range(I_prev_choice.shape[-1]):
#                     ax.plot(I_prev_choice[:,0,i])
#                 ax.set_ylim([-0.1, 1.1])
#                 plt.show()


            # generate input and target for 1 trial
            start_gendata = time.time()
            # fusi task
            if task=='salzman':
                _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_fusi(n_trials=hp['batch_size'], rule=current_rule, hp=hp, hp_fusi=hp_task_var_delay)
            # cxtdm task
            elif task=='cxtdm':
                if tr-1 in switches or tr==0:
                    _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_siegel(n_trials=hp['batch_size'], rule=current_rule, hp=hp, hp_cxtdm=hp_task_var_delay, trial_type='incongruent')    # such that the first trial after switch is always incongruent
                else:
                    _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_siegel(n_trials=hp['batch_size'], rule=current_rule, hp=hp, hp_cxtdm=hp_task_var_delay, trial_type='no_constraint')
            # wisconsin card sorting task
            elif task=='wcst':
                wcst = WCST(hp=hp, hp_wcst=hp_task_var_delay, rule=current_rule, rule_list=rule_list, n_features_per_rule=2, n_test_cards=3)
                _x, _x_rule, _yhat, _yhat_rule, task_data = wcst.make_task_batch(batch_size=hp['batch_size'])
            _x, _x_rule, _yhat, _yhat_rule = _x.to(device), _x_rule.to(device), _yhat.to(device), _yhat_rule.to(device)
            if hp['train_rule']==True:
                _yhat_rule = _yhat_rule.to(device)
            if hp['timeit_print']==True:
                print('generate data takes {:.2e}s, total time for this step = {:.2e}s'.format(time.time()-start_gendata, time.time()-start_step))

            # run model forward 1 trial
            start_forward = time.time()
            
            
#             # debug device
#             print('before forward pass')
#             for key in trial_history.keys():
#                 print(ba, tr, key, trial_history[key].device)
#             print(ba, tr, '_x device: {}'.format(_x.device)) 
#             if h_init is None:
#                 print(ba, tr, 'h_init is None')
#             else:
#                 print(ba, tr, 'h_init device: {}'.format(h_init.device))
#             if i_me_init is None:
#                 print(ba, tr, 'i_me_init is None')
#             else:
#                 print(ba, tr, 'i_me_init device: {}'.format(i_me_init.device))
#             for name, param in model.named_parameters():
#                 print(ba, tr, name, param.device)
#             model.to(device); model.rnn.to(device)
#             print('_x shape: {}\n'.format(_x.shape), flush=True)
            out, data = model(input=_x, init={'h': h_init, 'i_me': i_me_init}, trial_history=trial_history, hp=hp)
            if hp['timeit_print']==True:
                print('forward pass takes {:.2e}s, total time for this step = {:.2e}s'.format(time.time()-start_forward, time.time()-start_step))
#                 if ba>=0 and hp['timeit_print']==True:
#                     fptime = time.time()-start_forward
#                     print('forward pass takes {}s'.format(fptime), flush=True)
#                     times['forward'].append(fptime)
            rnn_activity = data['record']['hiddens']
            rnn_activity = torch.stack(rnn_activity, dim=0)
            rnn_activity_over_trials.append(rnn_activity)
            h_last = data['last_states']['hidden']
            i_me_last = data['last_states']['i_me']
            
            # check if some activity is exploding
            if hp['check_explode_cg']==True:
                explode_cgs = []
                for cg in model.rnn.cell_group_list:
                    mean_act_cg = torch.mean(rnn_activity[:,:,model.rnn.cg_idx[cg]])
                    if mean_act_cg>1e1 or torch.isnan(mean_act_cg):
                        print('step {}, tr {}, mean {} activity is {}!!!'.format(ba, tr, cg, mean_act_cg))
                        explode_cgs.append(cg)

            # get the performance
            _y = out['out']
            if hp['task']!='wcst':
                perf, choice_prob, choice = get_perf(y=_y, yhat=_yhat, hp=hp, hp_task=hp_task_var_delay)
#                 print('perf shape: {}'.format(perf.shape), flush=True)
            else:
                perf, choice_prob, choice = wcst.get_perf(y=_y, yhat=_yhat)
            if hp['train_rule']==True:
                _y_rule = out['out_rule']
                if hp['task']=='wcst':
                    perf_rule, _, _ = wcst.get_perf_rule(y_rule=_y_rule, yhat_rule=_yhat_rule)
                else:
                    perf_rule, _, _ = get_perf(y=_y_rule, yhat=_yhat_rule, hp=hp, hp_task=hp_task_var_delay)

            # accumulate loss
            # for debugging: mask our the 1st trial and see if error can backprop to the init state
    #         if tr==0:
    #             loss += 0
    #         else:
    #             loss += loss_fnc(_y, _yhat)
    #             if hp['train_rule']==True:
    #                 loss += loss_fnc(_y_rule, _yhat_rule)
            loss += loss_fnc(_y, _yhat) 
            loss += hp['l1_weight']*torch.norm(model.rnn.w_rec_eff, p=1)/torch.numel(model.rnn.w_rec_eff) + hp['l2_weight']*torch.norm(model.rnn.w_rec_eff, p=2)/np.sqrt(torch.numel(model.rnn.w_rec_eff)) + hp['l1_h']*torch.norm(rnn_activity, p=1)/torch.numel(rnn_activity) + hp['l2_h']*torch.norm(rnn_activity, p=2)/np.sqrt(torch.numel(rnn_activity))    # regularization term for the weights
            if 'sr_esoma' in hp['cell_group_list']:
                loss += hp['l2_h_sr']*torch.norm(rnn_activity[:, :, model.rnn.cg_idx['sr_esoma']], p=2)    # regularization for the activity
            if 'pfc_esoma' in hp['cell_group_list']:
                loss += hp['l2_h_pfc']*torch.norm(rnn_activity[:, :, model.rnn.cg_idx['pfc_esoma']], p=2)    # regularization for the activity
            if 'sr_edend' in hp['cell_group_list']:
                loss += hp['l1_h_sredend']*torch.norm(rnn_activity[:, :, model.rnn.cg_idx['sr_edend']], p=1)    # regularization for the dendritic activity for SR (l1 because the dendritic activity can be negative under the 'old' type dendritic nonlinearity)
    #         loss = loss_fnc(_y, _yhat)    # for backprop after each trial
            if hp['train_rule']==True:
                loss += loss_fnc(_y_rule, _yhat_rule)

            # test: would bp cross 1 trial work?
            if hp['bpx1tr']==True:
                optim.zero_grad()           # clear gradients for this training step
                loss_1tr = loss_fnc(_y, _yhat)
                loss_1tr += hp['l1_weight']*torch.norm(model.rnn.w_rec_eff, p=1)/torch.numel(model.rnn.w_rec_eff) + hp['l2_weight']*torch.norm(model.rnn.w_rec_eff, p=2)/np.sqrt(torch.numel(model.rnn.w_rec_eff)) + hp['l1_h']*torch.norm(rnn_activity, p=1)/torch.numel(rnn_activity) + hp['l2_h']*torch.norm(rnn_activity, p=2)/np.sqrt(torch.numel(rnn_activity))    # regularization term for the weights
                if 'sr_esoma' in hp['cell_group_list']:
                    loss_1tr += hp['l2_h_sr']*torch.norm(rnn_activity[:, :, model.rnn.cg_idx['sr_esoma']], p=2)    # regularization for the activity
                if 'pfc_esoma' in hp['cell_group_list']:
                    loss_1tr += hp['l2_h_pfc']*torch.norm(rnn_activity[:, :, model.rnn.cg_idx['pfc_esoma']], p=2)    # regularization for the activity
                if hp['train_rule']==True:
                    loss_1tr += loss_fnc(_y_rule, _yhat_rule)
                loss_1tr.backward(retain_graph=False)           # backpropagation, compute gradients
                loss_1tr = np.nan_to_num(loss_1tr.detach().cpu().numpy(), nan=np.inf)    # convert nan to inf in case loss blows up
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, error_if_nonfinite=False)    # gradient clip
                optim.step()
                
            # concatenate x and yhat across trials
#                 if tr==0:
#                     x = _x
#                     yhat = _yhat
#                     yhat_rule = _yhat_rule
#                     y = _y
#                     y_rule = _y_rule
#                 else:
#                     x = torch.cat((x, _x), axis=0)
#                     print(yhat.shape, _yhat.shape)
#                     yhat = torch.cat((yhat, _yhat), axis=2)
#                     y = torch.cat((y, _y), axis=2)
#                     if hp['train_rule']==True:
#                         yhat_rule = torch.cat((yhat_rule, _yhat_rule), axis=2)
#                         y_rule = torch.cat((y_rule, _y_rule), axis=2)           

            # collect perf across trials
            perf_list.append(torch.mean(perf.float().detach()).cpu())
            if hp['train_rule']==True:
                perf_rule_list.append(torch.mean(perf_rule.float().detach()).cpu())
            else:
                perf_rule_list.append(-1)

            # switch rule if necessary
            if tr in switches:
                if len(rule_list)==1:
                    next_rule = current_rule
                    print('only 1 rule in rule_list, not switching\n', flush=True)
                else:
                    next_rule = random.choice([r for r in rule_list if r!=current_rule])    # randomly switch to a different rule
                current_rule = next_rule

        # backprop at the end of block 
        if hp['bpx1tr']==False:
            start_backward = time.time()
            optim.zero_grad()           # clear gradients for this training step
            loss = loss/block_len
            loss.backward(retain_graph=True)           # backpropagation, compute gradients
#             print('loss device: {}'.format(loss.device))    # debug
            loss = np.nan_to_num(loss.detach().cpu().numpy(), nan=np.inf)    # convert nan to inf in case loss blows up
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, error_if_nonfinite=False)    # gradient clip
            # monitor average activity
            if grad_norm>=10:
                print('\nstep {}, large gradient norm encountered ({:0.4f}).\nchecking grad & avg activity...'.format(ba, grad_norm))
                for name, param in model.named_parameters():
                    if param.requires_grad==True:
                        if param.grad is not None:    # the param is part of the graph
                            print('{}, norm of grad: {:0.4f}, max grad: {:0.4f}'.format(name, torch.norm(param.grad.detach(), p=2), torch.max(param.grad.detach())))
    #             if torch.isnan(grad_norm)==False:
    
    
#             print('before optim.step()')
#             for name, param in model.named_parameters():   # debug
#                 print(ba, name, param.device)
#             print(model.rnn.w_rec.device, (prev_stim_mag*I_prev_stim).device, _x.device, _yhat.device, flush=True)    # debug
            
            
            optim.step()                # apply gradients when gradient is finite
            if hp['timeit_print']==True:
    #         if True:
                print('***backward pass takes {:.2e}s, total time for this step = {:.2e}s'.format(time.time()-start_backward, time.time()-start_step), flush=True)

        
        # record the recent network activity
        if ba%1==0 and hp['record_recent_rnn_activity']==True:
            min_tstep = (hp_task['trial_end']-delay_var)//hp['dt']    # since trial length is variable
            rnn_activity_over_trials_cut = [_[:min_tstep,:,:] for _ in rnn_activity_over_trials]    # cut to the minimum duration of a trial (since there is variable delay)
            rnn_activity_concat = concat_trials(torch.stack(rnn_activity_over_trials_cut))
            rnn_activity_list.append(rnn_activity_concat)    # renp.mean(data[i, model.rnn.cg_idx[cg],:], axis=0)h_initcord the activity
            if len(rnn_activity_list)>3:
                rnn_activity_list.pop(0)    # only record the last k steps to save memory

#                 for cg in model.rnn.cell_group_list:
#                     if (rnn_activity_concat[:, model.rnn.cg_idx[cg], :]>=10).any() or torch.isnan(rnn_activity_concat[:, model.rnn.cg_idx[cg], :]).any():
#                         print('step {}, {}, average activity over time in a block: {}\n'.format(ba, cg, torch.mean(rnn_activity_concat[:, model.rnn.cg_idx[cg], :], dim=(0,1))))


        # collect and print  
        if ba%1==0:
            perf_list_big.append(np.mean(perf_list))
            perf_rule_list_big.append(np.mean(perf_rule_list))
            loss_list_big.append(loss)

        if ba%10==0:
#                 print('Step {}, dend2soma={}'.format(ba, model.rnn.dend2soma.data))
#                 print('Step {}, SST bias {}'.format(ba, model.rnn.bias[model.rnn.cg_idx['sr_sst']]))
            print('Step {}, total loss={:0.4f}, perf={:0.4f}, perf rule={:0.4f}, time={:0.2f}s, ps/c/r={}/{}/{}, bl_len={}, swit={}'
                  .format(ba, loss, np.mean(perf_list), np.mean(perf_rule_list), time.time()-start+start_time, give_prev_stim, give_prev_choice, give_prev_rew, 
                          hp['block_len'], switches), flush=True)
            print('prev stim/choice/rew mag: {}/{}/{}\n'.format(prev_stim_mag, prev_choice_mag, prev_rew_mag))
    #         print('h0={}'.format(model.rnn.h0[0:3,0:3]))

        if ba%1000==0 and ba>0:
            print('torch seed = {}\n'.format(hp['torch_seed']))
            print('reset network: {}, give_prev_stim: {}, give_prev_choice: {}, give_prev_rew: {}\n'.format(reset_network, give_prev_stim, give_prev_choice, give_prev_rew), flush=True)
            print('Block length = {}. Number of switches = {}\n'.format(hp['block_len'], hp['n_switches']), flush=True)
            for key in hp.keys():
                print('{}: {}'.format(key, hp[key]))
#             print(hp)
            print('\n')
            print(optim)
            print('\n')
            print(hp_task)
            print('\n')
#             display_connectivity(model=model, plot=False)

            print(switches, flush=True)
            if plot==True:
                plot_y_yhat(y.cpu(), yhat.cpu())
                if hp['train_rule']==True:
                    plot_y_yhat(y_rule.cpu(), yhat_rule.cpu())
                plot_perf(perf_list)
                plot_perf(perf_rule_list, title='performance (rule)', ylabel='performance (rule)')
                
        # save checkpoint
        if ba%10==0 and ba>0:
            print('saving checkpoint. name={}_{}'.format('chkpt', hp['save_name']))
            saved_file = saved_file = {'step': ba, 'time': time.time()-start+start_time, 'model': model, 'model_state_dict': model.state_dict(), 'optimizer': optim, 'optim_state_dict': optim.state_dict(), 'hp': hp, 'hp_task': hp_task, 'perf_list':perf_list_big, 'perf_rule_list': perf_rule_list_big, 'loss_list': loss_list_big,'test_perf_list': test_perf_list_big, 'test_perf_rule_list': test_perf_rule_list_big, 'test_loss_list': test_loss_list_big, 'reset_network': reset_network, 'give_prev_stim': give_prev_stim, 'give_prev_choice': give_prev_choice, 'give_prev_rew': give_prev_rew, 'curriculum': curriculum, 'prev_stim_mag': prev_stim_mag, 'prev_choice_mag': prev_choice_mag, 'prev_rew_mag': prev_rew_mag}
            torch.save(saved_file, '/scratch/yl4317/two_module_rnn/saved_checkpoints/{}_{}'.format('chkpt', hp['save_name'][:150]))     # save a copy in scratch

            
        # testing
        perf_crit_train = 1-hp['n_switches']/hp['block_len']-0.2    # performance criteria
        mean_recent_perf = np.mean(perf_list_big[-1:])
        mean_recent_perf_rule = np.mean(perf_rule_list_big[-1:])  
        if mean_recent_perf>= perf_crit_train and mean_recent_perf_rule>=perf_crit_train:
            print('\nStep {}, perf: {:.4f}/{:.4f}, criteria for begin testing: {:.4f}. \n Start testing...........'.format(ba, mean_recent_perf, mean_recent_perf_rule, perf_crit_train), flush=True)
            n_trials_test = 200
            switch_every_test = 20
            max_perf_test = 1-1/switch_every_test
            if hp['n_switches']<2:
                print('\n<2 switches during training. Therefore skip testing\n', flush=True)
                perf_test = 1
                perf_rule_test = 1
                loss_test = 0
            else:
                init_rule = random.choice(rule_list)
                print('init_rule={}'.format(init_rule))
                perf_test, perf_rule_test, loss_test, _ = test_frozen_weights(model=model, 
                                                                              n_trials_test=n_trials_test, 
                                                                              switch_every_test=switch_every_test, 
                                                                              init_rule=init_rule,
                                                                              task=task,
                                                                              hp=hp, 
                                                                              hp_task=hp_task, 
                                                                              loss_fnc=loss_fnc, 
                                                                              delay_var = 200,
                                                                              give_prev_rew=give_prev_rew, 
                                                                              give_prev_stim=give_prev_stim, 
                                                                              give_prev_choice=give_prev_choice,
                                                                              noiseless=False)
#                     _, _, _, test_data = test_frozen_weights(model=model, n_trials_test=n_trials_test, switch_every_test=switch_every_test, 
#                                              init_rule=random.choice(rule_list), hp=hp_test, task=task,
#                                              loss_fnc=nn.MSELoss(), hp_task=hp_task_test,
#                                              delay_var=0, 
#                                              give_prev_choice=True, give_prev_stim=True, give_prev_rew=True, plot=to_plot, 
#                                              random_switch=random_switch, n_switches=n_switches)
            test_perf_list_big.append(perf_test)
            test_perf_rule_list_big.append(perf_rule_test)
            test_loss_list_big.append(loss_test)
            criteria_perf = max_perf_test - 0.15
            print('test perf: {}/{}, max_perf_test = {}, criteria for continuing to next stage of curriculum learning: {}'.format(perf_test, perf_rule_test, max_perf_test, criteria_perf), flush=True)
            # curriculum learning
#                 if np.mean(test_perf_list_big[-5:])>=max_perf_test-0.05 and np.mean(test_perf_rule_list_big[-5:])>=max_perf_test-0.05:
            if np.mean(test_perf_list_big[-1:])>=criteria_perf and np.mean(test_perf_rule_list_big[-1:])>=criteria_perf:
                print('recent testing performance meets criteria ({})\n'.format(criteria_perf), flush=True)
                if curriculum==False:
                    if curriculum_t==False:     # no curriculum learning
                        print('Testing performance reaches criterion after no curriculum learning. Training ends.', flush=True)
                        success = 'success'
                        break
                    elif curriculum_t==True and curr_t_progress!=len(hp['block_len_n_swit_comb']):    # curriculum learning on block length, progress +1
                        curr_t_progress += 1
                        hp['block_len'], hp['n_switches'] = hp['block_len_n_swit_comb'][curr_t_progress]
                        max_perf_train = 1-hp['n_switches']/hp['block_len']
                        print('*** Increase block length. New block length = {} trials. New n_swtiches = {} ***'.format(hp['block_len'], hp['n_switches']), flush=True)
                        give_prev_stim, give_prev_choice, give_prev_rew = True, True, True
                        continue
                    elif curriculum_t==True and curr_t_progress==len(hp['block_len_n_swit_comb']):    # curriculum learning on block length completed
                        print('Testing performance reaches criterion after curriculum training on block length. Training ends.', flush=True)
                        success = 'success'
                        break
                elif curriculum==True:
                    if give_prev_stim==True:    # curriculum learning on trial history, progress +1
                        give_prev_stim = False
                        print('******** Now not giving previous stimulus ********\n', flush=True)
                        optim = optimizer(params=model.parameters(), lr=hp['learning_rate'])    # re-initialize the optimizer?
                        continue
                    elif give_prev_choice==True and prev_stim_mag==0:    # curriculum learning on trial history, progress +1
                        give_prev_choice = False
                        print('******** Now not giving previous choice ********\n', flush=True)
                        optim = optimizer(params=model.parameters(), lr=hp['learning_rate'])    # re-initialize the optimizer?
                        continue
                    else:
                        print('prev_choice_mag={}'.format(prev_choice_mag))
                        if curriculum_t==False and prev_choice_mag==0:    # curriculum learning on trial history completed
                            print('Testing performance reaches criterion after curriculum training on trial history. Training ends.', flush=True)
                            success = 'success'
                            break
                        elif curriculum_t==True and curr_t_progress!=len(hp['block_len_n_swit_comb']):    # curriculum learning on block length, progress +1
                            curr_t_progress += 1
                            hp['block_len'], hp['n_switches'] = hp['block_len_n_swit_comb'][curr_t_progress]
                            max_perf_train = 1-hp['n_switches']/hp['block_len']
                            print('*** Increase block length. New block length = {} trials. New n_swtiches = {} ***'.format(hp['block_len'], hp['n_switches']), flush=True)
                            give_prev_stim, give_prev_choice, give_prev_rew = True, True, True
                            continue
                        elif curriculum_t==True and curr_t_progress==len(hp['block_len_n_swit_comb']):    # curriculum learning on block length and trial history completed
                            print('Testing performance reaches criterion after double curriculum training (trial history + block length). Training ends.', flush=True)
                            success = 'success'
                            break


        if ba>=1e5:    
            print('Does not converge after too many batches. Training ends.', flush=True)
            success = 'noConverge'
            break
        elif loss==np.inf:
            success = 'gradExplode'
            print('gradient explodes! training ends', flush=True)
            if hp['check_explode_cg']==True:
                print('cell groups that explode: {}'.format(explode_cgs))
            break
#         elif time.time()-start>=3600*47:    # when the running time is over (try using checkpoint and continuing the job)
#             success = 'timesup'
#             print('Time is up!')
#             break
        
        
    # save at the end of training       
    print('saving model. name={}_{} ... \n'.format(success, hp['save_name']))
    saved_file = {'model': model, 'rnn_activity_lastk': rnn_activity_list, 'model_state_dict': model.state_dict(), 'optimizer': optim, 'optim_state_dict': optim.state_dict(), 'hp': hp, 'hp_task': hp_task, 'perf_list':perf_list_big, 'perf_rule_list': perf_rule_list_big, 'loss_list': loss_list_big,'test_perf_list': test_perf_list_big, 'test_perf_rule_list': test_perf_rule_list_big, 'test_loss_list': test_loss_list_big}
#         torch.save(saved_file, '/home/yl4317/Documents/two_module_rnn/saved_models/{}_{}'.format(success, save_name[:150]))  
    torch.save(saved_file, '/scratch/yl4317/two_module_rnn/saved_models/{}_{}'.format(success, hp['save_name'][:150]))     # save a copy in scratch
    

    
#     if success=='success':
#         # TODO: save testing data
#         path_to_pickle_file = '/scratch/yl4317/two_module_rnn/saved_testdata/' + hp['save_name'][:150] + '_testdata_noiseless_no_current_matrix'
#         model.rnn.network_noise = 0
#         hp['input_noise_perceptual'] = 0
#         hp['input_noise_rule'] = 0
#         test_data = generate_neural_data_test(model=model, n_trials_test=100, switch_every_test=10, batch_size=10, to_plot=False, hp_test=hp, hp_task_test=hp_task, compute_current=False, random_switch=True, n_switches=10, concat_activity=False)
#         with open(path_to_pickle_file, 'wb') as f:
#             pickle.dump(test_data, f)
        


    
    if plot==True:
        fig = plt.figure()
        plt.plot(perf_list_big, label='perf')
        plt.plot(perf_rule_list_big, label='perf_rule')
        plt.plot(loss_list_big, label='loss')
        plt.xlabel('Step # (x100)')
        plt.ylabel('Perf')
        plt.legend()
        plt.show()

        fig = plt.figure()
        plt.plot(test_perf_list_big, label='perf')
        plt.plot(test_perf_rule_list_big, label='perf_rule')
        plt.plot(test_loss_list_big, label='loss')
        plt.xlabel('Test #')
        plt.ylabel('Perf')
        plt.legend()
        plt.show()
    
#     return times
        

    

    
    
def save_random_models(hp):
    ''' 
    randomly-intialized networks
    
    '''
    
    start = time.time()
    times = {}
    times['forward'] = []
    
    # reset some of the hps 
    for area in ['sr', 'pfc']:
        hp['n_{}_edend'.format(area)] = hp['n_branches'] * hp['n_{}_esoma'.format(area)]
        
    # ensure reproducibility 
    torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms(True) 
    torch.backends.cudnn.deterministic = True 
    torch.manual_seed(hp['torch_seed'])
    np.random.seed(0)
    random.seed(0)
    
    # trace gradient
    torch.autograd.set_detect_anomaly(False)
    
    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}\n'.format(device), flush=True)
    if device=='cpu' and hp['batch_size']>100:
        raise ValueError('batch size too large for CPU!')
    
    # list of rules
    task = hp['task']    # cxtdm, salzman, wcst
    if task=='cxtdm': 
        rule_list = ['color', 'motion']    # for the cxtdm task
    elif task=='salzman':
        rule_list = ['cxt1', 'cxt2']    # for the Fusi&Salzman task
    elif task=='wcst':
        rule_list = ['color', 'shape']
        print('rule_list={}\n'.format(rule_list), flush=True)
    print('\ntask name: {}\n'.format(task))

    # add hook to the parameters to monitor gradients (optional)
    # for name, param in model.named_parameters():  
    #     if param.requires_grad==True:
    #         param.register_hook(lambda grad: print(name, grad))

    # save_name = '{}_init={}_l2h={}_torchseed={}_lr={}_optim={}'.format(hp['jobname'], hp['initialization'], hp['l2_h'], hp['torch_seed'], hp['lr'], hp['optimizer'].__name__)
    hp['save_name'] = hp['jobname']
    
    # if the model has reached convergence, exit
    saved_models = [f for f in os.listdir('/scratch/yl4317/two_module_rnn/saved_models/') if hp['save_name'] in f]
    if len(saved_models)!=0:
        print('found a model: {}, exit'.format(saved_models))
        return
    
    # check if there is a checkpoint
    chkpt_files = [f for f in os.listdir('/scratch/yl4317/two_module_rnn/saved_checkpoints/') if 'chkpt' in f and hp['save_name'] in f]
    if len(chkpt_files)==0:
        # no checkpoint
        print('no checkpoint, start a new training\n')
        ba_start = 0
        start_time = 0
        if hp['task']=='cxtdm':
            hp_task = get_default_hp_cxtdm() 
        elif hp['task']=='wcst':
            hp_task = get_default_hp_wcst()
        else:
            raise ValueError('{} task not implemented!'.format(hp['task']))
        model = Net_readoutSR_working(hp)
        model.to(device); model.rnn.to(device)    
       # model = SimpleNet_readoutSR(hp)    # simplified net
        if hp['optimizer']=='adam':  
            optimizer = torch.optim.Adam 
        elif hp['optimizer']=='SGD':  
            optimizer = torch.optim.SGD 
        elif hp['optimizer']=='Rprop':  
            optimizer = torch.optim.Rprop 
        elif hp['optimizer']=='RMSprop':  
            optimizer = torch.optim.RMSprop 
        else:
            raise NotImplementedError   # Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD
        optim = optimizer(params=model.parameters(), lr=hp['learning_rate'])    # instantiate the optimizer. amsgrad may help with convergence (test?)  
    elif len(chkpt_files)==1:
        # found a checkpoint
#         if torch.cuda.is_available():
#             checkpoint = torch.load('/scratch/yl4317/two_module_rnn/saved_checkpoints/'+chkpt_files[0], map_location=device)
#         else:
        checkpoint = torch.load('/scratch/yl4317/two_module_rnn/saved_checkpoints/'+chkpt_files[0], map_location=device)
#         checkpoint = chkpt_files[0]
        ba_start = int(checkpoint['step'])
        start_time = checkpoint['time']    # how long it has take
        hp_task = checkpoint['hp_task']
#         with HiddenPrints():
        model = Net_readoutSR_working(hp)
        model.to(device); model.rnn.to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if hp['optimizer']=='adam':  
            optimizer = torch.optim.Adam 
        elif hp['optimizer']=='SGD':  
            optimizer = torch.optim.SGD 
        elif hp['optimizer']=='Rprop':  
            optimizer = torch.optim.Rprop 
        elif hp['optimizer']=='RMSprop':  
            optimizer = torch.optim.RMSprop 
        else:
            raise NotImplementedError   # Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD
        optim = optimizer(params=model.parameters(), lr=hp['learning_rate'])    # instantiate the optimizer here
        optim.load_state_dict(checkpoint['optim_state_dict'])
#         # debug: move the optimizer to CUDA
#         for state in optimizer.state.values():
#             for k, v in state.items():
#                 if isinstance(v, torch.Tensor):
#                 if torch.is_tensor(v):
# #                     state[k] = v.cuda()
#                     state[k] = v.to(device)
        print('starting from checkpoint (step {})\n'.format(ba_start))
    elif len(chkpt_files)>1:
        raise ValueError('found more than 1 checkpoints: {}'.format(chkpt_files))
        
    # define loss function
#     _, _, loss_fnc = get_default_hp()
    if hp['loss_type']=='mse':
        loss_fnc = nn.MSELoss()
    else:
        raise NotImplementedError

    
    # delay variability
    delay_var = 200    # variability in the delay. 200 ms
    hp_task['delay_var'] = delay_var

    # print some hps (optional)
    for key in hp.keys():
        if key in ['mglur', 'divide_sr_sst_vip', 'sr_sst_high_bias', 'no_pfcesoma_to_srsst',
                  'no_pfcesoma_to_sredend', 'no_pfcesoma_to_srpv', 'no_pfcesoma_to_srvip',
                  'fdbk_to_vip', 'grad_remove_history', 'trainable_dend2soma',
                  'divisive_dend_inh', 'divisive_dend_ei', 'scale_down_init_wexc'] and hp[key]==True:
            print('{}={}\n'.format(key, hp[key]))
        if key in 'dend_nonlinearity':
            print('dendritic nonlinearity = {}\n'.format(hp[key]))
        if key in 'activation':
            print('somatic nonlinearity = {}\n'.format(hp[key]))
            
    # print some more
    print('Hyperparameters:\n{}\n'.format(hp), flush=True)
    print('Hyperparameters for the task:\n{}\n'.format(hp_task), flush=True)
    print(optim, flush=True)
    print('\n')
    print(model, flush=True)
    print('\n')

    # display connectivity
#     display_connectivity(model, plot=False)

    
    #============================START===========================#
    plot=hp['plot_during_training']      # whether to plot during training 
    
    if len(chkpt_files)==0:
        perf_list_big = []
        perf_rule_list_big = []
        loss_list_big = []
        test_perf_list_big = []
        test_perf_rule_list_big = []
        test_loss_list_big = []
        rnn_activity_list = []

        # resetting network + curriculum learning
        reset_network = False
        give_prev_stim = True
        give_prev_choice = True
        give_prev_rew = True
        curriculum = True    # curriculum learning: gradually remove trial history inputs
        prev_stim_mag = 1
        prev_choice_mag = 1
        prev_rew_mag = 1
        print('Reset network: {}, give_prev_stim: {}, give_prev_choice: {}, give_prev_rew: {}\n'.
              format(reset_network, give_prev_stim, give_prev_choice, give_prev_rew), flush=True)
        curriculum_t = False
        if curriculum_t==True:    # curriculum learning: gradually increase block length
            hp['block_len_n_swit_comb'] = [(20,2)]    # curriculum: gradually increase number of trials
            hp['block_len'], hp['n_switches'] = hp['block_len_n_swit_comb'][0]
            curr_t_progress = 0    # an index that tracks progress through curriculum learning
            print('Curriculum learning in time dimension: Block length = {}. Number of switches = {}\n'.format(hp['block_len'], hp['n_switches']), flush=True)
    elif len(chkpt_files)==1:
        # TODO
        if torch.cuda.is_available():
            checkpoint = torch.load('/scratch/yl4317/two_module_rnn/saved_checkpoints/'+chkpt_files[0])
        else:
            checkpoint = torch.load('/scratch/yl4317/two_module_rnn/saved_checkpoints/'+chkpt_files[0], map_location=torch.device('cpu'))
#         checkpoint = chkpt_files[0]
        perf_list_big = checkpoint['perf_list']
        perf_rule_list_big = checkpoint['perf_rule_list']
        loss_list_big = checkpoint['loss_list']
        test_perf_list_big = checkpoint['test_perf_list']
        test_perf_rule_list_big = checkpoint['test_perf_rule_list']
        test_loss_list_big = checkpoint['test_loss_list']
        rnn_activity_list = []

        # resetting network + curriculum learning
        reset_network = checkpoint['reset_network']    # should be False
        give_prev_stim = checkpoint['give_prev_stim']
        give_prev_choice = checkpoint['give_prev_choice']
        give_prev_rew = checkpoint['give_prev_rew']
        curriculum = checkpoint['curriculum']    # curriculum learning: gradually remove trial history inputs
        prev_stim_mag = checkpoint['prev_stim_mag']
        prev_choice_mag = checkpoint['prev_choice_mag']
        prev_rew_mag = checkpoint['prev_rew_mag']
        print('Reset network: {}, give_prev_stim: {}, give_prev_choice: {}, give_prev_rew: {}\n'.
              format(reset_network, give_prev_stim, give_prev_choice, give_prev_rew), flush=True)
        curriculum_t = [checkpoint['curriculum_t'] if 'curriculum_t' in checkpoint.keys() else False]
        if curriculum_t==True:    # curriculum learning: gradually increase block length (not used anymore)
            raise NotImplementedError
    elif len(chkpt_files)>1:
        raise ValueError('found more than 1 checkpoints: {}'.format(chkpt_files))


    #=== start training ===#
    for ba in range(ba_start, int(1e10)): 
        start_step = time.time()
        
#         current_rule = random.choice(rule_list)    # randomly choose a rule to start
        current_rule = rule_list[0]    # use a fixed rule
        perf_list = []
        perf_rule_list = []
        rnn_activity_over_trials = []
        loss = 0

        block_len = hp['block_len'] + random.randint(0,0)    # test: variable block length
        switches = random.sample(range(block_len-1), hp['n_switches'])

        if hp['grad_remove_history']==True:
            if give_prev_stim==False:
#                 prev_stim_mag -= 1e-3    # gradually remove it
                prev_stim_mag = 0     # suddenly remove it
                prev_stim_mag = max(prev_stim_mag, 0)
            if give_prev_rew==False:
#                 prev_rew_mag -= 1e-3    # gradually remove it
                prev_rew_mag = 0
                prev_rew_mag =  max(prev_rew_mag, 0)
            if give_prev_choice==False:
#                 prev_choice_mag -= 1e-3    # gradually remove it
                prev_choice_mag = 0
                prev_choice_mag = max(prev_choice_mag, 0)

        for tr in range(block_len):
#             print('tr {}'.format(tr), flush=True)
            # compute the trial history
            if tr==0:
                h_init = None
                i_me_init = None
                last_rew = None
                prev_stim = None
                prev_choice = None
            else:
                if reset_network==False:
                    if hp['bpx1tr']==True:
                        h_init = h_last.detach()    # for backprop after each trial
                        i_me_init = i_me_last.detach()
                    else:
                        h_init = h_last 
                        i_me_init = i_me_last
                else:
                    h_init = None
                    i_me_init = None
                if give_prev_rew==True or hp['grad_remove_history']==True:
                    last_rew = perf.detach()    # detach it from the graph
                else:
                    last_rew = None
                if give_prev_stim==True or hp['grad_remove_history']==True:
                    prev_stim = _x.detach()    # detach it from the graph
                else:
                    prev_stim = None
                if give_prev_choice==True or hp['grad_remove_history']==True:
                    prev_choice = choice.detach()    # detach it from the graph
                else:
                    prev_choice = None



            # implement variable trial duration
            hp_task_var_delay = copy.deepcopy(hp_task)
            hp_task_var_delay['resp_start'] = hp_task['resp_start'] + np.random.uniform(low=-delay_var, high=delay_var)    # adjust this 
            hp_task_var_delay['resp_end'] = hp_task_var_delay['resp_start'] + (hp_task['resp_end']-hp_task['resp_start'])    # resp duration remains the same
            hp_task_var_delay['trial_end'] = hp_task_var_delay['resp_end']
            if hp['task']=='wcst':
                hp_task_var_delay['test_cards_on'] = hp_task_var_delay['resp_start']
                hp_task_var_delay['test_cards_off'] = hp_task_var_delay['resp_end']
            if hp['resp_cue']==True:
                hp_task_var_delay['resp_cue_start'] = hp_task_var_delay['resp_start']    # change the start of the response cue
                hp_task_var_delay['resp_cue_end'] = hp_task_var_delay['resp_cue_start'] + (hp_task['resp_cue_end']-hp_task['resp_cue_start'])     # keep the duration of the response cue the same 


            # compute the trial history current
            start_trialhistory = time.time()
            input_period = np.arange(int(hp_task_var_delay['trial_history_start']/hp['dt']), int(hp_task_var_delay['trial_history_end']/hp['dt']))    # input period for the trial history info
            n_steps = (hp_task_var_delay['trial_end'] - hp_task_var_delay['trial_start'])//hp['dt']
            n_steps = int(n_steps)
            if hp['task']=='cxtdm':
                ts_prev_stim_start = hp_task_var_delay['stim_start']
                ts_prev_stim_end = hp_task_var_delay['stim_end']
                I_prev_rew, I_prev_stim, I_prev_choice = compute_trial_history(last_rew=last_rew, prev_stim=prev_stim, prev_choice=prev_choice, input_period=input_period, batch_size=hp['batch_size'], n_steps=n_steps, input_dim=model.rnn.n['input'], stim_start=ts_prev_stim_start, stim_end=ts_prev_stim_end, dt=hp['dt'], choice_dim=2)    # each current is time*batch*feature
            if hp['task']=='wcst':
                ts_prev_stim_start = hp_task_var_delay['test_cards_on']    # here the center card is still shown
                ts_prev_stim_end = hp_task_var_delay['test_cards_off']
                I_prev_rew, I_prev_stim, I_prev_choice = compute_trial_history(last_rew=last_rew, prev_stim=prev_stim, prev_choice=prev_choice, input_period=input_period, batch_size=hp['batch_size'], n_steps=n_steps, input_dim=model.rnn.n['input'], stim_start=ts_prev_stim_start, stim_end=ts_prev_stim_end, dt=hp['dt'], choice_dim=3)    # each current is time*batch*feature
            I_prev_rew, I_prev_stim, I_prev_choice = I_prev_rew.to(device), I_prev_stim.to(device), I_prev_choice.to(device)
            trial_history = {'i_prev_rew': prev_rew_mag*I_prev_rew, 'i_prev_choice': prev_choice_mag*I_prev_choice, 'i_prev_stim': prev_stim_mag*I_prev_stim}
#             print('compute trial history input takes {:.2e}s, total time for this step = {:.2e}s'.format(time.time()-start_trialhistory, time.time()-start_step))
    
#                 # test: plot the history currents
#                 fig, ax=plt.subplots()
#                 ax.set_title('I_prev_rew')
#                 for i in range(I_prev_rew.shape[-1]):
#                     ax.plot(I_prev_rew[:,0,i])
#                 ax.set_ylim([-0.1, 1.1])
#                 fig, ax=plt.subplots()
#                 ax.set_title('I_prev_stim')
#                 for i in range(I_prev_stim.shape[-1]):
#                     ax.plot(I_prev_stim[:,0,i])
#                 ax.set_ylim([-0.1, 1.1])
#                 fig, ax=plt.subplots()
#                 ax.set_title('I_prev_choice')
#                 for i in range(I_prev_choice.shape[-1]):
#                     ax.plot(I_prev_choice[:,0,i])
#                 ax.set_ylim([-0.1, 1.1])
#                 plt.show()


            # generate input and target for 1 trial
            start_gendata = time.time()
            # fusi task
            if task=='salzman':
                _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_fusi(n_trials=hp['batch_size'], rule=current_rule, hp=hp, hp_fusi=hp_task_var_delay)
            # cxtdm task
            elif task=='cxtdm':
                if tr-1 in switches or tr==0:
                    _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_siegel(n_trials=hp['batch_size'], rule=current_rule, hp=hp, hp_cxtdm=hp_task_var_delay, trial_type='incongruent')    # such that the first trial after switch is always incongruent
                else:
                    _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_siegel(n_trials=hp['batch_size'], rule=current_rule, hp=hp, hp_cxtdm=hp_task_var_delay, trial_type='no_constraint')
            # wisconsin card sorting task
            elif task=='wcst':
                wcst = WCST(hp=hp, hp_wcst=hp_task_var_delay, rule=current_rule, rule_list=rule_list, n_features_per_rule=2, n_test_cards=3)
                _x, _x_rule, _yhat, _yhat_rule, task_data = wcst.make_task_batch(batch_size=hp['batch_size'])
            _x, _x_rule, _yhat, _yhat_rule = _x.to(device), _x_rule.to(device), _yhat.to(device), _yhat_rule.to(device)
            if hp['train_rule']==True:
                _yhat_rule = _yhat_rule.to(device)
            if hp['timeit_print']==True:
                print('generate data takes {:.2e}s, total time for this step = {:.2e}s'.format(time.time()-start_gendata, time.time()-start_step))

            # run model forward 1 trial
            start_forward = time.time()
            
            
#             # debug device
#             print('before forward pass')
#             for key in trial_history.keys():
#                 print(ba, tr, key, trial_history[key].device)
#             print(ba, tr, '_x device: {}'.format(_x.device)) 
#             if h_init is None:
#                 print(ba, tr, 'h_init is None')
#             else:
#                 print(ba, tr, 'h_init device: {}'.format(h_init.device))
#             if i_me_init is None:
#                 print(ba, tr, 'i_me_init is None')
#             else:
#                 print(ba, tr, 'i_me_init device: {}'.format(i_me_init.device))
#             for name, param in model.named_parameters():
#                 print(ba, tr, name, param.device)
#             model.to(device); model.rnn.to(device)
#             print('_x shape: {}\n'.format(_x.shape), flush=True)
            out, data = model(input=_x, init={'h': h_init, 'i_me': i_me_init}, trial_history=trial_history, hp=hp)
            if hp['timeit_print']==True:
                print('forward pass takes {:.2e}s, total time for this step = {:.2e}s'.format(time.time()-start_forward, time.time()-start_step))
#                 if ba>=0 and hp['timeit_print']==True:
#                     fptime = time.time()-start_forward
#                     print('forward pass takes {}s'.format(fptime), flush=True)
#                     times['forward'].append(fptime)
            rnn_activity = data['record']['hiddens']
            rnn_activity = torch.stack(rnn_activity, dim=0)
            rnn_activity_over_trials.append(rnn_activity)
            h_last = data['last_states']['hidden']
            i_me_last = data['last_states']['i_me']
            
            # check if some activity is exploding
            if hp['check_explode_cg']==True:
                explode_cgs = []
                for cg in model.rnn.cell_group_list:
                    mean_act_cg = torch.mean(rnn_activity[:,:,model.rnn.cg_idx[cg]])
                    if mean_act_cg>1e1 or torch.isnan(mean_act_cg):
                        print('step {}, tr {}, mean {} activity is {}!!!'.format(ba, tr, cg, mean_act_cg))
                        explode_cgs.append(cg)

            # get the performance
            _y = out['out']
            if hp['task']!='wcst':
                perf, choice_prob, choice = get_perf(y=_y, yhat=_yhat, hp=hp, hp_task=hp_task_var_delay)
#                 print('perf shape: {}'.format(perf.shape), flush=True)
            else:
                perf, choice_prob, choice = wcst.get_perf(y=_y, yhat=_yhat)
            if hp['train_rule']==True:
                _y_rule = out['out_rule']
                if hp['task']=='wcst':
                    perf_rule, _, _ = wcst.get_perf_rule(y_rule=_y_rule, yhat_rule=_yhat_rule)
                else:
                    perf_rule, _, _ = get_perf(y=_y_rule, yhat=_yhat_rule, hp=hp, hp_task=hp_task_var_delay)

            # accumulate loss
            # for debugging: mask our the 1st trial and see if error can backprop to the init state
    #         if tr==0:
    #             loss += 0
    #         else:
    #             loss += loss_fnc(_y, _yhat)
    #             if hp['train_rule']==True:
    #                 loss += loss_fnc(_y_rule, _yhat_rule)
            loss += loss_fnc(_y, _yhat) 
            loss += hp['l1_weight']*torch.norm(model.rnn.w_rec_eff, p=1)/torch.numel(model.rnn.w_rec_eff) + hp['l2_weight']*torch.norm(model.rnn.w_rec_eff, p=2)/np.sqrt(torch.numel(model.rnn.w_rec_eff)) + hp['l1_h']*torch.norm(rnn_activity, p=1)/torch.numel(rnn_activity) + hp['l2_h']*torch.norm(rnn_activity, p=2)/np.sqrt(torch.numel(rnn_activity))    # regularization term for the weights
            if 'sr_esoma' in hp['cell_group_list']:
                loss += hp['l2_h_sr']*torch.norm(rnn_activity[:, :, model.rnn.cg_idx['sr_esoma']], p=2)    # regularization for the activity
            if 'pfc_esoma' in hp['cell_group_list']:
                loss += hp['l2_h_pfc']*torch.norm(rnn_activity[:, :, model.rnn.cg_idx['pfc_esoma']], p=2)    # regularization for the activity
            if 'sr_edend' in hp['cell_group_list']:
                loss += hp['l1_h_sredend']*torch.norm(rnn_activity[:, :, model.rnn.cg_idx['sr_edend']], p=1)    # regularization for the dendritic activity for SR (l1 because the dendritic activity can be negative under the 'old' type dendritic nonlinearity)
    #         loss = loss_fnc(_y, _yhat)    # for backprop after each trial
            if hp['train_rule']==True:
                loss += loss_fnc(_y_rule, _yhat_rule)

            # test: would bp cross 1 trial work?
            if hp['bpx1tr']==True:
                optim.zero_grad()           # clear gradients for this training step
                loss_1tr = loss_fnc(_y, _yhat)
                loss_1tr += hp['l1_weight']*torch.norm(model.rnn.w_rec_eff, p=1)/torch.numel(model.rnn.w_rec_eff) + hp['l2_weight']*torch.norm(model.rnn.w_rec_eff, p=2)/np.sqrt(torch.numel(model.rnn.w_rec_eff)) + hp['l1_h']*torch.norm(rnn_activity, p=1)/torch.numel(rnn_activity) + hp['l2_h']*torch.norm(rnn_activity, p=2)/np.sqrt(torch.numel(rnn_activity))    # regularization term for the weights
                if 'sr_esoma' in hp['cell_group_list']:
                    loss_1tr += hp['l2_h_sr']*torch.norm(rnn_activity[:, :, model.rnn.cg_idx['sr_esoma']], p=2)    # regularization for the activity
                if 'pfc_esoma' in hp['cell_group_list']:
                    loss_1tr += hp['l2_h_pfc']*torch.norm(rnn_activity[:, :, model.rnn.cg_idx['pfc_esoma']], p=2)    # regularization for the activity
                if hp['train_rule']==True:
                    loss_1tr += loss_fnc(_y_rule, _yhat_rule)
                loss_1tr.backward(retain_graph=False)           # backpropagation, compute gradients
                loss_1tr = np.nan_to_num(loss_1tr.detach().cpu().numpy(), nan=np.inf)    # convert nan to inf in case loss blows up
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, error_if_nonfinite=False)    # gradient clip
                optim.step()
                
            # concatenate x and yhat across trials
#                 if tr==0:
#                     x = _x
#                     yhat = _yhat
#                     yhat_rule = _yhat_rule
#                     y = _y
#                     y_rule = _y_rule
#                 else:
#                     x = torch.cat((x, _x), axis=0)
#                     print(yhat.shape, _yhat.shape)
#                     yhat = torch.cat((yhat, _yhat), axis=2)
#                     y = torch.cat((y, _y), axis=2)
#                     if hp['train_rule']==True:
#                         yhat_rule = torch.cat((yhat_rule, _yhat_rule), axis=2)
#                         y_rule = torch.cat((y_rule, _y_rule), axis=2)           

            # collect perf across trials
            perf_list.append(torch.mean(perf.float().detach()).cpu())
            if hp['train_rule']==True:
                perf_rule_list.append(torch.mean(perf_rule.float().detach()).cpu())
            else:
                perf_rule_list.append(-1)

            # switch rule if necessary
            if tr in switches:
                if len(rule_list)==1:
                    next_rule = current_rule
                    print('only 1 rule in rule_list, not switching\n', flush=True)
                else:
                    next_rule = random.choice([r for r in rule_list if r!=current_rule])    # randomly switch to a different rule
                current_rule = next_rule

        # backprop at the end of block 
        if hp['bpx1tr']==False:
            start_backward = time.time()
            optim.zero_grad()           # clear gradients for this training step
            loss = loss/block_len
            loss.backward(retain_graph=True)           # backpropagation, compute gradients
#             print('loss device: {}'.format(loss.device))    # debug
            loss = np.nan_to_num(loss.detach().cpu().numpy(), nan=np.inf)    # convert nan to inf in case loss blows up
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, error_if_nonfinite=False)    # gradient clip
            # monitor average activity
            if grad_norm>=10:
                print('\nstep {}, large gradient norm encountered ({:0.4f}).\nchecking grad & avg activity...'.format(ba, grad_norm))
                for name, param in model.named_parameters():
                    if param.requires_grad==True:
                        if param.grad is not None:    # the param is part of the graph
                            print('{}, norm of grad: {:0.4f}, max grad: {:0.4f}'.format(name, torch.norm(param.grad.detach(), p=2), torch.max(param.grad.detach())))
    #             if torch.isnan(grad_norm)==False:
    
    
#             print('before optim.step()')
#             for name, param in model.named_parameters():   # debug
#                 print(ba, name, param.device)
#             print(model.rnn.w_rec.device, (prev_stim_mag*I_prev_stim).device, _x.device, _yhat.device, flush=True)    # debug
            
            
            optim.step()                # apply gradients when gradient is finite
            if hp['timeit_print']==True:
    #         if True:
                print('***backward pass takes {:.2e}s, total time for this step = {:.2e}s'.format(time.time()-start_backward, time.time()-start_step), flush=True)

        
        # record the recent network activity
        if ba%1==0 and hp['record_recent_rnn_activity']==True:
            min_tstep = (hp_task['trial_end']-delay_var)//hp['dt']    # since trial length is variable
            rnn_activity_over_trials_cut = [_[:min_tstep,:,:] for _ in rnn_activity_over_trials]    # cut to the minimum duration of a trial (since there is variable delay)
            rnn_activity_concat = concat_trials(torch.stack(rnn_activity_over_trials_cut))
            rnn_activity_list.append(rnn_activity_concat)    # renp.mean(data[i, model.rnn.cg_idx[cg],:], axis=0)h_initcord the activity
            if len(rnn_activity_list)>3:
                rnn_activity_list.pop(0)    # only record the last k steps to save memory

#                 for cg in model.rnn.cell_group_list:
#                     if (rnn_activity_concat[:, model.rnn.cg_idx[cg], :]>=10).any() or torch.isnan(rnn_activity_concat[:, model.rnn.cg_idx[cg], :]).any():
#                         print('step {}, {}, average activity over time in a block: {}\n'.format(ba, cg, torch.mean(rnn_activity_concat[:, model.rnn.cg_idx[cg], :], dim=(0,1))))


        # collect and print  
        if ba%1==0:
            perf_list_big.append(np.mean(perf_list))
            perf_rule_list_big.append(np.mean(perf_rule_list))
            loss_list_big.append(loss)

        if ba%10==0:
#                 print('Step {}, dend2soma={}'.format(ba, model.rnn.dend2soma.data))
#                 print('Step {}, SST bias {}'.format(ba, model.rnn.bias[model.rnn.cg_idx['sr_sst']]))
            print('Step {}, total loss={:0.4f}, perf={:0.4f}, perf rule={:0.4f}, time={:0.2f}s, ps/c/r={}/{}/{}, bl_len={}, swit={}'
                  .format(ba, loss, np.mean(perf_list), np.mean(perf_rule_list), time.time()-start+start_time, give_prev_stim, give_prev_choice, give_prev_rew, 
                          hp['block_len'], switches), flush=True)
            print('prev stim/choice/rew mag: {}/{}/{}\n'.format(prev_stim_mag, prev_choice_mag, prev_rew_mag))
    #         print('h0={}'.format(model.rnn.h0[0:3,0:3]))

        if ba%1000==0 and ba>0:
            print('torch seed = {}\n'.format(hp['torch_seed']))
            print('reset network: {}, give_prev_stim: {}, give_prev_choice: {}, give_prev_rew: {}\n'.format(reset_network, give_prev_stim, give_prev_choice, give_prev_rew), flush=True)
            print('Block length = {}. Number of switches = {}\n'.format(hp['block_len'], hp['n_switches']), flush=True)
            for key in hp.keys():
                print('{}: {}'.format(key, hp[key]))
#             print(hp)
            print('\n')
            print(optim)
            print('\n')
            print(hp_task)
            print('\n')
#             display_connectivity(model=model, plot=False)

            print(switches, flush=True)
            if plot==True:
                plot_y_yhat(y.cpu(), yhat.cpu())
                if hp['train_rule']==True:
                    plot_y_yhat(y_rule.cpu(), yhat_rule.cpu())
                plot_perf(perf_list)
                plot_perf(perf_rule_list, title='performance (rule)', ylabel='performance (rule)')
                
        # save checkpoint
        if ba%10==0 and ba>0:
            print('saving checkpoint. name={}_{}'.format('chkpt', hp['save_name']))
            saved_file = saved_file = {'step': ba, 'time': time.time()-start+start_time, 'model': model, 'model_state_dict': model.state_dict(), 'optimizer': optim, 'optim_state_dict': optim.state_dict(), 'hp': hp, 'hp_task': hp_task, 'perf_list':perf_list_big, 'perf_rule_list': perf_rule_list_big, 'loss_list': loss_list_big,'test_perf_list': test_perf_list_big, 'test_perf_rule_list': test_perf_rule_list_big, 'test_loss_list': test_loss_list_big, 'reset_network': reset_network, 'give_prev_stim': give_prev_stim, 'give_prev_choice': give_prev_choice, 'give_prev_rew': give_prev_rew, 'curriculum': curriculum, 'prev_stim_mag': prev_stim_mag, 'prev_choice_mag': prev_choice_mag, 'prev_rew_mag': prev_rew_mag}
            torch.save(saved_file, '/scratch/yl4317/two_module_rnn/saved_checkpoints/{}_{}'.format('chkpt', hp['save_name'][:150]))     # save a copy in scratch

            
        # testing
        perf_crit_train = 1-hp['n_switches']/hp['block_len']-0.2    # performance criteria
#         perf_crit_train = 0    # performance criteria (for debug)
        if ba%1==0 and ba>10:
            mean_recent_perf = np.mean(perf_list_big[-1:])
            mean_recent_perf_rule = np.mean(perf_rule_list_big[-1:])            
            print('\nmean recent perf: {:0.6f}/{:0.6f}. criteria for begin testing: {:0.4f}. max possible perf: {:0.4f}\n'.format(mean_recent_perf, mean_recent_perf_rule, perf_crit_train, 1-hp['n_switches']/hp['block_len']))
#             if np.mean(perf_list_big[-10:])>= perf_crit_train and np.mean(perf_rule_list_big[-10:])>=perf_crit_train:
            if mean_recent_perf>= perf_crit_train and mean_recent_perf_rule>=perf_crit_train:
#             if True:    # debug
                print('\nStart testing...........', flush=True)
                n_trials_test = 200
                switch_every_test = 20
                max_perf_test = 1-1/switch_every_test
                if hp['n_switches']<2:
                    print('\n<2 switches during training. Therefore skip testing\n', flush=True)
                    perf_test = 1
                    perf_rule_test = 1
                    loss_test = 0
                else:
                    init_rule = random.choice(rule_list)
                    print('init_rule={}'.format(init_rule))
                    perf_test, perf_rule_test, loss_test, _ = test_frozen_weights(model=model, 
                                                                                  n_trials_test=n_trials_test, 
                                                                                  switch_every_test=switch_every_test, 
                                                                                  init_rule=init_rule,
                                                                                  task=task,
                                                                                  hp=hp, 
                                                                                  hp_task=hp_task, 
                                                                                  loss_fnc=loss_fnc, 
                                                                                  delay_var = 200,
                                                                                  give_prev_rew=give_prev_rew, 
                                                                                  give_prev_stim=give_prev_stim, 
                                                                                  give_prev_choice=give_prev_choice,
                                                                                  noiseless=False)
#                     _, _, _, test_data = test_frozen_weights(model=model, n_trials_test=n_trials_test, switch_every_test=switch_every_test, 
#                                              init_rule=random.choice(rule_list), hp=hp_test, task=task,
#                                              loss_fnc=nn.MSELoss(), hp_task=hp_task_test,
#                                              delay_var=0, 
#                                              give_prev_choice=True, give_prev_stim=True, give_prev_rew=True, plot=to_plot, 
#                                              random_switch=random_switch, n_switches=n_switches)
                test_perf_list_big.append(perf_test)
                test_perf_rule_list_big.append(perf_rule_test)
                test_loss_list_big.append(loss_test)
                
                
                print('test perf: {}/{}'.format(perf_test, perf_rule_test, flush=True))
                criteria_perf = max_perf_test - 0.15
                print('max_perf_test = {}, criteria for continuing to next stage of curriculum learning: {}'.format(max_perf_test, criteria_perf), flush=True)



                # curriculum learning
#                 if np.mean(test_perf_list_big[-5:])>=max_perf_test-0.05 and np.mean(test_perf_rule_list_big[-5:])>=max_perf_test-0.05:
                if np.mean(test_perf_list_big[-1:])>=criteria_perf and np.mean(test_perf_rule_list_big[-1:])>=criteria_perf:
                    print('recent testing performance meets criteria ({})\n'.format(criteria_perf), flush=True)
                    if curriculum==False:
                        if curriculum_t==False:     # no curriculum learning
                            print('Testing performance reaches criterion after no curriculum learning. Training ends.', flush=True)
                            success = 'success'
                            break
                        elif curriculum_t==True and curr_t_progress!=len(hp['block_len_n_swit_comb']):    # curriculum learning on block length, progress +1
                            curr_t_progress += 1
                            hp['block_len'], hp['n_switches'] = hp['block_len_n_swit_comb'][curr_t_progress]
                            max_perf_train = 1-hp['n_switches']/hp['block_len']
                            print('*** Increase block length. New block length = {} trials. New n_swtiches = {} ***'.format(hp['block_len'], hp['n_switches']), flush=True)
                            give_prev_stim, give_prev_choice, give_prev_rew = True, True, True
                            continue
                        elif curriculum_t==True and curr_t_progress==len(hp['block_len_n_swit_comb']):    # curriculum learning on block length completed
                            print('Testing performance reaches criterion after curriculum training on block length. Training ends.', flush=True)
                            success = 'success'
                            break
                    elif curriculum==True:
                        if give_prev_stim==True:    # curriculum learning on trial history, progress +1
                            give_prev_stim = False
                            print('******** Now not giving previous stimulus ********\n', flush=True)
                            optim = optimizer(params=model.parameters(), lr=hp['learning_rate'])    # re-initialize the optimizer?
                            continue
                        elif give_prev_choice==True and prev_stim_mag==0:    # curriculum learning on trial history, progress +1
                            give_prev_choice = False
                            print('******** Now not giving previous choice ********\n', flush=True)
                            optim = optimizer(params=model.parameters(), lr=hp['learning_rate'])    # re-initialize the optimizer?
                            continue
                        else:
                            print('prev_choice_mag={}'.format(prev_choice_mag))
                            print('curriculum_t={}'.format(curriculum_t))
                            if curriculum_t==False and prev_choice_mag==0:    # curriculum learning on trial history completed
                                print('Testing performance reaches criterion after curriculum training on trial history. Training ends.', flush=True)
                                success = 'success'
                                break
                            elif curriculum_t==True and curr_t_progress!=len(hp['block_len_n_swit_comb']):    # curriculum learning on block length, progress +1
                                curr_t_progress += 1
                                hp['block_len'], hp['n_switches'] = hp['block_len_n_swit_comb'][curr_t_progress]
                                max_perf_train = 1-hp['n_switches']/hp['block_len']
                                print('*** Increase block length. New block length = {} trials. New n_swtiches = {} ***'.format(hp['block_len'], hp['n_switches']), flush=True)
                                give_prev_stim, give_prev_choice, give_prev_rew = True, True, True
                                continue
                            elif curriculum_t==True and curr_t_progress==len(hp['block_len_n_swit_comb']):    # curriculum learning on block length and trial history completed
                                print('Testing performance reaches criterion after double curriculum training (trial history + block length). Training ends.', flush=True)
                                success = 'success'
                                break


        if ba>=1e5:    
            print('Does not converge after too many batches. Training ends.', flush=True)
            success = 'noConverge'
            break
        elif loss==np.inf:
            success = 'gradExplode'
            print('gradient explodes! training ends', flush=True)
            if hp['check_explode_cg']==True:
                print('cell groups that explode: {}'.format(explode_cgs))
            break
#         elif time.time()-start>=3600*47:    # when the running time is over (try using checkpoint and continuing the job)
#             success = 'timesup'
#             print('Time is up!')
#             break
        
        
    # save at the end of training       
    print('saving model. name={}_{} ... \n'.format('success', hp['save_name']))
    saved_file = {'model': model, 'rnn_activity_lastk': rnn_activity_list, 'model_state_dict': model.state_dict(), 'optimizer': optim, 'optim_state_dict': optim.state_dict(), 'hp': hp, 'hp_task': hp_task, 'perf_list':perf_list_big, 'perf_rule_list': perf_rule_list_big, 'loss_list': loss_list_big,'test_perf_list': test_perf_list_big, 'test_perf_rule_list': test_perf_rule_list_big, 'test_loss_list': test_loss_list_big}
#         torch.save(saved_file, '/home/yl4317/Documents/two_module_rnn/saved_models/{}_{}'.format(success, save_name[:150]))  
    torch.save(saved_file, '/scratch/yl4317/two_module_rnn/saved_models/{}_{}'.format('success', hp['save_name'][:150]))     # save a copy in scratch
    

    
    if True:
        # TODO: save testing data
        path_to_pickle_file = '/scratch/yl4317/two_module_rnn/saved_testdata/' + hp['save_name'][:150] + '_testdata_noiseless_no_current_matrix'
        model.rnn.network_noise = 0
        hp['input_noise_perceptual'] = 0
        hp['input_noise_rule'] = 0
        test_data = generate_neural_data_test(model=model, n_trials_test=100, switch_every_test=10, batch_size=10, to_plot=False, hp_test=hp, hp_task_test=hp_task, compute_current=False, random_switch=True, n_switches=10, concat_activity=False)
        with open(path_to_pickle_file, 'wb') as f:
            pickle.dump(test_data, f)
        


    
    if plot==True:
        fig = plt.figure()
        plt.plot(perf_list_big, label='perf')
        plt.plot(perf_rule_list_big, label='perf_rule')
        plt.plot(loss_list_big, label='loss')
        plt.xlabel('Step # (x100)')
        plt.ylabel('Perf')
        plt.legend()
        plt.show()

        fig = plt.figure()
        plt.plot(test_perf_list_big, label='perf')
        plt.plot(test_perf_rule_list_big, label='perf_rule')
        plt.plot(test_loss_list_big, label='loss')
        plt.xlabel('Test #')
        plt.ylabel('Perf')
        plt.legend()
        plt.show()
    
#     return times
        