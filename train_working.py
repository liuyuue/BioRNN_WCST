import numpy as np; np.set_printoptions(precision=4); np.random.seed(0)
import torch; torch.set_printoptions(sci_mode=False, precision=4)
import torch.nn as nn
import matplotlib.pyplot as plt; plt.rc('font', size=15); plt.rc('font', family='Arial')
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

from task import *
from functions import *
from model_working import *


print(torch.__version__)
print(sys.version)
                
# %matplotlib inline





def train_bpxtrials_v2_working(hp):

    times = {}
    times['forward'] = []
    
    # ensure reproducibility 
    torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms(True) 
    torch.backends.cudnn.deterministic = True    
    np.random.seed(0)
    random.seed(0)
    
    torch.autograd.set_detect_anomaly(False)

    # list of rules
    task = hp['task']    # cxtdm, salzman
    if task=='cxtdm': 
        rule_list = ['color', 'motion']    # for the cxtdm task
    elif task=='salzman':
        rule_list = ['cxt1', 'cxt2']    # for the Fusi&Salzman task

    print('\n\ntask name: {}\n\n'.format(task))


    #============ train a model that operates across trials.==============#
    # train a model that operates across trials.
    start = time.time()
    
    # ensure reproducibility 
    torch.backends.cudnn.benchmark = False
#     torch.use_deterministic_algorithms(True) 
    torch.backends.cudnn.deterministic = True 
    torch.manual_seed(hp['torch_seed'])
    np.random.seed(0)
    random.seed(0)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print('device: {}\n'.format(device), flush=True)

    # customize hp
#     hp['initialization_weights'] = initialization    # customize initialization
#     hp['l2_h'] = l2_h
#     hp['learning_rate'] = lr
#     hp['optimizer'] = optimizer.__name__    # save this change to hp
    if device=='cpu' and hp['batch_size']>100:
        raise ValueError('batch size too large for CPU!')
    delay_var = 200    # variability in the delay. 200 ms

    # customize hp_task
    hp_task = get_default_hp_cxtdm()        

    # edit the name of the model
#         customization=''
    for key in hp.keys():
        if key in ['mglur', 'divide_sr_sst_vip', 'sr_sst_high_bias', 'no_pfcesoma_to_srsst',
                  'no_pfcesoma_to_sredend', 'no_pfcesoma_to_srpv', 'no_pfcesoma_to_srvip',
                  'fdbk_to_vip', 'grad_remove_history', 'trainable_dend2soma',
                  'divisive_dend_inh', 'divisive_dend_ei', 'scale_down_init_wexc'] and hp[key]==True:
            print('{}={}\n'.format(key, hp[key]))
#                 customization+=(key.replace('_','')+'_')
        if key in 'dend_nonlinearity':
#                 customization+=('dendnonlinear{}_'.format(hp[key]))
            print('dendritic nonlinearity = {}\n'.format(hp[key]))
        if key in 'activation':
            print('somatic nonlinearity = {}\n'.format(hp[key]))
#                 customization+=('somanonlinear{}_'.format(hp[key]))
        
    print('Batch size = {}\n'.format(hp['batch_size']))

#         save_name = '{}_init={}_l2h={}_torchseed={}_lr={}_optim={}'.format(hp['jobname'], hp['initialization'], hp['l2_h'], hp['torch_seed'], hp['lr'], hp['optimizer'].__name__)
    hp['save_name'] = hp['jobname']


    model = Net_readoutSR_working(hp)
    # model = SimpleNet_readoutSR(hp)    # simplified net
    model.to(device); model.rnn.to(device)    

    # add hook to the parameters to monitor gradients
    # for name, param in model.named_parameters():  
    #     if param.requires_grad==True:
    #         param.register_hook(lambda grad: print(name, grad))

    _, _, loss_fnc = get_default_hp()
    
    if hp['optimizer']=='adam':  
        optimizer = torch.optim.Adam
    else:
        raise NotImplementedError   # Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD
    optim = optimizer(params=model.parameters(), lr=hp['learning_rate'])    # instantiate the optimizer. amsgrad may help with convergence (test?)  

#         print('model leak rate (alpha) = {}\n'.format(model.rnn.decay), flush=True)    # set the decay rate smaller
    print('Hyperparameters:\n{}\n'.format(hp), flush=True)
    print('Hyperparameters for the task:\n{}\n'.format(hp_task), flush=True)
    print(optim, flush=True)
    print('\n')
    print(model, flush=True)
    print('\n')

    # display connectivity
    display_connectivity(model, plot=False)



    #============================START===========================#
    perf_list_big = []
    perf_rule_list_big = []
    loss_list_big = []
    test_perf_list_big = []
    test_perf_rule_list_big = []
    test_loss_list_big = []
    rnn_activity_list = []

    plot=hp['plot_during_training']    

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



    for ba in range(int(1e10)): 
        start_step = time.time()
        
        current_rule = random.choice(rule_list)    # randomly choose a rule to start
        perf_list = []
        perf_rule_list = []
        rnn_activity_over_trials = []
        loss = 0

        block_len = hp['block_len'] + random.randint(0,0)    # test: variable block length
        switches = random.sample(range(block_len-1), hp['n_switches'])

        if hp['grad_remove_history']==True:
            if give_prev_stim==False:
                prev_stim_mag -= 1e-3    # gradually remove it
                prev_stim_mag = max(prev_stim_mag, 0)
            if give_prev_rew==False:
                prev_rew_mag -= 1e-3    # gradually remove it
                prev_rew_mag =  max(prev_rew_mag, 0)
            if give_prev_choice==False:
                prev_choice_mag -= 1e-3    # gradually remove it
                prev_choice_mag = max(prev_choice_mag, 0)

        for tr in range(block_len):
            # compute the trial history
            if tr==0:
                h_init = None
                i_me_init = None
                last_rew = None
                prev_stim = None
                prev_choice = None
            else:
                if reset_network==False:
    #                 h_init = h_last.detach()    # for backprop after each trial
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
            hp_task_var_delay['resp_cue_start'] = hp_task_var_delay['resp_start']    # change the start of the response cue
            hp_task_var_delay['resp_cue_end'] = hp_task_var_delay['resp_cue_start'] + (hp_task['resp_cue_end']-hp_task['resp_cue_start'])     # keep the duration of the response cue the same 


            # compute the trial history current
            start_trialhistory = time.time()
            input_period = np.arange(int(hp_task_var_delay['trial_history_start']/hp['dt']), int(hp_task_var_delay['trial_history_end']/hp['dt']))    # input period for the trial history info
            n_steps = (hp_task_var_delay['trial_end'] - hp_task_var_delay['trial_start'])//hp['dt']
            n_steps = int(n_steps)
#             print('n_steps={}'.format(n_steps))
            I_prev_rew, I_prev_stim, I_prev_choice = compute_trial_history(last_rew=last_rew, prev_stim=prev_stim, prev_choice=prev_choice, input_period=input_period, batch_size=model.rnn.batch_size, n_steps=n_steps, input_dim=model.rnn.n['input'], hp_task=hp_task_var_delay, hp=hp)    # each current is time*batch*feature
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
                _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_fusi(n_trials=model.rnn.batch_size, rule=current_rule, hp=hp, hp_fusi=hp_task_var_delay)
            # cxtdm task
            elif task=='cxtdm':
                if tr-1 in switches or tr==0:
                    _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_siegel(n_trials=model.rnn.batch_size, rule=current_rule, hp=hp, hp_cxtdm=hp_task_var_delay, trial_type='incongruent')    # such that the first trial after switch is always incongruent
                else:
                    _x, _x_rule, _yhat, _yhat_rule, task_data = make_task_siegel(n_trials=model.rnn.batch_size, rule=current_rule, hp=hp, hp_cxtdm=hp_task_var_delay, trial_type='no_constraint')
            _x, _yhat = _x.to(device), _yhat.to(device)
            if hp['train_rule']==True:
                _yhat_rule = _yhat_rule.to(device)
            if hp['timeit_print']==True:
                print('generate data takes {:.2e}s, total time for this step = {:.2e}s'.format(time.time()-start_gendata, time.time()-start_step))

            # run model forward 1 trial
            start_forward = time.time()
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
            explode_cgs = []
            for cg in model.rnn.cell_group_list:
                mean_act_cg = torch.mean(rnn_activity[:,:,model.rnn.cg_idx[cg]])
                if mean_act_cg>1e1 or torch.isnan(mean_act_cg):
                    print('step {}, tr {}, mean {} activity is {}!!!'.format(ba, tr, cg, mean_act_cg))
                    explode_cgs.append(cg)

            # get the performance
            _y = out['out']
            perf, choice_prob, choice = get_perf(y=_y, yhat=_yhat, hp=hp, hp_task=hp_task_var_delay)
            if hp['train_rule']==True:
                _y_rule = out['out_rule']
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
            loss += hp['l1_weight']*torch.norm(model.rnn.w_rec_eff, p=1) + hp['l2_weight']*torch.norm(model.rnn.w_rec_eff, p=2)\
                    + hp['l1_h']*torch.norm(rnn_activity, p=1) + hp['l2_h']*torch.norm(rnn_activity, p=2)    # regularization term for the weights
            if 'sr_esoma' in hp['cell_group_list']:
                loss += hp['l2_h_sr']*torch.norm(rnn_activity[:, :, model.rnn.cg_idx['sr_esoma']], p=2)    # regularization for the activity
            if 'pfc_esoma' in hp['cell_group_list']:
                loss += hp['l2_h_pfc']*torch.norm(rnn_activity[:, :, model.rnn.cg_idx['pfc_esoma']], p=2)    # regularization for the activity
    #         loss = loss_fnc(_y, _yhat)    # for backprop after each trial
            if hp['train_rule']==True:
                loss += loss_fnc(_y_rule, _yhat_rule)


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
                next_rule = random.choice([r for r in rule_list if r!=current_rule])    # randomly switch to a different rule
                current_rule = next_rule

        # backprop at the end of block 
        start_backward = time.time()
        optim.zero_grad()           # clear gradients for this training step
        loss = loss/block_len
        loss.backward(retain_graph=True)           # backpropagation, compute gradients
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
        optim.step()                # apply gradients when gradient is finite
        if hp['timeit_print']==True:
#         if True:
            print('***backward pass takes {:.2e}s, total time for this step = {:.2e}s'.format(time.time()-start_backward, time.time()-start_step), flush=True)




        if ba%1==0:
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
        if ba%100==0:
            perf_list_big.append(np.mean(perf_list))
            perf_rule_list_big.append(np.mean(perf_rule_list))
            loss_list_big.append(loss)

        if ba%10==0:
#                 print('Step {}, dend2soma={}'.format(ba, model.rnn.dend2soma.data))
#                 print('Step {}, SST bias {}'.format(ba, model.rnn.bias[model.rnn.cg_idx['sr_sst']]))
            print('Step {}, total loss={:0.4f}, perf={:0.4f}, perf rule={:0.4f}, time={:0.2f}s, ps/c/r={}/{}/{}, bl_len={}, swit={}'
                  .format(ba, loss, np.mean(perf_list), np.mean(perf_rule_list), time.time()-start, give_prev_stim, give_prev_choice, give_prev_rew, 
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


        # testing
        perf_crit_train = 1-hp['n_switches']/hp['block_len']-0.1    # performance criteria
        if ba%100==0 and ba>0:
            print('\nmean recent perf: {:0.6f}/{:0.6f}. criteria: {:0.4f}. max possible perf: {:0.4f}\n'
                  .format(np.mean(perf_list_big[-10:]), np.mean(perf_rule_list_big[-10:]), perf_crit_train, 1-hp['n_switches']/hp['block_len']))
            if np.mean(perf_list_big[-10:])>= perf_crit_train and np.mean(perf_rule_list_big[-10:])>=perf_crit_train:
#                 if True:    # for debugging
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
                test_perf_list_big.append(perf_test)
                test_perf_rule_list_big.append(perf_rule_test)
                test_loss_list_big.append(loss_test)



                # curriculum learning
                if np.mean(test_perf_list_big[-1:])>=max_perf_test-0.05 and np.mean(test_perf_rule_list_big[-1:])>=max_perf_test-0.05:
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
                            continue
                        elif give_prev_choice==True and prev_stim_mag==0:    # curriculum learning on trial history, progress +1
                            give_prev_choice = False
                            print('******** Now not giving previous choice ********\n', flush=True)
                            continue
                        else:
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


        if ba>1e5:    
            print('Does not converge after too many batches. Training ends.', flush=True)
            success = 'noConverge'
            break
        if loss==np.inf:
            success = 'gradExplode'
            print('gradient explodes! training ends', flush=True)
            print('cell groups that explode: {}'.format(explode_cgs))
            break


    print('saving model. name={}_{} ... \n'.format(success, hp['save_name']))
    saved_file = {'model': model, 'rnn_activity_lastk': rnn_activity_list, 'model_state_dict': model.state_dict(), 'optimizer': optim, 'optim_state_dict': optim.state_dict(), 'hp': hp, 'hp_task': hp_task, 'perf_list':perf_list_big, 'perf_rule_list': perf_rule_list_big, 'loss_list': loss_list_big,'test_perf_list': test_perf_list_big, 'test_perf_rule_list': test_perf_rule_list_big, 'test_loss_list': test_loss_list_big, 'torch_seed': hp['torch_seed']}
#         torch.save(saved_file, '/home/yl4317/Documents/two_module_rnn/saved_models/{}_{}'.format(success, save_name[:150]))  
    torch.save(saved_file, '/scratch/yl4317/two_module_rnn/saved_models/{}_{}'.format(success, hp['save_name'][:150]))     # save a copy in scratch


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
        
        
        