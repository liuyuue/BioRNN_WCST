import subprocess
import itertools
import os
import sys
sys.path.append("/home/yl4317/Documents/two_module_rnn/code")
os.chdir('/home/yl4317/Documents/two_module_rnn/code')
# from model import *
# from task import *
from functions import *
# from train import * 

#=======================================================
# Submit a job or run it locally
#=======================================================

start_time = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

# This is where all the simulation job requests will be written to files
sbatchpath = '/scratch/yl4317/two_module_rnn/sbatch/'
# sbatchpath = '/scratch/yl4317/two_module_rnn/sbatch/'
# This is where all your screen prints get written to files
scratchpath = '/scratch/yl4317/two_module_rnn/sbatch/'
# scratchpath = '/scratch/yl4317/two_module_rnn/sbatch/'
# This is where your simulation code is
simpath = '/home/yl4317/Documents/two_module_rnn/code'



# Per-job parameters
nodes = 1 # Number of nodes per simulation
ppn = 4 # Number of cores you per simulation
mem = 32 # max GB's of memory desired per simulation
nhours = 48 # max time the simulation should run
nminutes = 0

task = 'cxtdm'

divisive_dend_inhs = [False]
divisive_dend_eis = [False]
divisive_dend_nonlinears = [True, False]
dend_nonlinearitys = ['old', 'v2', 'v3', 'step']
divide_sr_sst_vips = [True]
no_pfcesoma_to_sredends = [True]
no_pfcesoma_to_srpvs = [True]
no_pfcesoma_to_srssts = [True]
structured_sr_sst_to_sr_edends = [False]
activations = ['relu', 'relu_satu']
structured_sr_sst_to_sr_edend_branch_specifics = [True, False]
sparse_pfcesoma_to_srvips = [0.2, 1]


allParameterDict = {'divisive_dend_inh': divisive_dend_inhs, 
                    'divisive_dend_ei': divisive_dend_eis, 
                    'divisive_dend_nonlinear': divisive_dend_nonlinears
                    'dend_nonlinearity':dend_nonlinearitys, 
                    'divide_sr_sst_vip': divide_sr_sst_vips, 
                    'no_pfcesoma_to_sredend': no_pfcesoma_to_sredends,
                    'no_pfcesoma_to_srpv': no_pfcesoma_to_srpvs,
                    'no_pfcesoma_to_srsst': no_pfcesoma_to_srssts,
                    'structured_sr_sst_to_sr_edend': structured_sr_sst_to_sr_edends,
                    'structured_sr_sst_to_sr_edend_branch_specific': structured_sr_sst_to_sr_edend_branch_specifics,
                    'sparse_pfcesoma_to_srvip': sparse_pfcesoma_to_srvips,
                    'activation': activations}
# print('allParameterDict.items()={}\n'.format(allParameterDict.values()))
parameterCartProd = list(itertools.product(*allParameterDict.values()))
param_name_list = list(allParameterDict.keys())
# print('param_name_list={}\n'.format(param_name_list))



n_jobs = 0
for paramComb in parameterCartProd:
    print('\n')
    print('paramComb={}\n'.format(paramComb))
    for i in range(len(param_name_list)):
        print('{}={}'.format(param_name_list[i], paramComb[i]))
        
    allParameterDict_thisjob = {}
    for i in range(len(paramComb)):
        key = list(allParameterDict.keys())[i]
        allParameterDict_thisjob[key] = paramComb[i]
    print('allParameterDict_thisjob={}\n'.format(allParameterDict_thisjob))
        
    if (allParameterDict_thisjob['divisive_dend_inh']==True and allParameterDict_thisjob['divisive_dend_ei']==True) \
        or ((allParameterDict_thisjob['divisive_dend_inh']==True or allParameterDict_thisjob['divisive_dend_ei']==True) and (allParameterDict_thisjob['dend_nonlinearity']!='old')):
        print('pass\n')
        continue

    n_jobs+=1
    
    # create job name
#     customization=''
#     for key in allParameterDict_thisjob.keys():
#         customization+=(key.replace('_','')+'{}'.format(allParameterDict_thisjob[key])+'_')
    customization = str(n_jobs)+'_'

#     jobname = '{}_{}_{}fulltask_2branches_useReLUforWeff_sronlytopfcpv'.format(start_time, task, customization)
    jobname = '{}_{}_{}cleancode'.format(start_time, task, customization)
    jobname = str(jobname)
    
    cmdd = ""
    for key in allParameterDict_thisjob.keys():
        if type(allParameterDict_thisjob[key])==type(True):
            cmdd += '''hp[''' + '\'' + str(key) + '\'' + '''] = ''' + str(allParameterDict_thisjob[key]) + ''';'''
        elif type(allParameterDict_thisjob[key])==str:
            cmdd += '''hp[''' + '\'' + str(key) + '\'' + '''] = ''' + '\'' + str(allParameterDict_thisjob[key]) + '\'' + ''';'''
    cmd = r'''python3 -c "from train_working import train_bpxtrials_v2_working; from functions import *; hp, _, loss_fnc = get_default_hp();''' \
            + cmdd \
            + '''hp['task'] = 'cxtdm'; print(''' + '\'' + jobname + '\'' + '''); hp['jobname'] = ''' + '\'' + str(jobname) + '\'' + '''; train_bpxtrials_v2_working(''' + 'hp' + ''')"'''
#     print('cmd={}\n\ncmd2={}'.format(cmd, cmd2))
    
    
    
    jobfile = os.path.join(sbatchpath, jobname + '.s')
 
    with open(jobfile, 'w') as f:
        print('submit a new job')
        f.write(
           '#! /bin/bash\n'
           + '\n'
           + '#SBATCH --nodes={}\n'.format(nodes)
           + '#SBATCH --ntasks-per-node=1\n'
           + '#SBATCH --cpus-per-task={}\n'.format(ppn)
           + '#SBATCH --gres=gpu:0\n'    # request GPUs
           + '#SBATCH --mem={}GB\n'.format(mem)
           + '#SBATCH --time={}:{}:00\n'.format(nhours, nminutes)
           + '#SBATCH --job-name={}\n'.format(jobname)
           + '#SBATCH --output={}{}.o\n'.format(scratchpath, jobname)
           + '\n'
           + 'module purge\n'
           # This can be changed to your favorite version of python;
           # You can copy this line to include numpy, etc.
           # You can replace or follow it up by activating your virtual environment if required
           + 'module load python/intel/3.8.6\n'                                  
           + 'cd {}\n'.format(simpath)
           + 'echo "Job starts: $(date)" >> {}{}.log\n'.format(scratchpath, jobname)
           + '{} >> {}{}.log 2>&1\n'.format(cmd, scratchpath, jobname)
           + 'echo "Job ends: $(date)" >> {}{}.log\n'.format(scratchpath, jobname)
           + 'exit 0;\n'
        )
    subprocess.call(['sbatch', jobfile])

    
print('submitted {} jobs in total'.format(n_jobs))