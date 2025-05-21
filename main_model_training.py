#   main_model_training.py

#   This is the main deep learning training script used in the manuscript:
#   "EEG-based Brain-Computer Interface Enables Real-time Robotic Hand Control at Individual Finger Level"
#   
#   Takes in 5 arguments:
#   subj_id: (int) the subject ID, 1-21
#   session_num: (int) the session number, 1-5
#   nclass: (int) number of classes, 2 or 3
#   task: (string) motor imagery or execution, "ME" or "MI"
#   modeltype: (string) pre-training or fine-tuning, "Orig" or "Finetune"
#   
#   Example use: python main_model_training.py 1 1 2 ME Orig
#   Copyright (C) Yidan Ding 2025

# %%

from Functions import load_and_filter_data, generate_paths, train_models

import os
import sys
import numpy as np

# Read command-line arguments
subj_id = int(sys.argv[1])
session_num = int(sys.argv[2])
nclass = int(sys.argv[3])
task = sys.argv[4]
modeltype = sys.argv[5]

# validate inputs
if nclass not in (2, 3):
    raise ValueError("nclass must be either 2 or 3.")

if task not in ("MI", "ME"):
    raise ValueError("task must be either 'MI' (motor imagery) or 'ME' (motor execution).")

if modeltype not in ("Orig", "Finetune"):
    raise ValueError("modeltype must be either 'Orig' (pre-training) or 'Finetune' (fine-tuning).")

# parameters
params = {
    'maxtriallen':5, # in s
    'windowlen':1, # in s
    'block_size': 128, # (samples) same as the online config
    'downsrate': 100, # dawnsampling rate
    'bandpass_filt': [4,40], # (Hz) bandpass filtering
    'nclass': nclass
}
# Specify the paths to data and the models
# data_folder = 'pathToData'
data_folder = '/home/yidand/FingerMovement/code_share/data_example'
save_folder = 'pathToSave'
save_folder = '/home/yidand/FingerMovement/code_share/savedmodels'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

data_paths = generate_paths(subj_id, task, nclass, session_num, model_type = modeltype, data_folder = data_folder)


data, label, params = load_and_filter_data(data_paths, params)

np.savez('/home/yidand/FingerMovement/code_share/my_arrays.npz', array1=data, array2=label)

save_name = os.path.join(save_folder, f'S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_{modeltype}.h5')

if modeltype == 'Finetune':
    params['modelpath'] = save_name.replace('Finetune','Orig') # the pre-trained model to be fine-tuned on
save_name = train_models(data, label, save_name, params)
