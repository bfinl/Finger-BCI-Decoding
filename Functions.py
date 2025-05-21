import numpy as np
import tensorflow as tf

import glob

import scipy
import scipy.io
from scipy.signal import resample

import os

# EEGNet-specific imports
from EEGModels_tf import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K


###### data segmenting and relabeling functions ######
def segment_data(data, labels, segment_size, step_size):
    if segment_size <= 0 or step_size <= 0:
        raise ValueError("segment_size and step_size must be positive.")

    num_trials, num_channels, num_samples = data.shape
    segments = []

    for start in range(0, num_samples - segment_size + 1, step_size):
        end = start + segment_size
        segments.append(data[:, :, start:end])

    segmented_data = np.concatenate(segments, axis=0)
    # repeat labels 
    repeated_labels = np.tile(labels, len(segments))
    trial_indices = range(num_trials)
    repeated_indices = np.tile(trial_indices, len(segments))

    repeated_labels = repeated_labels[~np.isnan(segmented_data).any(axis=(1,2))]
    repeated_indices = repeated_indices[~np.isnan(segmented_data).any(axis=(1,2))]
    segmented_data = segmented_data[~np.isnan(segmented_data).any(axis=(1,2)),:,:]

    return segmented_data, repeated_labels, repeated_indices


def filter_and_relabel(data, label, keep_labels, new_labels):
    filtered_label = label[np.isin(label,keep_labels)]
    filtered_data = data[np.isin(label,keep_labels)]
    filtered_label = np.array([new_labels[l] for l in filtered_label])
    return filtered_data, filtered_label

def generate_paths(subj_id, task, nclass, session_num, model_type, data_folder):
    # get the file paths to the training data
    subject_folder = os.path.join(data_folder, f'S{subj_id:02}')

    if task == 'MI':
        prefix = '*Imagery'
    else:
        prefix = '*Movement'
    
    if model_type == 'Finetune':
        prefix_online = f'{prefix}_Sess{session_num:02}'
        if nclass == 3:
            suffix = f'{nclass}class_Base' # 3-class model is fine-tuned on 3-class same day data
        else:
            suffix = 'Base' # 2-class model is fine-tuned on both 2-class and 3-class same day data
    
        pattern = os.path.join(subject_folder, f'{prefix_online}*{suffix}')
        data_paths = sorted(glob.glob(pattern))
    else:
        # load the offline session data
        offline_pattern = os.path.join(subject_folder, prefix) 
        data_paths = sorted(glob.glob(offline_pattern))

        # load the prior online sessions 
        for session in range(1,session_num):
            prefix_online = f'{prefix}_Sess{session:02}'
            online_pattern = os.path.join(subject_folder, f'{prefix_online}*')
            data_paths.extend(sorted(glob.glob(online_pattern)))
    return data_paths


def load_and_filter_data(data_paths, params):
    label = [] #nTrials
    data = []#nTrials, nChannels, nSamples

    if params['nclass'] == 2:
        keep_labels = [1,4] # thumb; pinky
        new_labels = {1: 1, 4: 2}
    elif params['nclass'] == 3:
        keep_labels = [1,2,4] # thumb; index; pinky
        new_labels = {1: 1, 2: 2, 4: 3}
    else:
        raise ValueError("nclass must be either 2 or 3.")

    for filepath in data_paths:
        for filename in sorted(os.listdir(filepath)):
            cur_data = []

            file_path = os.path.join(filepath, filename)
            print(f"Processing file: {file_path}")

            mat = scipy.io.loadmat(file_path)
            eeg = mat['eeg']
            event = mat['event']

            signals = eeg['data'][0][0]
            params['srate'] = eeg['fsample'][0][0][0][0]
            start_idx, end_idx, target = [], [], []

            # Iterate through events
            for i in range(event.shape[1]):
                evt = event[0, i]
                event_type = evt['type'][0]
                sample = evt['sample'][0][0]
                value = evt['value'][0][0]

                if event_type == 'Target':
                    start_idx.append(sample-1) # 0-index
                    target.append(value)
                elif event_type == 'TrialEnd':
                    end_idx.append(sample-1) # 0-index

            cur_label = target

            for i in range(len(start_idx)):
                tmp = signals[:,int(start_idx[i]):int(end_idx[i])]
                tmp = tmp[:,:min(np.size(tmp,1), int(params['maxtriallen']*params['srate']))]
                tmp = np.pad(tmp,((0,0),(0,int(params['maxtriallen']*params['srate'])-np.size(tmp,1))), 'constant', constant_values=np.nan)
        
                cur_data.append(tmp)
            cur_data = np.array(cur_data)

            # CAR
            cur_data = cur_data-cur_data.mean(axis=1, keepdims=True)
            
            data.append(cur_data)
            label.append(cur_label) #nTrials

    #### Preprocessing ####

    data = np.concatenate(data,axis=0)
    label = np.concatenate(label,axis=0)
    label = label.flatten()
    print(data.shape)
    print(label.shape)

    # relabel the data
    data, label = filter_and_relabel(data, label, keep_labels, new_labels)
    return data, label, params

def train_models(data, label, save_name, params):

    if 'modelpath' in params.keys(): # finetune
        print(f'Fine-tuning model: {save_name}...')
    else:
        print(f'Training model: {save_name}...')
    K.set_image_data_format('channels_last')
    
    nTrial = len(data)
    nChan = np.size(data,axis=1)
    shuffled_idx = np.random.permutation(nTrial)

    # split into training/validation sets
    train_percent = 0.8
    train_idx = range(int(train_percent*nTrial))
    train_idx = shuffled_idx[train_idx]
    val_idx = np.setdiff1d(shuffled_idx,train_idx)
    X_train = data[train_idx,:,:]
    X_validate = data[val_idx,:,:]
    Y_train = label[train_idx]
    Y_validate = label[val_idx]

    ############################# preprocessing ##################################
    # segment data
    times = np.arange(0,params['maxtriallen'],1/params['srate'])
    DesiredLen = int(params['windowlen']*params['downsrate'])

    segment_size = int(params['windowlen']*params['srate'])  # size of each segment - 1 s
    step_size = 128    # step size
    X_train, Y_train, I_train = segment_data(X_train, Y_train, segment_size, step_size)
    X_validate, Y_validate, I_validate = segment_data(X_validate, Y_validate, segment_size, step_size)

    # downsample
    X_train = resample(X_train, DesiredLen, t=None, axis=2, window=None, domain='time')
    X_validate = resample(X_validate, DesiredLen, t=None, axis=2, window=None, domain='time')

    # bandpass filtering
    padding_length = 100  # Number of zeros to pad
    padded_train = np.pad(X_train, ((0,0),(0,0),(padding_length,padding_length)), 'constant', constant_values=0)
    padded_validate = np.pad(X_validate, ((0,0),(0,0),(padding_length,padding_length)), 'constant', constant_values=0)

    b, a = scipy.signal.butter(4, params['bandpass_filt'], btype='bandpass', fs=params['downsrate'])
    X_train = scipy.signal.lfilter(b, a, padded_train, axis=-1)
    X_validate = scipy.signal.lfilter(b, a, padded_validate, axis=-1)

    X_train = X_train[:,:,padding_length:-padding_length]
    X_validate = X_validate[:,:,padding_length:-padding_length]

    # zscore
    X_train = scipy.stats.zscore(X_train, axis=2, nan_policy='omit')
    X_validate = scipy.stats.zscore(X_validate, axis=2, nan_policy='omit')
        
    ############################# EEGNet portion ##################################
    kernels, chans, samples = 1, nChan, DesiredLen
    batch_size, epochs = 16, 300
    # convert labels to one-hot encodings.
    Y_train      = np_utils.to_categorical(Y_train-1)
    Y_validate   = np_utils.to_categorical(Y_validate-1)

    X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    if 'modelpath' in params.keys(): # finetune: larger dropout ratio
        params['dropout_ratio'] = 0.65
    else:
        params['dropout_ratio'] = 0.5
    model = EEGNet(nb_classes = params['nclass'], Chans = chans, Samples = samples, 
                dropoutRate = params['dropout_ratio'], kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                dropoutType = 'Dropout')  
    
    model.summary()

    # Callbacks
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=80)
    callback_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30)

    if 'modelpath' in params.keys(): # finetune: smaller starting lr
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
    else:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                metrics = ['accuracy'])
   
    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath=save_name, verbose=1,monitor='val_accuracy',
                                    mode='max',save_best_only=True)

    class_weights = {0:1, 1:1, 2:1, 3:1}

    if 'modelpath' in params.keys(): # finetune
        params['epochs'] = 100
        params['layers_fine_tune'] = 12
        model.load_weights(params['modelpath'])
        model.trainable = True
        num_layers = len(model.layers)
        num_layers_fine_tune = params['layers_fine_tune']

        for model_layer in model.layers[:num_layers - num_layers_fine_tune]:
            print(f"FREEZING LAYER: {model_layer}")
            model_layer.trainable = False
        
    else:
        params['epochs'] = 300

    model.fit(X_train, Y_train, batch_size = batch_size, epochs = params['epochs'], 
                verbose = 2, validation_data=(X_validate, Y_validate),
                callbacks=[checkpointer, callback_es, callback_lr], class_weight = class_weights)

    print("Training Finished!")
    print(f"Model saved to {save_name}")
    return save_name
