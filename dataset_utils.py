# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 20:47:30 2023

@author: Ruben
"""

import os
import numpy as np
from audio_utils import load_wav_16k_mono, load_wav_16k_mono_v2, zero_pad_wav, compute_mfcc, compute_stft, compute_chroma_stft
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

def load_dataset(config, ratio = 2):
    input_dir = config.get("Dataset", "dir")

    files = []
    labels = []

    """
    if num_classes == 2:
        pos_dir = os.path.join(input_dir, 'pos')
        neg_dir = os.path.join(input_dir, 'neg')
        
        pos_files = os.listdir(pos_dir)
        neg_files = os.listdir(neg_dir)

        pos_labels = np.ones(len(pos_files))
        neg_labels = np.zeros(len(neg_files))

        pos_files = np.array(pos_files)
        neg_files = np.array(neg_files)
        
        pos_files = [os.path.join(pos_dir, pos_file) for pos_file in pos_files]
        neg_files = [os.path.join(neg_dir, neg_file) for neg_file in neg_files]
        
        pos_files, pos_labels = shuffle(pos_files, pos_labels, random_state=5342)
        neg_files, neg_labels = shuffle(neg_files, neg_labels, random_state=6754)

        neg_files = neg_files[:len(pos_files)*ratio]
        neg_labels = neg_labels[:len(pos_labels)*ratio]

        files = np.concatenate((pos_files, neg_files))
        labels = np.concatenate((pos_labels, neg_labels))
        
        files, labels = shuffle(files, labels, random_state = 45678876)
        classes = ["neg", "pos"]
    else:
    """

    classes_dirs = [os.path.join(input_dir, folder) for folder in os.listdir(input_dir)]
    for class_dir in classes_dirs:
        for file in os.listdir(class_dir):
            files.append(os.path.join(class_dir, file))
            labels.append(os.path.basename(class_dir))

    classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=np.array(labels))
    class_weights_dict = dict(zip(classes, class_weights))

    encoder = OneHotEncoder(sparse=False)
    labels = np.array(labels).reshape(-1, 1)
    labels_one_hot = encoder.fit_transform(labels)

    class_indices = range(len(classes))
    class_weights_indexed = {class_index: class_weights_dict[cls] for cls, class_index in zip(classes, class_indices)}

    return files, labels_one_hot, classes, class_weights_indexed

def switch_compute_fn(feature_type):
    if feature_type == "mfcc":
        return compute_mfcc
    elif feature_type == "stft":
        return compute_stft
    elif feature_type == "chroma_stft":
        return compute_chroma_stft
    else:
        return compute_stft

def data_generator(config, files, labels, batch_size, augmentation_generator = None):
    while True:
        for i in range(0, len(files), batch_size):
            files_batch = files[i:i+batch_size]
            labels_batch = labels[i:i+batch_size]

            feature_types = config.get("Audio data", "feature_types").split(',')
            time_axis = config.getint('Audio data', 'time_axis')
            k_axis = config.getint('Audio data', 'k_axis')
            n_fft = config.getint('Audio data', 'n_fft')
            keep_samples = config.getint("Audio data", "keep_samples")

            wavs = np.array([zero_pad_wav(load_wav_16k_mono_v2(filename), keep_samples) for filename in files_batch])
            
            compute_fns = [switch_compute_fn(feature_type) for feature_type in feature_types]

            features = [[compute_fn(wav, time_axis, k_axis, n_fft) for wav in wavs] for compute_fn in compute_fns]

            features = [(feature-np.min(feature))/(np.max(feature)-np.min(feature)) for feature in features]

            features = np.stack(features, axis = 3)
            
            # features = features.astype(np.uint8)
            yield features, labels_batch

def test_generator(files, batch_size):
    while True:
        for i in range(0, len(files), batch_size):
            files_batch = files[i:i+batch_size]

            wavs = np.array([zero_pad_wav(load_wav_16k_mono_v2(filename)) for filename in files_batch])
            
            # stfts = [compute_stft(wav)[:128] for wav in wavs]
            stfts = [compute_stft(wav)[1:257] for wav in wavs]
            # mfccs = [compute_mfcc(wav)[:128] for wav in wavs]
            mfccs = [compute_mfcc(wav) for wav in wavs]
            
            stfts = np.array(stfts)
            mfccs = np.array(mfccs)
            
            stfts = (stfts-np.min(stfts))/(np.max(stfts)-np.min(stfts)) #*255
            mfccs = (mfccs-np.min(mfccs))/(np.max(mfccs)-np.min(mfccs)) #*255
            
            # stfts = stfts.astype(np.uint8)
            # mfccs = mfccs.astype(np.uint8)
            
            # features = np.stack((stfts, mfccs), axis=3)
            yield stfts

def representative_dataset_gen(files, batch_size):
    representative_data = []
    for i in range(0, len(files), batch_size):
        files_batch = files[i:i + batch_size]

        wavs = np.array([zero_pad_wav(load_wav_16k_mono_v2(filename)) for filename in files_batch])

        stfts = [compute_stft(wav)[1:257] for wav in wavs]
        mfccs = [compute_mfcc(wav) for wav in wavs]

        stfts = np.array(stfts)
        mfccs = np.array(mfccs)

        stfts = (stfts - np.min(stfts)) / (np.max(stfts) - np.min(stfts)) * 255
        mfccs = (mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs)) * 255

        stfts = stfts.astype(np.uint8)
        mfccs = mfccs.astype(np.uint8)

        representative_data.append(stfts)

    return representative_data