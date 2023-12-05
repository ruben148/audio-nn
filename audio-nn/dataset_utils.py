# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 20:47:30 2023

@author: Ruben
"""

import os
import numpy as np
from audio_utils import load_wav_16k_mono, load_wav_16k_mono_v2, zero_pad_wav, compute_mfcc, compute_stft
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder


def load_dataset(config, ratio = 2):
    input_dir = config.get("Dataset", "dir")
    num_classes = config.getint('Model', 'num_classes')

    files = []
    labels = []

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
        classes_dirs = [os.path.join(input_dir, folder) for folder in os.listdir(input_dir)]
        for class_dir in classes_dirs:
            for file in os.listdir(class_dir):
                files.append(os.path.join(class_dir, file))
                labels.append(os.path.basename(class_dir))
        encoder = OneHotEncoder(sparse=False)
        classes = np.unique(labels)
        labels = np.array(labels).reshape(-1, 1)
        labels = encoder.fit_transform(labels)

    return files, labels, classes

def load_dataset_full(config):
    input_dir = config.get("Dataset", "dir")
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

    files = np.concatenate((pos_files, neg_files))
    labels = np.concatenate((pos_labels, neg_labels))
    
    files, labels = shuffle(files, labels, random_state = 45678876)
    return files, labels

def data_generator(files, labels, batch_size, augmentation_generator):
    while True:
        for i in range(0, len(files), batch_size):
            files_batch = files[i:i+batch_size]
            labels_batch = labels[i:i+batch_size]

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
            yield stfts, labels_batch

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