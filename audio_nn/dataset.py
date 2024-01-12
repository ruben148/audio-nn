# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 20:47:30 2023

@author: Ruben
"""

import os
import numpy as np
from audio_nn.audio import load_wav_16k_mono, zero_pad_wav, compute_mfcc, compute_stft, compute_chroma_stft, add_noise, random_gain, mix_advanced, random_crop
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import random

np.random.seed(876323456)

def load_dataset(config, ratio = None, keep = 1.0, input_dir = None, files_only = False):
    if input_dir is None:
        input_dir = config.get("Dataset", "dir")
    # input_dir = os.path.join(input_dir, config.get("Audio data", "keep_samples"))

    if files_only:
        files = []
        for file in os.listdir(input_dir):
            files.append(os.path.join(input_dir, file))
        np.random.shuffle(files)
        return files
    
    files = []
    labels = []

    classes_dirs = [os.path.join(input_dir, folder) for folder in os.listdir(input_dir)]
    for class_dir in classes_dirs:
        per_class_files = []
        per_class_labels = []
        for file in os.listdir(class_dir):
            per_class_files.append(os.path.join(class_dir, file))
            per_class_labels.append(os.path.basename(class_dir))
            
        combined = list(zip(per_class_files, per_class_labels))
        np.random.shuffle(combined)
        per_class_files, per_class_labels = zip(*combined)

        per_class_files = per_class_files[:int(len(per_class_files)*keep)]
        per_class_labels = per_class_labels[:int(len(per_class_labels)*keep)]
        files.extend(per_class_files)
        labels.extend(per_class_labels)

    classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=np.array(labels))
    class_weights_dict = dict(zip(classes, class_weights))

    encoder = OneHotEncoder(sparse=False)
    labels = np.array(labels).reshape(-1, 1)
    labels_one_hot = encoder.fit_transform(labels)

    class_indices = range(len(classes))
    class_weights_indexed = {class_index: class_weights_dict[cls] for cls, class_index in zip(classes, class_indices)}

    combined = list(zip(files, labels_one_hot))

    np.random.shuffle(combined)
    np.random.shuffle(combined)
    np.random.shuffle(combined)

    files, labels_one_hot = zip(*combined)

    files = list(files)
    labels_one_hot = np.array(labels_one_hot)

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

"""
def data_generator(config, files, labels, batch_size, augmentation_generator = None, time_axis = None, k_axis = None):
    while True:
        for i in range(0, len(files), batch_size):
            files_batch = files[i:i+batch_size]
            labels_batch = labels[i:i+batch_size]

            feature_types = config.get("Audio data", "feature_types").split(',')
            if time_axis is None:
                time_axis = config.getint('Audio data', f'{feature_types[0]}_time_axis')
            if k_axis is None:
                k_axis = config.getint('Audio data', f'{feature_types[0]}_k_axis')
            n_fft = config.getint('Audio data', 'n_fft')
            keep_samples = config.getint("Audio data", "keep_samples")

            wavs = np.array([zero_pad_wav(load_wav_16k_mono(filename), keep_samples) for filename in files_batch])
            
            compute_fns = [switch_compute_fn(feature_type) for feature_type in feature_types]

            features = [[compute_fn(wav, time_axis, k_axis, n_fft) for wav in wavs] for compute_fn in compute_fns]

            features = [(feature-np.min(feature))/(np.max(feature)-np.min(feature)) for feature in features]

            features = np.stack(features, axis = 3)
            
            # features = features.astype(np.uint8)
            yield features, labels_batch
"""

def augmentation_generator(augmentation_functions):
    def augment(sound):
        for augmentation_func in augmentation_functions:
            sound = augmentation_func(sound)
        return sound
    return augment

def noise_augmentation(noise_types=['white', 'pink', 'brown', 'gaussian'], min_noise_ratio=0.04, max_noise_ratio=0.10, p=0.5):
    def add_noise_aux(sound):
        if random.random() > p:
            return sound
        noise_type = random.choice(noise_types)
        sound = add_noise(sound, min_noise_ratio, max_noise_ratio, noise_type)
        return sound
    return add_noise_aux

def mix_augmentation(negative_dataset, min_ratio=0.05, max_ratio=0.5, p=0.5):
    def mix_sounds(sound1):
        if random.random() > p:
            return sound1
        index = random.randint(0, len(negative_dataset)-1)
        sound2 = load_wav_16k_mono(negative_dataset[index])
        sound2 = random_crop(sound2, len(sound1))
        ratio = random.uniform(1-max_ratio, 1-min_ratio)
        mixed = mix_advanced(sound1, sound2, ratio)
        return mixed
    return mix_sounds

def gain_augmentation(max_db=10, p=0.5):
    def random_gain_aux(sound):
        if random.random() > p:
            return sound
        sound = random_gain(sound, max_db)
        return sound
    return random_gain_aux

def crop_augmentation(new_length=12288, p=0.5):
    def random_crop_aux(sound):
        if random.random() > p:
            return sound
        sound = random_crop(sound, new_length)
        return sound
    return random_crop_aux

def data_generator(config, files, labels, batch_size, augmentation_generator = None, time_axis = None, k_axis = None, quantize = False):
    while True:
        for i in range(0, len(files), batch_size):
            files_batch = files[i:i+batch_size]
            labels_batch = labels[i:i+batch_size]

            n_fft = config.getint('Audio data', 'n_fft')
            feature_types = config.get("Audio data", "feature_types").split(',')
            keep_samples = config.getint("Audio data", "keep_samples")
            time_steps = config.getint("Model", "time_steps")

            wavs = [load_wav_16k_mono(filename) for filename in files_batch]
            wavs = [augmentation_generator(wav) for wav in wavs]
            wavs = np.array([zero_pad_wav(wav, samples=keep_samples) for wav in wavs])

            features_dict = {}
            for feature_type in feature_types:
                if time_axis is None:
                    time_axis = config.getint('Audio data', f'{feature_type}_time_axis')
                if k_axis is None:
                    k_axis = config.getint('Audio data', f'{feature_type}_k_axis')
                compute_fn = switch_compute_fn(feature_type)
                feature = [compute_fn(wav, time_axis, k_axis, n_fft) for wav in wavs]

                feature = (feature-np.min(feature))/(np.max(feature)-np.min(feature))
                # feature = (feature-(-1))/(1-(-1))
                # TODO ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                feature = np.array(feature)
                if quantize:
                    feature = feature.astype(np.uint8)
                features_dict[feature_type] = feature

                if time_steps > 1:
                    assert feature.shape[1] % time_steps == 0
                    new_shape = (feature.shape[0], time_steps, int(feature.shape[1] / time_steps), feature.shape[2])
                    feature = feature.reshape(new_shape)
            
            yield [features_dict[feature_type] for feature_type in feature_types], labels_batch

def representative_data_generator(config, files, batch_size, time_axis = None, k_axis = None, quantize = False):
    representative_data = []
    for i in range(0, len(files), batch_size):
        files_batch = files[i:i+batch_size]

        n_fft = config.getint('Audio data', 'n_fft')
        feature_types = config.get("Audio data", "feature_types").split(',')
        keep_samples = config.getint("Audio data", "keep_samples")
        time_steps = config.getint("Model", "time_steps")
        wavs = np.array([zero_pad_wav(load_wav_16k_mono(filename), keep_samples) for filename in files_batch])

        features_dict = {}
        for feature_type in feature_types:
            if time_axis is None:
                time_axis = config.getint('Audio data', f'{feature_type}_time_axis')
            if k_axis is None:
                k_axis = config.getint('Audio data', f'{feature_type}_k_axis')
            compute_fn = switch_compute_fn(feature_type)
            feature = [compute_fn(wav, time_axis, k_axis, n_fft) for wav in wavs]
            feature = (feature-np.min(feature))/(np.max(feature)-np.min(feature))
            feature = np.array(feature)
            if quantize:
                feature = feature.astype(np.uint8)
            features_dict[feature_type] = feature

            if time_steps > 1:
                assert feature.shape[1] % time_steps == 0
                new_shape = (feature.shape[0], time_steps, int(feature.shape[1] / time_steps), feature.shape[2])
                feature = feature.reshape(new_shape)
        
        feature = features_dict[feature_types[0]]
        # if feature.ndim == 3:
        #     feature = np.expand_dims(feature, axis=0)
        yield [feature]

        # assert len(feature_types)==1
        # yield [features_dict[feature_types[0]]]
        # representative_data.append(features_dict[feature_types[0]])

    # return representative_data