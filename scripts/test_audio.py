# -*- coding: utf-8 -*-

import configparser

import tensorflow_model_optimization as tfmot
import tensorflow as tf
from audio_nn import model as model_utils, dataset as dataset_utils, audio as audio_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import pandas as pd
import soundfile as sf

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[2], 'GPU')

# audio_path = "/home/buu3clj/radar_ws/datasets_2_0/WE FOUND THE BEST SKATEPARK EVER.wav"
audio_path = "/home/buu3clj/radar_ws/datasets_2_0/all.wav"
# audio_path = "/home/buu3clj/radar_ws/datasets_2_0/test_recordings/Voice 002.wav"
# output_path = "/home/buu3clj/radar_ws/datasets_2_0/split_recordings"
# audio_path = "/home/buu3clj/radar_ws/datasets_2_0/chainsaw_original/RP15_1h10_till_1h25.wav"
# audio_path = "/home/buu3clj/radar_ws/datasets_2_0/chainsaw_original/RP3_0h30_to1h20.wav"

wav = audio_utils.load_wav_8k_mono(audio_path)

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')

with tfmot.quantization.keras.quantize_scope():
    model, config = model_utils.load_model(config)

print("Model summary: ", model.summary())



"""
files, labels, classes, class_weights = dataset_utils.load_dataset(config)
files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.3, random_state=42, stratify=labels)

augmentation_datasets = []
for augmentation_dataset in config.get("Dataset", "augmentation_dir").split(','):
    d = dataset_utils.load_dataset(config, input_dir=augmentation_dataset, files_only=True)
    augmentation_datasets.append(d)

augmentation_gen = dataset_utils.augmentation_generator([
    dataset_utils.crop_augmentation(new_length=config.getint("Audio data", "keep_samples"), p=1.0),
    dataset_utils.gain_augmentation(max_db=3, p=1.0),
    dataset_utils.noise_augmentation(max_noise_ratio=0.06, p=1.0),
    dataset_utils.mix_augmentation(augmentation_datasets[0], min_ratio=0.001, max_ratio=0.25, p=1.0),
    dataset_utils.mix_augmentation(augmentation_datasets[1], min_ratio=0.001, max_ratio=0.25, p=1.0),
    dataset_utils.noise_augmentation(min_noise_ratio=0.01, max_noise_ratio=0.03, p=1.0),
    dataset_utils.gain_augmentation(max_db=2, p=1.0)
])

data_gen_train = dataset_utils.data_generator_testing(config, files_train, labels_train, 16, augmentation_gen)

# os.rmdir('/home/buu3clj/radar_ws/datasets_2_0/testing_batch')
# os.mkdir('/home/buu3clj/radar_ws/datasets_2_0/testing_batch')

for j in range(10):
    wavs, labels = next(data_gen_train)

    for i, wav in enumerate(wavs):
        if labels[i][1]==1:
            filename = f'/home/buu3clj/radar_ws/datasets_2_0/testing_batch/{j}_{i}_pos.wav'
            print("Writing positive")
        else:
            filename = f'/home/buu3clj/radar_ws/datasets_2_0/testing_batch/{j}_{i}_neg.wav'
            print("Writing negative")
        sf.write(filename, wav, 8000)



input("Press enter.")
"""


n_fft = config.getint('Audio data', 'n_fft')
keep_samples = config.getint("Audio data", "keep_samples")
time_axis = config.getint('Audio data', f'stft_time_axis')
k_axis = config.getint('Audio data', f'stft_k_axis')

j = 0
for i in range(0, wav.size-keep_samples, keep_samples):
    w = wav[i:i+7168]
    # sf.write(f'{output_path}/002_{j}.wav', w, 8000)
    # j = j + 1
    # """
    w = np.array(audio_utils.zero_pad_wav(w, samples = keep_samples))
    feature = audio_utils.compute_stft(w, time_axis, k_axis, n_fft)
    # feature = (feature-np.min(feature))/(np.max(feature)-np.min(feature))
    feature = np.array(feature)
    feature = feature.astype(np.float32)
    feature = np.expand_dims(feature, axis=0)
    feature = np.expand_dims(feature, axis=-1)
    y_pred = model(feature)
    if(y_pred[0][1]>0.6):
        start = i/8000
        end = (i+keep_samples)/8000
        print(f'{start}-{end} contains chainsaw: {y_pred[0][1]:.3f}\n')
        start = i-2000
        end = i+14000
        sf.write(f'/home/buu3clj/radar_ws/datasets_2_0/output_sequences/seq_{j}.wav', wav[start:end], 8000)
        j += 1
    # """
