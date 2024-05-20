# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:57:20 2023

@author: Ruben B
"""
import configparser

import tensorflow_model_optimization as tfmot
import tensorflow as tf
from tensorflow import keras
from audio_nn import model as model_utils, dataset as dataset_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import pandas as pd

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[1], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')

with tfmot.quantization.keras.quantize_scope():
    model, config = model_utils.load_model(config)

files, labels, classes, class_weights = dataset_utils.load_dataset(config, keep = 1, input_dir=config.get("Validation", "dir"))

batch_size = config.getint("Training", "batch_size")

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

test_gen = dataset_utils.data_generator(config, files, labels, batch_size, augmentation_gen)

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
#                   loss='categorical_crossentropy', 
#                   metrics=['accuracy'])

print("Model summary: ", model.summary())

evaluation = model.evaluate_generator(test_gen, steps = len(files)//batch_size, verbose=1)

print("Evaluation Results: ", evaluation)

y_pred = model.predict_generator(test_gen, verbose = 1, steps = len(files)//batch_size+1)

y_pred = np.argmax(y_pred, axis=1)

y_true = np.argmax(labels, axis=1)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("\nConfusion Matrix:")
print(pd.DataFrame(cm, index=['True 0', 'True 1'], columns=['Predicted 0', 'Predicted 1']))

report = classification_report(y_true, y_pred)
print("Classification Report:")
print(report)