# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:57:20 2023

@author: Ruben B
"""

import configparser
import tensorflow as tf
from audio_nn import model as model_utils, dataset as dataset_utils
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
from audio_nn import callbacks as callbacks
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[0], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')

model, config = model_utils.load_model(config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='BinaryCrossentropy', metrics=['accuracy'])
print(model.summary())

files, labels, classes, class_weights = dataset_utils.load_dataset(config)

files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)

batch_size = config.getint("Training", "batch_size")

data_gen_train = dataset_utils.data_generator(config, files_train, labels_train, batch_size, None)
data_gen_val = dataset_utils.data_generator(config, files_val, labels_val, batch_size, None)

save_each_epoch_callback = callbacks.SaveModelEachEpoch(config)
save_best_model_callback = callbacks.SaveBestModel(config)

model.fit_generator(generator = data_gen_train,
                        steps_per_epoch = len(files_train)//batch_size,
                        validation_data = data_gen_val,
                        validation_steps = len(files_val)//batch_size,
                        epochs = config.getint("Training", "epochs"),
                        verbose = 1,
                        callbacks=[save_each_epoch_callback, save_best_model_callback],
                        class_weight=class_weights)

model_utils.save_model(config, model, "_fine_tuned")