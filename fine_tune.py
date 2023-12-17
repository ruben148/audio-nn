# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:57:20 2023

@author: Ruben B
"""

import configparser
import tensorflow as tf
from model_utils import create_model_v5, train_model, save_model, quantize_model, load_model
from dataset_utils import load_dataset, data_generator, representative_dataset_gen
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
from callbacks import SaveModelEachEpoch, SaveBestModel
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[0], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio-nn/config.ini')


files, labels = load_dataset(config)

files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.3, random_state=42, stratify=labels)

class_labels = np.unique(labels)
class_weights = compute_class_weight('balanced', classes = class_labels, y = labels)
class_weights_dict = dict(zip(class_labels, class_weights))

batch_size = config.getint("Training", "batch_size")

data_gen_train = data_generator(files_train, labels_train, batch_size, None)
data_gen_val = data_generator(files_val, labels_val, batch_size, None)

model = load_model(config, "best_chainsaw")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='BinaryCrossentropy', metrics=['accuracy'])
print(model.summary())

input("Press enter to continue...")

batch_size = config.getint("Training", "batch_size")
model.fit_generator(generator = data_gen_train,
                        steps_per_epoch = len(files_train)//batch_size,
                        validation_data = data_gen_val,
                        validation_steps = len(files_val)//batch_size,
                        epochs = config.getint("FineTuning", "epochs"),
                        verbose = 1,
                        callbacks=[],
                        class_weight=class_weights)

save_model(config, model, "fine_tuned_chainsaw.h5")