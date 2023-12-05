# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:57:20 2023

@author: Ruben B
"""

import configparser
import tensorflow as tf
from model_utils import create_model_v5, train_model, save_model, quantize_model
from dataset_utils import load_dataset, data_generator, representative_dataset_gen
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
from callbacks import SaveModelEachEpoch, SaveBestModel
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[1], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio-nn/config.ini')


files, labels, c = load_dataset(config)

files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)

# class_labels = np.unique(labels)
# class_weights = compute_class_weight('balanced', classes = class_labels, y = labels)
# class_weights_dict = dict(zip(class_labels, class_weights))

batch_size = config.getint("Training", "batch_size")

data_gen_train = data_generator(files_train, labels_train, batch_size, None)
data_gen_val = data_generator(files_val, labels_val, batch_size, None)

model = create_model_v5(config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

images, labels = next(data_gen_train)

print(images.shape)
print(labels.shape)

input("Press enter to continue...")

save_each_epoch_callback = SaveModelEachEpoch(config)
save_best_model_callback = SaveBestModel(config)

train_model(config, model,
             data_gen_train,
             data_gen_val,
             len(files_train),
             len(files_val),
             callbacks=[save_each_epoch_callback, save_best_model_callback]
            #  class_weights=class_weights_dict
             )

save_model(config, model, "final_chainsaw.h5")