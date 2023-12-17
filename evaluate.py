# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:57:20 2023

@author: Ruben B
"""

import configparser
import tensorflow as tf
from model_utils import train_model, save_model, load_model
from dataset_utils import load_dataset, load_dataset_full, data_generator, test_generator
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
config.read('/home/buu3clj/radar_ws/audio-nn/config.ini')


files, labels = load_dataset_full(config)

batch_size = config.getint("Training", "batch_size")

test_gen = data_generator(files, labels, batch_size, None)

model = load_model(config, "chainsaw_best.h5")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),loss='BinaryCrossentropy',metrics=['accuracy'])
print(model.summary())

"""
(features, labels) = next(data_gen)

y_pred = model.predict(features)
y_pred = np.array(y_pred).flatten().round(2)

data = {'Filename': files, 'True Label': labels, 'Prediction': y_pred}
df = pd.DataFrame(data)

print(df.to_string())
"""

evaluation = model.evaluate_generator(test_gen, steps = len(files)//batch_size, verbose=1)

print("Evaluation Results: ", evaluation)

y_pred = model.predict_generator(test_gen, verbose = 1, steps = len(files)//batch_size+1)

y_pred_labels = (y_pred > 0.5).astype(int)

y_true = labels

cm = confusion_matrix(y_true, y_pred_labels, labels=[1, 0])
print("Confusion Matrix:")
print(cm)

report = classification_report(y_true, y_pred_labels)
print("Classification Report:")
print(report)