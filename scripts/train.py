# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:57:20 2023

@author: Ruben B
"""

import configparser
import tensorflow as tf
from audio_nn import dataset as dataset_utils, model as model_utils
from sklearn.model_selection import train_test_split
from audio_nn import callbacks
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[1], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')

files, labels, classes, class_weights = dataset_utils.load_dataset(config)

class_weights[0] = 1.2
class_weights[1] = 6

augmentation_datasets = []
for augmentation_dataset in config.get("Dataset", "augmentation_dir").split(','):
    d = dataset_utils.load_dataset(config, input_dir=augmentation_dataset, files_only=True)
    augmentation_datasets.append(d)

files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)

batch_size = config.getint("Training", "batch_size")

augmentation_gen = dataset_utils.augmentation_generator([
    dataset_utils.crop_augmentation(new_length=config.getint("Audio data", "keep_samples"), p=1.0),
    dataset_utils.gain_augmentation(max_db=3, p=1.0),
    dataset_utils.noise_augmentation(max_noise_ratio=0.06, p=1.0),
    dataset_utils.mix_augmentation(augmentation_datasets[0], min_ratio=0.001, max_ratio=0.25, p=0.4),
    dataset_utils.mix_augmentation(augmentation_datasets[1], min_ratio=0.001, max_ratio=0.25, p=0.4),
    dataset_utils.mix_augmentation(augmentation_datasets[2], min_ratio=0.001, max_ratio=0.25, p=0.4),
    dataset_utils.noise_augmentation(min_noise_ratio=0.01, max_noise_ratio=0.03, p=1.0),
    dataset_utils.gain_augmentation(max_db=2, p=1.0)
])

data_gen_train = dataset_utils.data_generator(config, files_train, labels_train, batch_size, augmentation_gen)
data_gen_val = dataset_utils.data_generator(config, files_val, labels_val, batch_size, augmentation_gen)

model = model_utils.create_model_v999(config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

input("Press enter to continue...")

save_each_epoch_callback = callbacks.SaveModelEachEpoch(config)
save_best_model_callback = callbacks.SaveBestModel(config)

model.fit_generator(generator = data_gen_train,
                        steps_per_epoch = len(files_train)//batch_size,
                        validation_data = data_gen_val,
                        validation_steps = len(files_val)//batch_size,
                        epochs = config.getint("Training", "epochs"),
                        verbose = 1,
                        callbacks=[save_best_model_callback, save_each_epoch_callback],
                        class_weight=class_weights)