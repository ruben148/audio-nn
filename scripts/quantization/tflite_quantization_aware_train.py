import configparser
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from audio_nn import model as model_utils, dataset as dataset_utils, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os

import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[2], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')
batch_size = config.getint("Training", "batch_size")

model, config = model_utils.load_model(config)

files, labels, classes, class_weights = dataset_utils.load_dataset(config)
files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.3, random_state=42, stratify=labels)

data_gen_train = dataset_utils.data_generator(config, files_train, labels_train, batch_size, None)
data_gen_val = dataset_utils.data_generator(config, files_val, labels_val, batch_size, None)

quantize_model = tfmot.quantization.keras.quantize_model

q_aware_model = quantize_model(model)

q_aware_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.getfloat("Training", "lr")), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
print(q_aware_model.summary())

save_best_callback = callbacks.SaveBestModel(config)

q_aware_model.fit_generator(generator = data_gen_train,
                        steps_per_epoch = len(files_train)//batch_size,
                        validation_data = data_gen_val,
                        validation_steps = len(files_val)//batch_size,
                        epochs = config.getint("Training", "epochs"),
                        verbose = 1,
                        callbacks=[save_best_callback],
                        class_weight=class_weights)

model_utils.save_model(config, q_aware_model)