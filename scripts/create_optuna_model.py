import configparser
import tensorflow as tf
from audio_nn import model as model_utils, dataset as dataset_utils, callbacks
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.keras import models, layers, activations
import tensorflow_model_optimization as tfmot
from tensorflow.keras.regularizers import l1
import tensorflow_hub as hub
import optuna

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[1], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')

sqlite_url = config.get("Optuna", "study_file")
study_name = config.get("Optuna", "study_name")

study = optuna.create_study(direction='minimize', study_name = study_name, storage=sqlite_url, load_if_exists=True)

trial = study.best_trial

keep_samples = trial.suggest_categorical("keep_samples", [10752, 11264, 11776, 12288, 12800])

config.set("Audio data", "keep_samples", str(keep_samples))

feature_types = 'stft'
config.set("Audio data", "feature_types", feature_types)
for feature_type in feature_types.split(','):
    time_axis_name = f'{feature_type}_time_axis'
    k_axis_name = f'{feature_type}_k_axis'
    time_axis = trial.suggest_categorical(time_axis_name, [32, 64, 128, 256, 512])
    k_axis = trial.suggest_int(k_axis_name, 8, (128 if feature_type == "mfcc" else 512))
    config.set("Audio data", time_axis_name, str(time_axis))
    config.set("Audio data", k_axis_name, str(k_axis))

model = model_utils.create_model_optuna(config, trial)

lr=1e-5
config.set("Training", "lr", str(lr))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.getfloat("Training", "lr")), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

print("Model summary: ", model.summary())

input("Press enter.")

model_utils.save_model(config, model)