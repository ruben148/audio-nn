import configparser
import tensorflow as tf
from audio_nn import model as model_utils, dataset as dataset_utils, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')

model, config = model_utils.load_model(config)

files, labels, classes, class_weights = dataset_utils.load_dataset(config, keep=0.2)

batch_size = config.getint("Training", "batch_size")

rdg = lambda: dataset_utils.representative_data_generator(config, files, batch_size)

quant_model = model_utils.quantize_model(config, model, rdg)

model_utils.save_model(quant_model)