import configparser
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from audio_nn import model as model_utils, dataset as dataset_utils
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')

with tfmot.quantization.keras.quantize_scope():
    model, config = model_utils.load_model(config)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
quant_model = converter.convert()

print(len(quant_model))

model_utils.save_tfmodel(config, quant_model)