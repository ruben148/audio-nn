import configparser
import tensorflow as tf
from audio_nn import model as model_utils, dataset as dataset_utils, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np
from tensorflow_model_optimization.python.core.quantization.keras.quantize_layer import QuantizeLayer
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_scope

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')

with quantize_scope():
    model, config = model_utils.load_model(config)

files, labels, classes, class_weights = dataset_utils.load_dataset(config)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
print("\nQuantization started...\n")
quant_model = converter.convert()
print("\nQuantization ended.\n")

print("Model size (bytes): ", len(quant_model))

model_utils.save_tfmodel(config, quant_model)