import configparser
import tensorflow as tf
from model_utils import load_model, save_model, save_tfmodel, quantize_model
from dataset_utils import load_dataset, data_generator, representative_dataset_gen
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
from callbacks import SaveModelEachEpoch, SaveBestModel
import numpy as np
from tensorflow_model_optimization.python.core.quantization.keras.quantize_layer import QuantizeLayer
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_scope

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio-nn/config.ini')

with quantize_scope():
    model = load_model(config, "chainsaw_quantization_aware.h5")

files, labels = load_dataset(config)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
print("\nQuantization started...\n")
quant_model = converter.convert()
print("\nQuantization ended.\n")

print(len(quant_model))

save_tfmodel(config, quant_model, "chainsaw_quantized_best.tflite")