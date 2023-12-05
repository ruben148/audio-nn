import configparser
import tensorflow as tf
from model_utils import create_model_v1, create_model_v4, create_model_v3, create_mobilenet, train_model, load_model, save_model, quantize_model
from dataset_utils import load_dataset, data_generator, representative_dataset_gen
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
from callbacks import SaveModelEachEpoch, SaveBestModel
import numpy as np

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio-nn/config.ini')

model = load_model(config, "chainsaw_best.h5")

files, labels = load_dataset(config)

rdg = representative_dataset_gen(files, 32)

quant_model = quantize_model(config, model, rdg)

save_model(quant_model, "quantized_chainsaw.h5")