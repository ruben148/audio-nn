# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:48:56 2023

@author: Ruben B
"""

import tensorflow as tf
import os
from tensorflow.keras import models, layers, activations
from tensorflow.keras.regularizers import l1
import tensorflow_hub as hub
import ssl

def create_model_v5(config):
    input_shape = tuple(map(int, config.get('Model', 'input_shape').split(',')))
    num_classes = config.getint('Model', 'num_classes')

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), input_shape=input_shape))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))

    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))

    model.add(layers.MaxPooling2D((1, 2)))
    model.add(layers.Dropout(0.15)) # 0.1 -> 0.15

    model.add(layers.Conv2D(32, (3,3))) # strides=2 -> strides=1
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))

    model.add(layers.Conv2D(32, (3,3)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.15)) # 0.1 -> 0.15

    model.add(layers.Conv2D(32, (3,3)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))

    model.add(layers.Conv2D(32, (3,3)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))

    model.add(layers.MaxPooling2D((2, 2))) # (3, 3) -> (2, 2)
    model.add(layers.Dropout(0.15)) # 0.1 -> 0.15

    model.add(layers.Conv2D(32, (3,3)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))

    model.add(layers.Conv2D(64, (3,3)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))

    model.add(layers.Conv2D(128, (3,3))) # 96 -> 128
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation(activations.relu))

    model.add(layers.Dropout(0.15)) # 0.1 -> 0.15
    model.add(layers.MaxPooling2D((3, 3)))

    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu')) # 10 -> 16
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        model.add(layers.Dense(num_classes  , activation='softmax'))
    return model

def create_yamnet(config):
    input_shape = tuple(map(int, config.get('Model', 'input_shape').split(',')))

    ssl._create_default_https_context = ssl._create_unverified_context
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    yamnet_model.build(input_shape)

    model = tf.keras.Sequential([
        yamnet_model,
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model

def train_model(config, model, data_gen_train, data_gen_val, n_data_train, n_data_val, callbacks=[], class_weights=None):
    batch_size = config.getint("Training", "batch_size")
    model.fit_generator(generator = data_gen_train,
                        steps_per_epoch = n_data_train//batch_size,
                        validation_data = data_gen_val,
                        validation_steps = n_data_val//batch_size,
                        epochs = config.getint("Training", "epochs"),
                        verbose = 1,
                        callbacks=callbacks,
                        class_weight=class_weights)

def save_model(config, model, filename = None):
    if filename == None:
        filename = config.get("Model", "filename")
        filename = filename + ".h5"
    output_dir = config.get("Model", "dir")
    full_path = os.path.join(output_dir, filename)
    model.save(full_path)

def save_tfmodel(config, model, filename = None):
    if filename == None:
        filename = config.get("Model", "filename")
        filename = filename + ".h5"
    output_dir = config.get("Model", "dir")
    full_path = os.path.join(output_dir, filename)
    with open(full_path, 'wb') as f:
        f.write(model)

def load_model(config, filename = None, custom_objects = None):
        if filename == None:
            filename = config.get("Model", "filename")
            filename = filename + ".h5"
        input_dir = config.get("Model", "dir")
        full_path = os.path.join(input_dir, filename)
        loaded_model = tf.keras.models.load_model(full_path, custom_objects=custom_objects)
        return loaded_model

def quantize_model(config, model, representative_dataset):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    print("\nQuantization started...\n")
    quant_model = converter.convert()
    print("\nQuantization ended!\n")
    return quant_model