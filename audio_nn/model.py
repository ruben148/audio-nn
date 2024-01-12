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
import configparser
import copy

def create_model_v4(config):
    input_shape = tuple(map(int, config.get('Model', 'input_shape').split(',')))
    num_classes = config.getint('Dataset', 'num_classes')

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
        
    model.add(layers.Dense(num_classes  , activation='softmax'))
    return model

def create_model_v5(config):
    input_shape = tuple(map(int, config.get('Model', 'input_shape').split(',')))
    num_classes = config.getint('Dataset', 'num_classes')

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), input_shape=input_shape))
    model.add(layers.Activation(activations.relu))

    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.Activation(activations.relu))

    model.add(layers.MaxPooling2D((1, 2)))
    model.add(layers.Dropout(0.15)) # 0.1 -> 0.15

    model.add(layers.Conv2D(32, (3,3))) # strides=2 -> strides=1
    model.add(layers.Activation(activations.relu))

    model.add(layers.Conv2D(32, (3,3)))
    model.add(layers.Activation(activations.relu))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.15)) # 0.1 -> 0.15

    model.add(layers.Conv2D(32, (3,3)))
    model.add(layers.Activation(activations.relu))

    model.add(layers.Conv2D(32, (3,3)))
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

    model.add(layers.Dense(num_classes  , activation='softmax'))
    return model

def create_model_v6(config):
    feature_types = config.get("Audio data", "feature_types").split(',')
    time_axis = config.getint('Audio data', f'{feature_types[0]}_time_axis')
    k_axis = config.getint('Audio data', f'{feature_types[0]}_k_axis')
    input_shape = (time_axis, k_axis, len(feature_types))

    num_classes = config.getint('Dataset', 'num_classes')

    input = layers.Input(input_shape)

    conv = layers.Conv2D(filters = 16, kernel_size = (3,3), padding = 'same')(input)
    activation = layers.Activation(activations.relu)(conv)
    max_pool = layers.MaxPooling2D((2, 2))(activation)

    conv = layers.Conv2D(filters = 32, kernel_size = (3,3), padding = 'same')(max_pool)
    activation = layers.Activation(activations.relu)(conv)
    max_pool = layers.MaxPooling2D((2, 2))(activation)

    shape = max_pool.shape.as_list()
    reshape = layers.Reshape((4, int(shape[1]/4), shape[2], shape[3]))(max_pool)

    conv_lstm_1 = layers.ConvLSTM2D(32, (3,3), name="conv_lstm_1", return_sequences=True)(reshape)

    time_distributed_pool = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(conv_lstm_1)

    conv_lstm_2 = layers.ConvLSTM2D(32, (3,3), name="conv_lstm_2", return_sequences=False)(time_distributed_pool)

    flatten = layers.Flatten()(conv_lstm_2)
    dense = layers.Dense(16, activation='relu')(flatten)
    output = layers.Dense(num_classes, activation='softmax')(dense)

    model = models.Model(inputs=input, outputs=output)
    return model

def create_model_optuna(config, trial):
    feature_types = config.get("Audio data", "feature_types").split(',')
    num_classes = config.getint('Dataset', 'num_classes')

    flatten = []
    inputs = []
    for feature_type in feature_types:
        time_axis = config.getint('Audio data', f'{feature_type}_time_axis')
        k_axis = config.getint('Audio data', f'{feature_type}_k_axis')
        input = layers.Input(shape=(time_axis, k_axis, 1))
        inputs.append(input)
        num_conv_layers = trial.suggest_int(f'num_conv_layers_{feature_type}', 2, 8)
        aux = input
        for j in range(num_conv_layers):
            shape = aux.shape
            k = trial.suggest_int(f'kernel_size_{feature_type}_{j}', 1, min(shape[1], shape[2], 6))
            filters = trial.suggest_int(f'num_filters_{feature_type}_{j}', 16, 512)
            kernel_size = (k, k)

            conv = layers.Conv2D(filters, kernel_size)(aux)
            act = layers.Activation(activations.relu)(conv)

            s = trial.suggest_int(f'pooling_strides_{feature_type}_{j}', 1, 6)
            strides = (s, s)
            shape = act.shape
            pooling_size = trial.suggest_int(f'pooling_size_{feature_type}_{j}', 1, min(shape[1],shape[2],6))
            
            if shape[1]/shape[2]>1.9:
                pooling_size = (pooling_size, 1)
            else:
                pooling_size = (pooling_size, pooling_size)
            pooling = layers.MaxPooling2D(pooling_size, strides = strides)(act)

            dropout = layers.Dropout(0.10)(pooling)
            aux = dropout
        flatten.append(layers.Flatten()(dropout))

    if len(feature_types)==1:
        concat = flatten[0]
    else:
        concat = layers.Concatenate()(flatten)

    dense_units = trial.suggest_int('dense_units', 12, 36)
    dense = layers.Dense(dense_units, activation='relu')(concat)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)
    dropout = layers.Dropout(dropout_rate)(dense)
    output = layers.Dense(num_classes, activation='softmax')(dropout)
    model = models.Model(inputs=inputs, outputs=output)
    return model

def create_model_optuna_one_feature(config, trial, time_axis=None, k_axis=None):
    feature_types = config.get("Audio data", "feature_types").split(',')
    num_classes = config.getint('Dataset', 'num_classes')
    feature_type = feature_types[0]

    if time_axis is None:
        time_axis = config.getint('Audio data', f'{feature_type}_time_axis')
    if k_axis is None:
        k_axis = config.getint('Audio data', f'{feature_type}_k_axis')
    inputs = layers.Input(shape=(time_axis, k_axis, 1))

    num_conv_layers = trial.suggest_int(f'num_conv_layers', 2, 8)
    aux = inputs
    for j in range(num_conv_layers):
        shape = aux.shape
        k = trial.suggest_int(f'kernel_size_{j}', 1, min(shape[1], shape[2], 6))
        filters = trial.suggest_int(f'num_filters_{j}', 16, 512)
        kernel_size = (k, k)

        conv = layers.Conv2D(filters, kernel_size)(aux)
        act = layers.Activation(activations.relu)(conv)

        s = trial.suggest_int(f'pooling_strides_{j}', 1, 5)
        strides = (s, s)
        shape = act.shape
        pooling_size = trial.suggest_int(f'pooling_size_{j}', 1, min(shape[1],shape[2],5))
        
        if shape[1]/shape[2]>1.9:
            pooling_size = (pooling_size, 1)
        else:
            pooling_size = (pooling_size, pooling_size)
        pooling = layers.MaxPooling2D(pooling_size, strides = strides)(act)

        dropout = layers.Dropout(0.10)(pooling)
        aux = dropout

    flatten = layers.Flatten()(dropout)
    dense_units = trial.suggest_int('dense_units', 12, 36)
    dense = layers.Dense(dense_units, activation='relu')(flatten)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)
    dropout = layers.Dropout(dropout_rate)(dense)
    output = layers.Dense(num_classes, activation='softmax')(dropout)
    model = models.Model(inputs=[inputs], outputs=[output])
    return model

def create_model_optuna_lstm(config, trial):
    feature_types = config.get("Audio data", "feature_types").split(',')
    num_classes = config.getint('Dataset', 'num_classes')
    time_steps = config.getint("Model", "time_steps")

    flatten = []
    inputs = []
    for feature_type in feature_types:
        time_axis = config.getint('Audio data', f'{feature_type}_time_axis')
        k_axis = config.getint('Audio data', f'{feature_type}_k_axis')
        input = layers.Input(shape=(time_axis, k_axis, 1))
        inputs.append(input)
        num_conv_layers = trial.suggest_int(f'num_conv_layers_{feature_type}', 1, 2)
        num_conv_lstm_layers = trial.suggest_int(f'num_conv_lstm_layers_{feature_type}', 4, 8)
        aux = input
        for j in range(num_conv_layers+num_conv_lstm_layers):
            shape = aux.shape
            k = trial.suggest_int(f'kernel_size_{feature_type}_{j}', 1, min(shape[1], shape[2], 5))
            filters = trial.suggest_int(f'num_filters_{feature_type}_{j}', 8, 64)
            kernel_size = (k, k)

            if j < num_conv_layers:
                conv = layers.SeparableConv2D(filters, kernel_size, padding='same')(aux)
            else:
                if j == num_conv_layers:
                    if shape[1] % time_steps == 0:
                        print("Shapes is divisible by time_steps.")
                    else:
                        print("\nShapes is not divisible by time_steps: ")
                        print("Shape is: ", shape, " after ", num_conv_layers, " convolution layers.")
                    assert shape[1] % time_steps == 0
                    reshape = layers.Reshape((time_steps, int(shape[1]/time_steps), int(shape[2]), int(shape[3])))(aux)
                    aux = reshape
                if j == num_conv_layers+num_conv_lstm_layers-1:
                    return_sequences = False
                else:
                    return_sequences = True
                conv = layers.ConvLSTM2D(filters, kernel_size, return_sequences=return_sequences, padding='same')(aux)

            act = layers.Activation(activations.relu)(conv)

            if j>=num_conv_layers:
                    # pooling = layers.TimeDistributed(layers.MaxPooling2D(pooling_size))(act)
                pooling = act
            else:
                shape = act.shape
                s = trial.suggest_int(f'pooling_strides_{feature_type}_{j}', 1, 2)
                strides = (s, s)
                pooling_size = trial.suggest_int(f'pooling_size_{feature_type}_{j}', 1, min(shape[1],shape[2],5))
                pooling_size = (pooling_size, pooling_size)
                pooling = layers.MaxPooling2D(pooling_size, strides = strides, padding='same')(act)

            # dropout_rate = trial.suggest_float('dropout_rate_F{feature_type}_B{j}', 0, 0.2)
            dropout = layers.Dropout(0.1)(pooling)
            aux = dropout
        flatten.append(layers.Flatten()(dropout))

    if len(feature_types)==1:
        concat = flatten[0]
    else:
        concat = layers.Concatenate()(flatten)

    dense_units = trial.suggest_int('dense_units', 12, 48)
    dense = layers.Dense(dense_units, activation='relu')(concat)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.3)
    dropout = layers.Dropout(dropout_rate)(dense)
    output = layers.Dense(num_classes, activation='softmax')(dropout)
    model = models.Model(inputs=inputs, outputs=output)
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

def save_model(config, model, suffix = None):
    full_path = config.get("Model", "save_dir")
    if suffix is not None:
        full_path = full_path + suffix
    else:
        suffix = ""
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    model.save(os.path.join(full_path, "model.h5"))
    configModified = copy.deepcopy(config)
    configModified.set("Model", "load_dir", config.get("Model", "load_dir")+suffix)
    configModified.set("Model", "save_dir", config.get("Model", "save_dir")+suffix)
    
    with open(os.path.join(full_path, "config.ini"), 'w') as configfile:
        configModified.write(configfile)

def save_tfmodel(config, model, suffix = None):
    full_path = config.get("Model", "save_dir")
    if suffix is not None:
        full_path = full_path + suffix
    else:
        suffix = ""
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    with open(os.path.join(full_path, "model.tflite"), 'wb') as f:
        f.write(model)
    configModified = copy.deepcopy(config)
    configModified.set("Model", "load_dir", config.get("Model", "load_dir")+suffix)
    configModified.set("Model", "save_dir", config.get("Model", "save_dir")+suffix)
    with open(os.path.join(full_path, "config.ini"), 'w') as configfile:
        configModified.write(configfile)

def load_model(config, custom_objects = None):
        full_path = config.get("Model", "load_dir")
        model_path = os.path.join(full_path, "model.h5")
        loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        configOld = configparser.ConfigParser()
        configOld.read(os.path.join(full_path, 'config.ini'))
        #TODO copy the dirs from the new config to the old config
        # assert False
        return loaded_model, configOld

def quantize_model(config, model, representative_dataset):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    print("\nQuantization started...\n")
    quant_model = converter.convert()
    print("\nQuantization ended!\n")
    return quant_model