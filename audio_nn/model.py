# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:48:56 2023

@author: Ruben B
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os
from tensorflow.keras import models, layers, activations
from tensorflow.keras.regularizers import l1
import tensorflow_hub as hub
import ssl
import configparser
import copy
import math
import numpy as np

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

def create_model_v999(config):
    input_shape = (32, 256, 1)
    model = models.Sequential()


    model.add(layers.Conv2D(64, (5, 5), input_shape=input_shape, padding = 'same'))
    model.add(layers.Activation(activations.relu))

    model.add(layers.MaxPooling2D((3, 3)))

    model.add(layers.Dropout(0.10))


    model.add(layers.Conv2D(128, (3, 3), padding = 'same'))
    model.add(layers.Activation(activations.relu))

    model.add(layers.MaxPooling2D((3, 3)))

    model.add(layers.Dropout(0.10))


    model.add(layers.Conv2D(128, (3, 3), padding = 'same'))
    model.add(layers.Activation(activations.relu))

    model.add(layers.MaxPooling2D((3, 3)))

    model.add(layers.Dropout(0.10))


    model.add(layers.Conv2D(256, (1, 3), padding = 'same'))
    model.add(layers.Activation(activations.relu))

    model.add(layers.MaxPooling2D((1, 3)))

    model.add(layers.Dropout(0.10))


    model.add(layers.Flatten())

    model.add(layers.Dense(32, activation='relu'))

    model.add(layers.Dropout(0.20))

    model.add(layers.Dense(2  , activation='softmax'))
    return model

max_tensor_size = 500000 # should be higher
# max_tensor_size = 350000
max_parameters_per_layer = 100000 # should be lower
# max_parameters_per_layer = 90000
max_kernel_size = 4
min_kernel_size_pool = 2
min_kernel_size_conv = 2
min_stride_pool = 2
min_stride_conv = 1
max_dense_links = 3072
include_batch_norm = False
# max_dense_links = 2048

def create_model_optuna_v2(config, trial):
    time_axis = config.getint('Audio data', f'stft_time_axis')
    k_axis = config.getint('Audio data', f'stft_k_axis')
    input_shape = (time_axis, k_axis, 1)

    model = models.Sequential()
    
    k1 = trial.suggest_int("conv1_k1", min_kernel_size_conv, max_kernel_size)
    k2 = trial.suggest_int("conv1_k2", min_kernel_size_conv, max_kernel_size)
    stride1 = trial.suggest_int("conv1_stride1", min(time_axis, min_stride_conv), k1)
    stride2 = trial.suggest_int("conv1_stride2", min(k_axis, min_stride_conv), k2)
    time_axis = math.ceil(time_axis / stride1)
    k_axis = math.ceil(k_axis / stride2)
    max_filter_number = min(math.floor(max_tensor_size / time_axis / k_axis),
                            512)
    filters = trial.suggest_int("conv1_filters", 0, max(1, math.floor((max_filter_number - 2) / 8)))
    filters = (filters * 8) + 2
    model.add(layers.Conv2D(filters, (k1, k2), (stride1, stride2), input_shape=input_shape, padding = 'same', activation = 'relu'))

    if(include_batch_norm):
        model.add(layers.BatchNormalization())

    pool_type = trial.suggest_categorical("pool_type_1", ['max', 'avg'])
    k1 = trial.suggest_int("pool1_k1", min(time_axis, min_kernel_size_pool), min(time_axis, max_kernel_size))
    k2 = trial.suggest_int("pool1_k2", min(k_axis, min_kernel_size_pool), min(k_axis, max_kernel_size))
    stride1 = trial.suggest_int("pool1_stride1", min(time_axis, min_stride_pool), k1)
    stride2 = trial.suggest_int("pool1_stride2", min(k_axis, min_stride_pool), k2)
    if pool_type == 'max':
        model.add(layers.MaxPooling2D((k1, k2), strides = (stride1, stride2), padding = 'same'))
    elif pool_type == 'avg':
        model.add(layers.AveragePooling2D((k1, k2), strides = (stride1, stride2), padding = 'same'))

    time_axis = math.floor((time_axis - 1) / stride1) + 1
    k_axis = math.floor((k_axis - 1) / stride2) + 1

    # model.add(layers.Dropout(0.10))


    k1 = trial.suggest_int("conv2_k1", min(time_axis, min_kernel_size_conv), max_kernel_size)
    k2 = trial.suggest_int("conv2_k2", min(k_axis, min_kernel_size_conv), max_kernel_size)
    stride1 = trial.suggest_int("conv2_stride1", min(time_axis, min_stride_conv), k1)
    stride2 = trial.suggest_int("conv2_stride2", min(k_axis, min_stride_conv), k2)
    time_axis = math.ceil(time_axis / stride1)
    k_axis = math.ceil(k_axis / stride2)
    max_filter_number = min(math.floor(max_parameters_per_layer/(k1*k2*filters+1)),
                            int(max_tensor_size / time_axis / k_axis),
                            512) 
    # print("max filters: ", int(max_filter_number))
    filters = trial.suggest_int("conv2_filters", 0, max(1, math.floor((max_filter_number - 2) / 8)))
    filters = (filters * 8) + 2 # TODO make this work
    model.add(layers.Conv2D(filters, (k1, k2), (stride1, stride2), padding = 'same', activation = 'relu'))
    
    if(include_batch_norm):
        model.add(layers.BatchNormalization())

    pool_type = trial.suggest_categorical("pool_type_2", ['max', 'avg'])
    k1 = trial.suggest_int("pool2_k1", min(time_axis, min_kernel_size_pool), min(time_axis, max_kernel_size))
    k2 = trial.suggest_int("pool2_k2", min(k_axis, min_kernel_size_pool), min(k_axis, max_kernel_size))
    stride1 = trial.suggest_int("pool2_stride1", min(time_axis, min_stride_pool), k1)
    stride2 = trial.suggest_int("pool2_stride2", min(k_axis, min_stride_pool), k2)
    if pool_type == 'max':
        model.add(layers.MaxPooling2D((k1, k2), strides = (stride1, stride2), padding = 'same'))
    elif pool_type == 'avg':
        model.add(layers.AveragePooling2D((k1, k2), strides = (stride1, stride2), padding = 'same'))

    time_axis = math.floor((time_axis - 1) / stride1) + 1
    k_axis = math.floor((k_axis - 1) / stride2) + 1

    # model.add(layers.Dropout(0.10))



    k1 = trial.suggest_int("conv3_k1", min(time_axis, min_kernel_size_conv), max_kernel_size)
    k2 = trial.suggest_int("conv3_k2", min(k_axis, min_kernel_size_conv), max_kernel_size)
    stride1 = trial.suggest_int("conv3_stride1", min(time_axis, min_stride_conv), k1)
    stride2 = trial.suggest_int("conv3_stride2", min(k_axis, min_stride_conv), k2)
    time_axis = math.ceil(time_axis / stride1)
    k_axis = math.ceil(k_axis / stride2)
    max_filter_number = min(math.floor(max_parameters_per_layer/(k1*k2*filters+1)),
                            int(max_tensor_size / time_axis / k_axis),
                            512)
    # print("max filters: ", math.floor(max_filter_number))
    filters = trial.suggest_int("conv3_filters", 0, max(1, math.floor((max_filter_number - 2) / 8)))
    filters = (filters * 8) + 2
    model.add(layers.Conv2D(filters, (k1, k2), (stride1, stride2), padding = 'same', activation = 'relu'))

    if(include_batch_norm):
        model.add(layers.BatchNormalization())

    pool_type = trial.suggest_categorical("pool_type_3", ['max', 'avg'])
    k1 = trial.suggest_int("pool3_k1", min(time_axis, min_kernel_size_pool), min(time_axis, max_kernel_size))
    k2 = trial.suggest_int("pool3_k2", min(k_axis, min_kernel_size_pool), min(k_axis, max_kernel_size))
    stride1 = trial.suggest_int("pool3_stride1", min(time_axis, min_stride_pool), k1)
    stride2 = trial.suggest_int("pool3_stride2", min(k_axis, min_stride_pool), k2)
    if pool_type == 'max':
        model.add(layers.MaxPooling2D((k1, k2), strides = (stride1, stride2), padding = 'same'))
    elif pool_type == 'avg':
        model.add(layers.AveragePooling2D((k1, k2), strides = (stride1, stride2), padding = 'same'))

    time_axis = math.floor((time_axis - 1) / stride1) + 1
    k_axis = math.floor((k_axis - 1) / stride2) + 1

    # model.add(layers.Dropout(0.10))


    

    pool_type = trial.suggest_categorical("pool_type_4", ['max', 'avg'])

    k1 = trial.suggest_int("pool4_k1", min(min_kernel_size_pool, time_axis), min(time_axis, max_kernel_size))
    k2 = trial.suggest_int("pool4_k2", min(min_kernel_size_pool, k_axis), min(k_axis, max_kernel_size))
    
    stride1 = trial.suggest_int("pool4_stride1", min(time_axis, min_stride_pool), k1)
    stride2 = trial.suggest_int("pool4_stride2", min(k_axis, min_stride_pool), k2)

    k1conv = trial.suggest_int("conv4_k1", min(time_axis, min_kernel_size_conv), max_kernel_size)
    k2conv = trial.suggest_int("conv4_k2", min(k_axis, min_kernel_size_conv), max_kernel_size)
    max_filter_number =  min(math.floor(max_tensor_size / time_axis / k_axis),
                       math.floor(max_dense_links/(math.floor((time_axis - 1) / stride1) + 1)/(math.floor((k_axis - 1) / stride2) + 1)),
                       math.floor(max_parameters_per_layer/(k1conv*k2conv*filters+1)),
                       512)
    # print("max filters (tensor size): ", math.floor(max_tensor_size / time_axis / k_axis))
    # print("max filters (dense links): ", math.floor(max_dense_links/(math.floor((time_axis - 1) / stride1) + 1)/(math.floor((k_axis - 1) / stride2) + 1)))
    # print("max filters (parameters): ", math.floor(max_parameters_per_layer/(k1conv*k2conv*filters+1)))
    filters = trial.suggest_int("conv4_filters", 0, max(1, math.floor((max_filter_number - 2) / 8)))
    filters = (filters * 8) + 2
    model.add(layers.Conv2D(filters, (k1conv, k2conv), padding = 'same', activation = 'relu'))

    if(include_batch_norm):
        model.add(layers.BatchNormalization())

    if pool_type == 'max':
        model.add(layers.MaxPooling2D((k1, k2), strides = (stride1, stride2), padding = 'same'))
    elif pool_type == 'avg':
        model.add(layers.AveragePooling2D((k1, k2), strides = (stride1, stride2), padding = 'same'))

    time_axis = math.floor((time_axis - 1) / stride1) + 1
    k_axis = math.floor((k_axis - 1) / stride2) + 1

    total_size = time_axis * k_axis * filters

    # model.add(layers.Dropout(0.10))


    model.add(layers.Flatten())

    max_dense_nodes = max_dense_links/total_size
    dense_nodes = trial.suggest_int("dense", 4, max(4, max_dense_nodes))
    model.add(layers.Dense(dense_nodes, activation='relu'))

    if(include_batch_norm):
        model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(2, activation='softmax'))


    # new_model = models.Sequential()

    # skip_index = [i for i, l in enumerate(model.layers) if isinstance(l, layers.MaxPooling2D)][3]

    # for i, layer in enumerate(model.layers):
    #     if i != skip_index:
    #         new_model.add(layer)

    # for i, layer in enumerate(new_model.layers):
    #     original_index = i if i < skip_index else i + 1
    #     layer.set_weights(model.layers[original_index].get_weights())



    return model

def create_model_optuna(config, trial, optimizing = True):
    feature_types = config.get("Audio data", "feature_types").split(',')
    num_classes = config.getint('Dataset', 'num_classes')

    flatten = []
    inputs = []
    for feature_type in feature_types:
        time_axis = time_axis = config.getint('Audio data', f'{feature_type}_time_axis')
        k_axis = config.getint('Audio data', f'{feature_type}_k_axis')
        input = layers.Input(shape=(time_axis, k_axis, 1))
        inputs.append(input)
        num_conv_layers = trial.suggest_int(f'num_conv_layers_{feature_type}', 2, 8)
        aux = input
        for j in range(num_conv_layers):
            shape = aux.shape
            k = trial.suggest_int(f'kernel_size_{feature_type}_{j}', 3, max(3, min(shape[1], shape[2], 9)))
            filters = trial.suggest_int(f'num_filters_{feature_type}_{j}', 16, 384)
            # strides = trial.suggest_int(f'stride_{feature_type}_{j}', 1, 5)
            strides = 1

            if shape[2]/shape[1] >= 3.0 and shape[1] == 1 :
                kernel_size = (1, k)
            else:
                kernel_size = (k, k)

            conv = layers.Conv2D(filters, kernel_size, strides)(aux)

            act = layers.Activation(activations.relu)(conv)

            if ((shape[1]-k/2)/strides)*((shape[2]-k/2)/strides)*filters > 800000:
                print("Tensor size: ", (shape[1]-k/2)*(shape[2]-k)*filters)
                return None
                
            
            shape = act.shape
            s = trial.suggest_int(f'pooling_strides_{feature_type}_{j}', 1, 6)
            pooling_size = trial.suggest_int(f'pooling_size_{feature_type}_{j}', s, min(shape[1],shape[2],6))
            
            
            strides = (s, s)

            if shape[2]/shape[1]>1.9:
                pooling_size = (1, pooling_size)
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
    if optimizing:
        dropout_rate = trial.suggest_float('dropout_rate', 0, 0.3)
    else:
        dropout_rate = 0.1
    dropout = layers.Dropout(dropout_rate)(dense)
    output = layers.Dense(num_classes, activation='softmax')(dropout)
    model = models.Model(inputs=inputs, outputs=output)
    return model

def create_model_optuna_depthwise(config, trial):
    feature_types = config.get("Audio data", "feature_types").split(',')
    num_classes = config.getint('Dataset', 'num_classes')

    flatten = []
    inputs = []
    for feature_type in feature_types:
        time_axis = time_axis = config.getint('Audio data', f'{feature_type}_time_axis')
        k_axis = config.getint('Audio data', f'{feature_type}_k_axis')
        input = layers.Input(shape=(time_axis, k_axis, 1))
        inputs.append(input)
        num_conv_layers = trial.suggest_int(f'num_conv_layers_{feature_type}', 2, 8)
        aux = input
        for j in range(num_conv_layers):
            shape = aux.shape
            k = trial.suggest_int(f'kernel_size_{feature_type}_{j}', 1, min(shape[1], shape[2], 6))
            # filters = trial.suggest_int(f'num_filters_{feature_type}_{j}', 16, 400)

            max_depthwise_multiplier = 1000000 / ((shape[1]-k/2)*(shape[2]-k/2))

            depth_multiplier = trial.suggest_int(f'depth_multiplier_{feature_type}_{j}', 1, 4)
            kernel_size = (k, k)

            # conv = layers.Conv2D(filters, kernel_size)(aux)
            conv = layers.DepthwiseConv2D(kernel_size, depth_multiplier = depth_multiplier)(aux)

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
        configOld["Model"]["save_dir"] = config.get("Model", "save_dir")
        # assert False
        return loaded_model, configOld

def load_tflite_model(config):
    full_path = config.get("Model", "load_dir")
    model_path = os.path.join(full_path, "model.tflite")
    # Create an interpreter for the TFLite model
    loaded_model = tf.lite.Interpreter(model_path=model_path)
    loaded_model.allocate_tensors()  # Allocate tensors to the interpreter

    configOld = configparser.ConfigParser()
    configOld.read(os.path.join(full_path, 'config.ini'))
    #TODO: copy the dirs from the new config to the old config
    configOld["Model"]["save_dir"] = config.get("Model", "save_dir")

    return loaded_model, configOld

def predict_tflite(interpreter, input_data):
    # Get input and output details from the interpreter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check the expected datatype of the input tensor and adjust if necessary
    if input_details[0]['dtype'] == np.float32:
        input_scale, input_zero_point = input_details[0]['quantization']
        if input_scale != 0:
            input_data = input_data / input_scale + input_zero_point

    # Set the model's input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Extract the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # If output data is quantized, convert it back to float32
    if output_details[0]['dtype'] == np.int8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data - output_zero_point) * output_scale
        output_data = output_data.astype(np.float32)

    return output_data

def quantize_model(config, model, representative_dataset):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    print("\nQuantization started...\n")
    quant_model = converter.convert()
    print("\nQuantization ended!\n")
    return quant_model

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5, 2.0]) # for example
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss, optimizer='adam')
    """
    weights = tf.keras.backend.variable(weights)
    
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = y_true * tf.keras.backend.log(y_pred) * weights
        loss = -tf.keras.backend.sum(loss, -1)
        return loss

    return loss

class BatchNormQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [
            (layer.gamma, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
                num_bits=8, per_axis=False, symmetric=False, narrow_range=False)),
            (layer.beta, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
                num_bits=8, per_axis=False, symmetric=False, narrow_range=False))
        ]

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        layer.gamma, layer.beta = quantize_weights

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]

    def get_config(self):
        return {}