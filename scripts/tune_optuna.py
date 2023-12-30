import configparser
import tensorflow as tf
from audio_nn import model as model_utils, dataset as dataset_utils, callbacks
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.keras import models, layers, activations
from tensorflow.keras.regularizers import l1
import tensorflow_hub as hub
import optuna

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[2], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')

keep = config.getfloat("Optuna", "keep_files")
sqlite_url = config.get("Optuna", "study_file")
epochs = config.getint("Optuna", "epochs")
study_name = config.get("Optuna", "study_name")
batch_size = config.getint("Optuna", "batch_size")

def objective(trial):

    MAX_PARAMS = 250000
    MIN_PARAMS = 150000

    keep_samples = '12288'

    config.set("Audio data", "keep_samples", keep_samples)

    files, labels, classes, class_weights_dict = dataset_utils.load_dataset(config, keep=keep)

    files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)

    feature_types = 'stft'
    config.set("Audio data", "feature_types", feature_types)
    for feature_type in feature_types.split(','):
        time_axis_name = f'{feature_type}_time_axis'
        k_axis_name = f'{feature_type}_k_axis'
        time_axis = trial.suggest_categorical(time_axis_name, [32, 64, 128, 256, 512])
        k_axis = trial.suggest_int(k_axis_name, 8, (128 if feature_type == "mfcc" else 512))
        config.set("Audio data", time_axis_name, str(time_axis))
        config.set("Audio data", k_axis_name, str(k_axis))

    
    data_gen_train = dataset_utils.data_generator(config, files_train, labels_train, batch_size, None, quantize=False)
    data_gen_val = dataset_utils.data_generator(config, files_val, labels_val, batch_size, None, quantize=False)

    try:
        model = model_utils.create_model_optuna(config, trial)
    except Exception as e:
        # trial.report(value=None, step=0)
        # trial.set_user_attr('failed...', str(e))
        print("\nSomething's wrong: ")
        print(str(e))
        trial.set_user_attr('failed_reason', str(e))
        return 3

    lr = 15e-6
    config.set("Training", "lr", str(lr))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.getfloat("Training", "lr")), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    print("Model summary: ", model.summary())

    num_params = model.count_params()
    if num_params > MAX_PARAMS or num_params < MIN_PARAMS:
        trial.set_user_attr('failed_reason', f'Too many/few parameters: {num_params}')
        return 3

    early_stopping_callback = callbacks.EarlyStoppingMixedCriteria()

    try:
        history = model.fit(data_gen_train,
                        steps_per_epoch=len(files_train) // batch_size,
                        validation_data=data_gen_val,
                        validation_steps=len(files_val) // batch_size,
                        epochs=epochs, 
                        class_weight=class_weights_dict,
                        verbose=1,
                        callbacks=[early_stopping_callback])
    except Exception as e:
        print("\nSomething's wrong: ")
        print(str(e))
        trial.set_user_attr('failed_reason', str(e))
        return 3

    if early_stopping_callback.stopped_early:
        return 2.99

    val_loss = history.history['val_loss']
    train_loss = history.history['loss']
    last_val_loss = val_loss[-1]
    last_train_loss = train_loss[-1]

    penalty = (last_train_loss - last_val_loss)*1
    if penalty > 0:
        penalty = 0
    k_loss = last_val_loss - penalty/1.5

    val_accuracy = history.history['val_accuracy']
    train_accuracy = history.history['accuracy']
    last_val_accuracy = val_accuracy[-1]
    last_train_accuracy = train_accuracy[-1]

    penalty = (last_train_accuracy - last_val_accuracy)*1
    if penalty < 0:
        penalty = 0
    k_accuracy = last_val_accuracy - penalty/1.5


    k = k_loss + (1-k_accuracy)

    trial.set_user_attr('Number of parameters', str(model.count_params()))
    return k

study = optuna.create_study(direction='minimize', study_name = study_name, storage=sqlite_url, load_if_exists=True)
study.optimize(objective, n_trials = 100000)

optuna.visualization.plot_optimization_history(study)
# optuna.visualization.plot_contour(study, params=['param1', 'param2'])