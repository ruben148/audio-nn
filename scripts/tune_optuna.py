import configparser
import tensorflow as tf
from model_utils import create_model_optuna
from dataset_utils import load_dataset, data_generator_multi_shape, representative_dataset_gen
from sklearn.model_selection import train_test_split
from callbacks import EarlyStoppingAccuracy
import numpy as np
import os
from tensorflow.keras import models, layers, activations
from tensorflow.keras.regularizers import l1
import tensorflow_hub as hub
import optuna

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[1], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio-nn/config.ini')

keep = config.getfloat("Optuna", "keep_files")
sqlite_url = config.get("Optuna", "study_file")
epochs = config.getint("Optuna", "epochs")
study_name = config.get("Optuna", "study_name")

files, labels, classes, class_weights_dict = load_dataset(config, keep=keep)

files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)

batch_size = config.getint("Training", "batch_size")

def objective(trial):

    MAX_PARAMS = 200000

    batch_size = config.getint("Optuna", "batch_size")
    data_gen_train = data_generator_multi_shape(config, files_train, labels_train, batch_size, None)
    data_gen_val = data_generator_multi_shape(config, files_val, labels_val, batch_size, None)

    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    try:
        model = create_model_optuna(config, trial)
    except Exception as e:
        # trial.report(value=None, step=0)
        # trial.set_user_attr('failed...', str(e))
        print("\nSomething's wrong: ")
        print(str(e))
        trial.set_user_attr('failed_reason', str(e))
        return 0

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    print("Model summary: ", model.summary())

    num_params = model.count_params()
    if num_params > MAX_PARAMS:
        trial.set_user_attr('failed_reason', f'Too many parameters: {num_params}')
        return 0

    early_stopping_acc = EarlyStoppingAccuracy(threshold=0.5, patience=3)

    history = model.fit(data_gen_train,
                        steps_per_epoch=len(files_train) // batch_size,
                        validation_data=data_gen_val,
                        validation_steps=len(files_val) // batch_size,
                        epochs=epochs, 
                        class_weight=class_weights_dict,
                        verbose=1,
                        callbacks=[early_stopping_acc])

    val_accuracies = history.history['val_accuracy']
    train_accuracies = history.history['accuracy']
    last_val_accuracy = val_accuracies[-1]
    last_train_accuracy = train_accuracies[-1]

    penalty = (last_train_accuracy - last_val_accuracy)*1.5

    return last_val_accuracy - penalty

study = optuna.create_study(direction='maximize', study_name = study_name, storage=sqlite_url, load_if_exists=True)
study.optimize(objective, n_trials = 1000)

optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_contour(study, params=['param1', 'param2'])