import configparser
import tensorflow as tf
from audio_nn import model as model_utils, dataset as dataset_utils, callbacks
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.keras import models, layers, activations
from tensorflow.keras.regularizers import l1
import tensorflow_hub as hub
import tensorflow_model_optimization as tfmot
import optuna

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[3], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')

keep = config.getfloat("Optuna", "keep_files")
sqlite_url = config.get("Optuna", "study_file")
epochs = config.getint("Optuna", "epochs")
study_name = config.get("Optuna", "study_name")
batch_size = config.getint("Optuna", "batch_size")

feature_types = 'stft'
config.set("Audio data", "feature_types", feature_types)
for feature_type in feature_types.split(','):
    time_axis_name = f'{feature_type}_time_axis'
    k_axis_name = f'{feature_type}_k_axis'
    time_axis = 32
    k_axis = 256
    config.set("Audio data", time_axis_name, str(time_axis))
    config.set("Audio data", k_axis_name, str(k_axis))

def objective(trial):

    MAX_PARAMS = 500000
    MIN_PARAMS = 200000

    keep_samples = 6144

    config.set("Audio data", "keep_samples", str(keep_samples))

    files, labels, classes, class_weights_dict = dataset_utils.load_dataset(config, keep = keep)

    class_weights_dict[0] = 1.2
    class_weights_dict[1] = 7

    files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)

    augmentation_datasets = []
    for augmentation_dataset in config.get("Dataset", "augmentation_dir").split(','):
        d = dataset_utils.load_dataset(config, input_dir=augmentation_dataset, files_only=True)
        augmentation_datasets.append(d)

    augmentation_gen = dataset_utils.augmentation_generator([
        dataset_utils.crop_augmentation(new_length=config.getint("Audio data", "keep_samples"), p=1.0),
        dataset_utils.gain_augmentation(max_db=3, p=1.0),
        dataset_utils.noise_augmentation(max_noise_ratio=0.06, p=1.0),
        dataset_utils.mix_augmentation(augmentation_datasets[0], min_ratio=0.001, max_ratio=0.25, p=0.4),
        dataset_utils.mix_augmentation(augmentation_datasets[1], min_ratio=0.001, max_ratio=0.25, p=0.4),
        dataset_utils.mix_augmentation(augmentation_datasets[2], min_ratio=0.001, max_ratio=0.4, p=0.4),
        dataset_utils.noise_augmentation(min_noise_ratio=0.01, max_noise_ratio=0.03, p=1.0),
        dataset_utils.gain_augmentation(max_db=2, p=1.0)
    ])


    data_gen_train = dataset_utils.data_generator(config, files_train, labels_train, batch_size, augmentation_gen, quantize=False)
    data_gen_val = dataset_utils.data_generator(config, files_val, labels_val, batch_size, augmentation_gen, quantize=False)

    try:
        model = model_utils.create_model_optuna(config, trial)
    except Exception as e:
        # trial.report(value=None, step=0)
        # trial.set_user_attr('failed...', str(e))
        print("\nSomething's wrong: ")
        print(e)
        trial.set_user_attr('failed_reason', str(e))
        return 3
    
    if model is None:
        trial.set_user_attr('failed_reason', f'Tensor too large.')
        return 3.01

    # quantize_model = tfmot.quantization.keras.quantize_model

    # model = quantize_model(model)

    lr = 1e-5
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
    # k_loss = last_val_loss - penalty/1.5
    k_loss = (last_val_loss + last_train_loss)/2

    val_accuracy = history.history['val_accuracy']
    train_accuracy = history.history['accuracy']
    last_val_accuracy = val_accuracy[-1]
    last_train_accuracy = train_accuracy[-1]

    penalty = (last_train_accuracy - last_val_accuracy)*1
    if penalty < 0:
        penalty = 0
    k_accuracy = last_val_accuracy - penalty/1.5


    k = k_loss
    # k = k_loss + (1-k_accuracy)

    trial.set_user_attr('Number of parameters', str(model.count_params()))
    return last_val_loss

study = optuna.create_study(direction='minimize', study_name = study_name, storage=sqlite_url, load_if_exists=True)
study.optimize(objective, n_trials = 500)

# optuna.visualization.plot_optimization_history(study)
# optuna.visualization.plot_contour(study, params=['param1', 'param2'])