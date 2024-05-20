import configparser
import tensorflow as tf
from audio_nn import model as model_utils, dataset as dataset_utils, callbacks
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.keras import models, layers, activations
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.regularizers import l1
import tensorflow_hub as hub
import tensorflow_model_optimization as tfmot
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

files, labels, classes, class_weights_dict = dataset_utils.load_dataset(config, keep = keep)

class_weights_dict[0] = 2
class_weights_dict[1] = 6

files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.2, random_state=523454, stratify=labels)

print(np.size(files_train))

augmentation_datasets = []
for augmentation_dataset in config.get("Dataset", "augmentation_dir").split(','):
    d = dataset_utils.load_dataset(config, input_dir=augmentation_dataset, files_only=True)
    augmentation_datasets.append(d)

augmentation_gen = dataset_utils.augmentation_generator([
    dataset_utils.crop_augmentation(new_length=config.getint("Audio data", "keep_samples"), p=1.0),
    # dataset_utils.pitch_augmentation(max_variation = 0.05, p=0.3),
    dataset_utils.gain_augmentation(max_db=3, p=1.0),
    dataset_utils.noise_augmentation(max_noise_ratio=0.06, p=1.0),
    dataset_utils.mix_augmentation(augmentation_datasets[0], min_ratio=0.001, max_ratio=0.25, p=0.3),
    dataset_utils.mix_augmentation(augmentation_datasets[1], min_ratio=0.001, max_ratio=0.25, p=0.3),
    # dataset_utils.mix_augmentation(augmentation_datasets[2], min_ratio=0.001, max_ratio=0.4, p=0.4),
    dataset_utils.noise_augmentation(min_noise_ratio=0.01, max_noise_ratio=0.03, p=1.0),
    # dataset_utils.pitch_augmentation(max_variation = 0.02, p=0.3),
    dataset_utils.gain_augmentation(max_db=2, p=1.0)
])

# augmentation_gen = dataset_utils.augmentation_generator([
#     dataset_utils.crop_augmentation(new_length=config.getint("Audio data", "keep_samples"), p=1.0)
# ])

def objective(trial):

    # keep_samples = trial.suggest_categorical("keep_samples", range(5120, 7169, 128))
    keep_samples = config.getint("Audio data", "keep_samples")
    # config.set("Audio data", "keep_samples", str(keep_samples))

    time_axis = trial.suggest_categorical("time_axis", range(32, 64+1, 8))
    k_axis = trial.suggest_categorical("k_axis", [128, 256])

    config.set("Audio data", "stft_time_axis", str(time_axis))
    config.set("Audio data", "stft_k_axis", str(k_axis))

    data_gen_train = dataset_utils.data_generator(config, files_train, labels_train, batch_size, augmentation_gen, quantize=False)
    data_gen_val = dataset_utils.data_generator(config, files_val, labels_val, batch_size, augmentation_gen, quantize=False)

    try:
        model = model_utils.create_model_optuna_v2(config, trial)
    except:
        return 3

    lr = 5e-5
    config.set("Training", "lr", str(lr))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', Precision(), Recall()])
    
    print(model.summary())

    params = model.count_params()
    # if params >300000:
    #     print("Model summary: ", model.summary())
        # input("Number of parameters is over 300000.")

    # return 1

    early_stopping_callback = callbacks.EarlyStoppingMixedCriteria()

    history = model.fit(data_gen_train,
                        steps_per_epoch=len(files_train) // batch_size,
                        validation_data=data_gen_val,
                        validation_steps=len(files_val) // batch_size,
                        epochs=epochs, 
                        class_weight=class_weights_dict,
                        verbose=1,
                        callbacks=[
                            # early_stopping_callback
                            ])

    all_metrics = model.evaluate(data_gen_val, steps = 50, verbose=1)
    precision_metric = all_metrics[3]
    print(all_metrics)

    breakpoint()

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    avg_loss = (loss[-1] + val_loss[-1] * 2) / 2

    # return avg_loss
    return 1/precision_metric

study = optuna.create_study(direction='minimize', study_name = study_name, storage=sqlite_url, load_if_exists=True)
study.optimize(objective, n_trials = 500)