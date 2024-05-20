import configparser
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from audio_nn import model as model_utils, dataset as dataset_utils, callbacks
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import Precision, Recall, AUC
from keras.callbacks import LearningRateScheduler
import os

import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[3], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')
batch_size = config.getint("Training", "batch_size")

model, config = model_utils.load_model(config)

files, labels, classes, class_weights = dataset_utils.load_dataset(config)

class_weights[0] = 3
class_weights[1] = 6

files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)

augmentation_datasets = []
for augmentation_dataset in config.get("Dataset", "augmentation_dir").split(','):
    d = dataset_utils.load_dataset(config, input_dir=augmentation_dataset, files_only=True)
    augmentation_datasets.append(d)

augmentation_gen = dataset_utils.augmentation_generator([
    dataset_utils.crop_augmentation(new_length=config.getint("Audio data", "keep_samples"), p=1.0),
    dataset_utils.pitch_augmentation(max_variation = 0.05, p=0.3),
    dataset_utils.gain_augmentation(max_db=3, p=1.0),
    dataset_utils.noise_augmentation(max_noise_ratio=0.06, p=1.0),
    dataset_utils.mix_augmentation(augmentation_datasets[0], min_ratio=0.1, max_ratio=0.5, p=0.3),
    dataset_utils.mix_augmentation(augmentation_datasets[1], min_ratio=0.1, max_ratio=0.5, p=0.3),
    # dataset_utils.mix_augmentation(augmentation_datasets[2], min_ratio=0.001, max_ratio=0.4, p=0.2),
    dataset_utils.noise_augmentation(min_noise_ratio=0.01, max_noise_ratio=0.03, p=1.0),
    dataset_utils.pitch_augmentation(max_variation = 0.02, p=0.3),
    dataset_utils.gain_augmentation(max_db=2, p=1.0)
])

data_gen_train = dataset_utils.data_generator(config, files_train, labels_train, batch_size, augmentation_gen)
data_gen_val = dataset_utils.data_generator(config, files_val, labels_val, batch_size, augmentation_gen)

def apply_quantization_to_bn(layer):
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, model_utils.BatchNormQuantizeConfig())
    return layer

with tf.keras.utils.custom_object_scope({'BatchNormQuantizeConfig': model_utils.BatchNormQuantizeConfig}):
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_quantization_to_bn,
    )

quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

# with tf.keras.utils.custom_object_scope({'BatchNormQuantizeConfig': model_utils.BatchNormQuantizeConfig}):
#     q_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

q_aware_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.getfloat("Training", "lr")), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', Precision(), Recall()])
print(q_aware_model.summary())

save_best_callback = callbacks.SaveBestModel(config)

def lr_schedule(epoch):
    if epoch < 50:
        return 5e-5
    elif epoch < 100:
        return 1e-5
    else:
        return 1e-6
lr_scheduler = LearningRateScheduler(lr_schedule)

q_aware_model.fit_generator(generator = data_gen_train,
                        steps_per_epoch = len(files_train)//batch_size,
                        validation_data = data_gen_val,
                        validation_steps = len(files_val)//batch_size,
                        epochs = config.getint("Training", "epochs"),
                        verbose = 1,
                        callbacks=[save_best_callback, lr_scheduler],
                        class_weight=class_weights)

model_utils.save_model(config, q_aware_model)