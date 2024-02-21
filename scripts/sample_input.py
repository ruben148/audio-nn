import configparser
import tensorflow as tf
from audio_nn import model as model_utils, dataset as dataset_utils
from sklearn.model_selection import train_test_split
from audio_nn import callbacks as callbacks
import tensorflow_model_optimization as tfmot
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[2], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')

with tfmot.quantization.keras.quantize_scope():
    model, config = model_utils.load_model(config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.getfloat("Training", "lr")), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
print(model.summary())

files, labels, classes, class_weights = dataset_utils.load_dataset(config)

augmentation_datasets = []
for augmentation_dataset in config.get("Dataset", "augmentation_dir").split(','):
    d = dataset_utils.load_dataset(config, input_dir=augmentation_dataset, files_only=True)
    augmentation_datasets.append(d)

files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)

batch_size = config.getint("Training", "batch_size")

augmentation_gen = dataset_utils.augmentation_generator([
    dataset_utils.crop_augmentation(new_length=config.getint("Audio data", "keep_samples"), p=1.0),
    dataset_utils.gain_augmentation(max_db=5, p=0.8),
    dataset_utils.noise_augmentation(max_noise_ratio=0.08, p=0.6),
    dataset_utils.mix_augmentation(augmentation_datasets[0], p=0.35),
    dataset_utils.mix_augmentation(augmentation_datasets[1], p=0.35),
    dataset_utils.noise_augmentation(min_noise_ratio=0.01, max_noise_ratio=0.05, p=0.6),
    dataset_utils.gain_augmentation(max_db=5)
])

data_gen_train = dataset_utils.data_generator(config, files_train, labels_train, batch_size, augmentation_gen)
data_gen_val = dataset_utils.data_generator(config, files_val, labels_val, batch_size, augmentation_gen)

images, samples = next(data_gen_val)

image_to_save = np.array(images[0][0])

image_flat = image_to_save.flatten()

outputs = model(np.reshape(image_to_save, (1,32,179,1)))

print(outputs)

with open("/home/buu3clj/radar_ws/sample_image.bin", "wb") as file:
    file.write(image_flat)