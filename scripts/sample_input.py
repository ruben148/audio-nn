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
    model, config = model_utils.load_tflite_model(config)

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.getfloat("Training", "lr")), 
#                 loss='categorical_crossentropy', 
#                 metrics=['accuracy'])
# print(model.summary())

files, labels, classes, class_weights = dataset_utils.load_dataset(config)

augmentation_datasets = []
for augmentation_dataset in config.get("Dataset", "augmentation_dir").split(','):
    d = dataset_utils.load_dataset(config, input_dir=augmentation_dataset, files_only=True)
    augmentation_datasets.append(d)

files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.2, random_state=42, stratify=labels)

batch_size = config.getint("Training", "batch_size")

# augmentation_gen = dataset_utils.augmentation_generator([
#     dataset_utils.crop_augmentation(new_length=config.getint("Audio data", "keep_samples"), p=1.0),
#     dataset_utils.gain_augmentation(max_db=3, p=1.0),
#     dataset_utils.noise_augmentation(max_noise_ratio=0.06, p=1.0),
#     dataset_utils.mix_augmentation(augmentation_datasets[0], min_ratio=0.001, max_ratio=0.25, p=0.4),
#     dataset_utils.mix_augmentation(augmentation_datasets[1], min_ratio=0.001, max_ratio=0.25, p=0.4),
#     dataset_utils.mix_augmentation(augmentation_datasets[2], min_ratio=0.001, max_ratio=0.25, p=0.4),
#     dataset_utils.noise_augmentation(min_noise_ratio=0.01, max_noise_ratio=0.03, p=1.0),
#     dataset_utils.gain_augmentation(max_db=2, p=1.0)
# ])

data_gen_train = dataset_utils.data_generator_testing(config, files_train, labels_train, batch_size, None)
data_gen_val = dataset_utils.data_generator_testing(config, files_val, labels_val, batch_size, None)

images, samples = next(data_gen_val)

img = np.array(images[0][0])
print(img.shape)

# image_flat = image_to_save.flatten()

# outputs = model(np.reshape(image_to_save, (1,32,256,1)))

for i, stft in enumerate(images[0]):
        with open(f"/home/buu3clj/radar_ws/samples/sample_stft_{i}.bin", "wb") as file:
            file.write(stft)

for i, image_to_save in enumerate(images[0]):
    image_to_save = np.reshape(image_to_save, (1,32,256,1))
    outputs = model_utils.predict_tflite(model, image_to_save)
    print(i, outputs)

# with open("/home/buu3clj/radar_ws/sample_image_2.bin", "wb") as file:
#     file.write(image_flat)