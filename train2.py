import configparser
import tensorflow as tf
from model_utils import create_model, train_model, save_model
from dataset_utils import load_dataset, data_generator
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
tf.config.set_visible_devices(physical_devices[0], 'GPU')

config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio-nn/config.ini')


files, labels = load_dataset(config)

files_train, files_val, labels_train, labels_val = train_test_split(files, labels, test_size=0.2, random_state=42)

batch_size = config.getint("Training", "batch_size")

data_gen_train = data_generator(files_train, labels_train, batch_size, None)
data_gen_val = data_generator(files_val, labels_val, batch_size, None)

model = create_model(config)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss='BinaryCrossentropy',metrics=['accuracy'])
print(model.summary())

train_model(config, model, data_gen_train, data_gen_val, len(files_train), len(files_val))

save_model(config, model, "test.h5")