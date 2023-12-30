import configparser
import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from audio_nn import dataset as dataset_utils
from sklearn.metrics import confusion_matrix, classification_report

# Load TFLite model
interpreter = tflite.Interpreter(model_path='/path/to/your/model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load dataset
config = configparser.ConfigParser()
config.read('/home/buu3clj/radar_ws/audio_nn/scripts/config.ini')
files, labels, classes, class_weights = dataset_utils.load_dataset(config, keep=1, input_dir=config.get("Validation", "dir"))
batch_size = config.getint("Training", "batch_size")
test_gen = dataset_utils.data_generator(config, files, labels, batch_size)

# Run inference and collect predictions
y_pred = []
for batch in test_gen:
    input_data = np.array(batch[0], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    y_pred.extend(np.argmax(output_data, axis=1))

# Ensure y_true is in the correct format
y_true = np.argmax(labels, axis=1)

# Evaluate the model
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("Confusion Matrix:")
print(cm)

report = classification_report(y_true, y_pred)
print("Classification Report:")
print(report)