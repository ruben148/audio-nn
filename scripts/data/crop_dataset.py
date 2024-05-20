import os
import numpy as np
import random

dir = os.path.join("/home/buu3clj/radar_ws/datasets_2_0/chainsaw/train/7168/neg")

wav_files = [filename for filename in os.listdir(dir) if filename.endswith('.wav')]

random.seed(1234567)
random.shuffle(wav_files)
random.seed(7654)
random.shuffle(wav_files)
random.seed(54376)
random.shuffle(wav_files)

print(len(wav_files))

wav_files = wav_files[5000:]
print(len(wav_files))
for wav_file in wav_files:
    os.remove(os.path.join(dir, wav_file))