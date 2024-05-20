import os
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio
from audio_nn import audio as audio_utils

# input_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\chainsaw\\chainsaw_original")
output_dir = os.path.join("/home/buu3clj/radar_ws/datasets/skatepark")

# pos_dir = os.path.join(output_dir, 'pos')
# neg_dir = os.path.join(output_dir, 'neg')

wav_file = "/home/buu3clj/radar_ws/datasets_2_0/WE FOUND THE BEST SKATEPARK EVER.wav"

wav = audio_utils.load_wav_8k_mono(wav_file)

j=0
for i in range(0, wav.size-7168, 7168):
    w = wav[i:i+7168]
    sf.write(os.path.join(output_dir, f'AAA_{j}.wav'), w, 8000)
    j = j + 1