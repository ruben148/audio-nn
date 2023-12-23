# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:05:03 2023

@author: Ruben
"""

import textgrid
import os
import librosa
import numpy as np
import soundfile as sf
from random import randrange
import tensorflow as tf
import tensorflow_io as tfio

#%%

input_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\esc50\\esc50_original")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\esc50\\esc50_12288") #12288 24576 49152

#%%

wav_files = [filename for filename in os.listdir(input_dir) if filename.endswith('.wav')]

wav_files_complete = [os.path.join(input_dir, wav_file) for wav_file in wav_files]

#%%

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    try:
        wav, sample_rate = tf.audio.decode_wav(contents=file_contents, desired_channels=1)
    except Exception as e:
        print("File: ", filename)
        print("Error:\n", str(e))
        return None

    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(input=wav, rate_in=sample_rate, rate_out=16000)
    return np.array(wav)

def load_wav_16k_mono_v2(filename):
    wav, sample_rate = librosa.load(filename)
    wav = librosa.resample(y = wav, orig_sr = sample_rate, target_sr = 16000)
    wav = np.array(wav)
    return wav

#%%

keep_length = 12288
sr = 16000

for (i, wav_file_complete) in enumerate(wav_files_complete):
    wav = load_wav_16k_mono(wav_file_complete)
    if wav is None:
        continue
    wav_length = len(wav)
    padding = (wav_length - keep_length)
    start = randrange(0, padding)
    keep_wav = wav[int(start):int(start+keep_length)]
    out_filename = os.path.join(output_dir, wav_files[i])
    sf.write(out_filename, keep_wav, sr)

