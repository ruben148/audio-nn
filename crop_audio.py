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

#%%

input_dir = os.path.join(os.getcwd(), 'OneDrive', 'Desktop', 'Licenta', 'spyder', 'datasets', 'fsc22', 'audio')
output_dir = os.path.join(os.getcwd(), 'OneDrive', 'Desktop', 'Licenta', 'spyder', 'datasets', 'fsc22_processed')

#%%

wav_files = [filename for filename in os.listdir(input_dir) if filename.endswith('.wav')]

wav_files_complete = [os.path.join(input_dir, wav_file) for wav_file in wav_files]

#%%

def load_wav_16k_mono(filename):
    wav, sample_rate = librosa.load(filename)
    wav = librosa.resample(y = wav, orig_sr = sample_rate, target_sr = 16000)
    wav = np.array(wav)
    return wav

#%%

keep_length = 48000
sr = 16000

for (i, wav_file_complete) in enumerate(wav_files_complete):
    wav = load_wav_16k_mono(wav_file_complete)
    wav_length = len(wav)
    padding = (wav_length - keep_length)
    start = randrange(0, padding)
    keep_wav = wav[int(start):int(start+keep_length)]
    out_filename = os.path.join(output_dir, wav_files[i])
    sf.write(out_filename, keep_wav, sr)

