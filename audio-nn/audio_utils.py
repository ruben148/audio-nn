# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 20:04:40 2023

@author: Ruben
"""

import tensorflow as tf
import tensorflow_io as tfio
import librosa
import numpy as np

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(contents=file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(input=wav, rate_in=sample_rate, rate_out=16000)
    return np.array(wav)

def load_wav_16k_mono_v2(filename):
    wav, sample_rate = librosa.load(filename)
    wav = librosa.resample(y = wav, orig_sr = sample_rate, target_sr = 16000)
    wav = np.array(wav)
    return wav

def zero_pad_wav(wav, samples = 48128):
    wav = wav[:samples]
    zero_padding = np.zeros(samples-len(wav), dtype = np.float32)
    wav = np.concatenate((wav, zero_padding))
    return wav

def compute_stft(wav):
    # s = tf.signal.stft(wav, frame_length=896, frame_step=94, fft_length=512, pad_end=True)
    # s = tf.abs(s)
    # s = tf.transpose(s)
    s = librosa.stft(y=wav, n_fft=896, hop_length=94, pad_mode="constant")
    s = np.abs(s)
    s = s[:, :-1]
    return s

def compute_mfcc(wav):
    m = librosa.feature.mfcc(y=wav, n_fft=400, hop_length=188, n_mfcc=201)
    return m