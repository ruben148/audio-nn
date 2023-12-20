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

def zero_pad_wav(wav, samples = 24064):
    wav = wav[:samples]
    zero_padding = np.zeros(samples-len(wav), dtype = np.float32)
    wav = np.concatenate((wav, zero_padding))
    return wav

def compute_stft(wav, time_axis, k_axis, n_fft = 1024):
    # s = tf.signal.stft(wav, frame_length=896, frame_step=94, fft_length=512, pad_end=True)
    # s = tf.abs(s)
    # s = tf.transpose(s)
    hop_length = wav.size / time_axis
    assert int(hop_length) == hop_length
    hop_length = int(hop_length)
    n_fft = k_axis * 2
    n_fft = int(n_fft)
    s = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, pad_mode="constant")
    # print("STFT shape: ", s.shape)
    s = np.abs(s)
    s = s[1:, :-1]
    k, t = s.shape
    assert k == k_axis
    assert t == time_axis
    s = np.transpose(s)
    return s

def compute_mfcc(wav, time_axis, k_axis, n_fft = 1024):
    hop_length = wav.size / time_axis
    assert int(hop_length)==hop_length
    hop_length = int(hop_length)
    m = librosa.feature.mfcc(y=wav, n_fft=n_fft, hop_length=hop_length, n_mfcc=k_axis)
    m = m[:, :-1]
    # print("MFCC shape: ", m.shape)
    k, t = m.shape
    assert k == k_axis
    assert t == time_axis
    m = np.transpose(m)
    return m

def compute_chroma_stft(wav, time_axis, k_axis, n_fft = 1024):
    hop_length = wav.size / time_axis
    assert int(hop_length) == hop_length
    hop_length = int(hop_length)
    n_fft = k_axis * 2
    n_fft = int(n_fft)
    s = librosa.feature.chroma_stft(y=wav, sr=16000, n_fft=n_fft, hop_length=hop_length, pad_mode="constant", n_chroma=k_axis)
    s = np.abs(s)
    s = s[:, :-1]
    # print("Chroma STFT shape: ", s.shape)
    k, t = s.shape
    assert k == k_axis
    assert t == time_axis
    s = np.transpose(s)
    return s