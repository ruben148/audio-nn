# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 20:04:40 2023

@author: Ruben
"""

import tensorflow as tf
import tensorflow_io as tfio
import librosa
import numpy as np
import random
import math

def load_wav_16k_mono_tf(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(contents=file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(input=wav, rate_in=sample_rate, rate_out=16000)
    return np.array(wav)

def load_wav_16k_mono(filename):
    wav, sample_rate = librosa.load(filename)
    wav = librosa.resample(y = wav, orig_sr = sample_rate, target_sr = 16000)
    wav = np.array(wav)
    return wav

def load_wav_8k_mono_tf(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(contents=file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(input=wav, rate_in=sample_rate, rate_out=8000)
    return np.array(wav)

def load_wav_8k_mono(filename):
    wav, sample_rate = librosa.load(filename)
    wav = librosa.resample(y = wav, orig_sr = sample_rate, target_sr = 8000)
    wav = np.array(wav)
    return wav

def zero_pad_wav(wav, samples = 24576):
    wav = wav[:samples]
    zero_padding = np.zeros(samples-len(wav), dtype = np.float32)
    wav = np.concatenate((wav, zero_padding))
    return wav

def compute_stft(wav, time_axis, k_axis, n_fft = 512):
    # s = tf.signal.stft(wav, frame_length=896, frame_step=94, fft_length=512, pad_end=True)
    # s = tf.abs(s)
    # s = tf.transpose(s)
    
    n_fft = k_axis * 2
    n_fft = int(n_fft)

    # print("n fft",n_fft)

    # hop_length = (wav.size - n_fft) / (time_axis - 1)
    hop_length = math.floor(wav.size/ time_axis + 1)

    # print("hop length",hop_length)

    # hop_length = int(np.ceil(hop_length))
    assert int(hop_length) == hop_length
    hop_length = int(hop_length)
    # print(hop_length)

    s = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, pad_mode="constant")
    s = np.abs(s)
    # print(s.shape)
    s = s[1:, :]
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
    s = librosa.feature.chroma_stft(y=wav, sr=8000, n_fft=n_fft, hop_length=hop_length, pad_mode="constant", n_chroma=k_axis)
    s = np.abs(s)
    s = s[:, :-1]
    # print("Chroma STFT shape: ", s.shape)
    k, t = s.shape
    assert k == k_axis
    assert t == time_axis
    s = np.transpose(s)
    return s

def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight

def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 8000:
        n_fft = 1024
    elif fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db

def change_pitch(sound, max_variation, fs = 8000):
    semitones = 12 * np.log2(1 + random.uniform(-max_variation, max_variation))
        
    return librosa.effects.pitch_shift(sound, sr=fs, n_steps=semitones, bins_per_octave=12, res_type='soxr_hq', scale=True)

def mix_advanced(sound1, sound2, r, fs=8000):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    return sound

def mix(y1, y2, rate=0.5):
    y1_scaled = rate * y1
    y2_scaled = (1-rate) * y2

    if len(y1_scaled) > len(y2_scaled):
        y2_scaled = np.pad(y2_scaled, (0, len(y1_scaled) - len(y2_scaled)), 'constant')
    elif len(y2_scaled) > len(y1_scaled):
        y1_scaled = np.pad(y1_scaled, (0, len(y2_scaled) - len(y1_scaled)), 'constant')

    mixed_signal = y1_scaled + y2_scaled

    mixed_signal = mixed_signal / np.max(np.abs(mixed_signal))

    return mixed_signal

def random_gain(sound, db):
    return sound * np.power(10, random.uniform(-db, db) / 20.0)

def random_crop(sound, final_length):
    cut_length = len(sound) - final_length
    cut_start = random.randrange(0, cut_length+1)
    return sound[cut_start:cut_start+final_length]

def generate_white_noise(length, a=0, b=0.5):
    return np.random.normal(a, b, length)

def generate_pink_noise(length, a=0, b=0.5):
    white_noise = np.random.normal(a, b, length)
    pink_noise = np.cumsum(white_noise)
    pink_noise -= np.mean(pink_noise)
    return pink_noise

def generate_brown_noise(length, a=0, b=0.5):
    brown_noise = np.cumsum(np.random.normal(a, b, length))
    brown_noise -= np.mean(brown_noise)
    return brown_noise

def generate_gaussian_noise(length, mean=0, std=1):
    return np.random.normal(mean, std, length)

def normalize_loudness(noise, target_rms=0.1):
    current_rms = np.sqrt(np.mean(noise**2))
    return noise * (target_rms / current_rms)

def add_noise(sound, min_noise_ratio=0.05, max_noise_ratio=0.15, noise_type = 'white'):
    length = len(sound)

    if noise_type == 'white':
        noise = generate_white_noise(length, b=1)
    elif noise_type == 'pink':
        noise = generate_pink_noise(length, b=1)
    elif noise_type == 'brown':
        noise = generate_brown_noise(length, b=1)
    elif noise_type == 'gaussian':
        noise = generate_gaussian_noise(length)

    noise = noise / np.max(np.abs(noise))

    noise_ratio = random.uniform(min_noise_ratio, max_noise_ratio)

    noised_sound = mix_advanced(sound, noise, 1-noise_ratio)
    
    return noised_sound