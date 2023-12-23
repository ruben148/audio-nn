import textgrid
import os
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_io as tfio

#%%

input_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\chainsaw\\chainsaw_original")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\chainsaw\\chainsaw_24576")

pos_dir = os.path.join(output_dir, 'pos')
neg_dir = os.path.join(output_dir, 'neg')

#%%

wav_files = [filename for filename in os.listdir(input_dir) if filename.endswith('.wav')]
textgrid_files = [filename for filename in os.listdir(input_dir) if filename.endswith('.TextGrid')]

wav_files.sort()
textgrid_files.sort()

wav_files_complete = [os.path.join(input_dir, wav_file) for wav_file in wav_files]
textgrid_files_complete = [os.path.join(input_dir, textgrid_file) for textgrid_file in textgrid_files]

#%%

print(textgrid_files_complete)

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

tg = textgrid.TextGrid.fromFile(textgrid_files_complete[3])
print(textgrid_files_complete[3])
print(tg[0][0])
print(tg[0][0].minTime)
print(tg[0][0].maxTime)
print(tg[0][0].mark)

#%%

avg_duration = 0
min_duration = 100000
max_duration = 0
total = 0

for textgrid_file in textgrid_files_complete:
    tg = textgrid.TextGrid.fromFile(textgrid_file)
    
    for interval in tg[0].intervals:
        if(interval.mark == "saw"):
            duration = interval.maxTime - interval.minTime
            avg_duration += duration
            total += 1
            if(duration > max_duration):
                max_duration = duration
            if(duration < min_duration):
                min_duration = duration

avg_duration /= total
print("Total intervals: ", total)
print("Average length: ", avg_duration)
print("Minimum length", min_duration)
print("Maximum length", max_duration)
#%%

block_length_seconds = 1.536 # 3.072 1.536 0.768
sr = 16000
buffer_size = sr * block_length_seconds
total = 0

for i in range(len(wav_files_complete)):
    wav = load_wav_16k_mono(wav_files_complete[i])
    tg = textgrid.TextGrid.fromFile(textgrid_files_complete[i])
    counter_interval = 0
    for interval in tg[0].intervals:
        if(interval.mark == "saw"):
            duration = interval.maxTime - interval.minTime
            padding = (duration % block_length_seconds) / 2
            remainder = abs(duration % block_length_seconds - block_length_seconds) / 2
            
            if(padding >= block_length_seconds/1000):
                start = interval.minTime - remainder
            else:
                start = interval.minTime
            
            # if(duration < block_length_seconds)
            
            if(start < 0):
                start = 0
            counter = 0
            
            while(start + block_length_seconds <= interval.maxTime + remainder + 0.1):
                start_sample = (int)(start * sr)
                end_sample = (int)(start_sample + block_length_seconds * sr)
                assert end_sample < len(wav)
                block = wav[start_sample : end_sample]
                out_filename = os.path.join(pos_dir, wav_files[i][:-4] + "_" + str(counter_interval) + "_" + str(counter) + "_" + str(start) + ".wav")
                sf.write(out_filename, block, sr)
                start += block_length_seconds
                counter += 1
                total += 1
        counter_interval += 1
        
print("Total intervals: ", total)

#%%

for i in range(len(wav_files_complete)):
    wav = load_wav_16k_mono(wav_files_complete[i])
    tg = textgrid.TextGrid.fromFile(textgrid_files_complete[i])
    counter_interval = 0
    for interval in tg[0].intervals:
        if(interval.mark == ""):
            duration = interval.maxTime - interval.minTime
            padding = (duration % block_length_seconds) / 2
            
            start = interval.minTime + padding
            
            if(start < 0):
                start = 0
            counter = 0
            
            while(start + block_length_seconds <= interval.maxTime - 0.3):
                start_sample = (int)(start * sr)
                end_sample = (int)(start_sample + block_length_seconds * sr)
                if end_sample > len(wav):
                    break
                block = wav[start_sample : end_sample]
                out_filename = os.path.join(neg_dir, wav_files[i][:-4] + "_" + str(counter_interval) + "_" + str(counter) + "_" + str(start) + ".wav")
                sf.write(out_filename, block, sr)
                start += block_length_seconds
                counter += 1
        counter_interval += 1





