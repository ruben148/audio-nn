import os
import numpy as np
import random

input_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\chainsaw\\chainsaw_49152")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\chainsaw\\chainsaw_49152") #12288 24576 49152
pos_dir = os.path.join(output_dir, 'pos')
neg_dir = os.path.join(output_dir, 'neg')

wav_files = [filename for filename in os.listdir(neg_dir) if filename.endswith('.wav')]



random.seed(1234567)
random.shuffle(wav_files)

print(len(wav_files))

wav_files = wav_files[2000:]
print(len(wav_files))
for wav_file in wav_files:
    os.remove(os.path.join(neg_dir, wav_file))