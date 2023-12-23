from audio_nn import audio as au
import random
import soundfile as sf
from playsound import playsound
import os

random.seed(7654321)

input_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\49152\\neg")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\49152\\neg")

input_files = [filename for filename in os.listdir(input_dir) if filename.endswith('.wav')]

noise_types = ['white', 'pink', 'brown', 'gaussian']
for file in input_files:
    sound = au.load_wav_16k_mono_tf(os.path.join(input_dir, file))
    
    noise_type = random.choice(noise_types)

    sound = au.add_noise(sound, 0.04, 0.10, noise_type)
    filename = f"{os.path.join(output_dir, file[:-4])}_{noise_type}_noise.wav"
    if(os.path.exists(filename)):
        os.remove(filename)
    sf.write(filename, sound, 16000)

input_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\24576\\neg")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\24576\\neg")

input_files = [filename for filename in os.listdir(input_dir) if filename.endswith('.wav')]

noise_types = ['white', 'pink', 'brown', 'gaussian']
for file in input_files:
    sound = au.load_wav_16k_mono_tf(os.path.join(input_dir, file))
    
    noise_type = random.choice(noise_types)

    sound = au.add_noise(sound, 0.04, 0.10, noise_type)
    filename = f"{os.path.join(output_dir, file[:-4])}_{noise_type}_noise.wav"
    if(os.path.exists(filename)):
        os.remove(filename)
    sf.write(filename, sound, 16000)

input_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\12288\\neg")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\12288\\neg")

input_files = [filename for filename in os.listdir(input_dir) if filename.endswith('.wav')]

noise_types = ['white', 'pink', 'brown', 'gaussian']
for file in input_files:
    sound = au.load_wav_16k_mono_tf(os.path.join(input_dir, file))
    
    noise_type = random.choice(noise_types)

    sound = au.add_noise(sound, 0.04, 0.10, noise_type)
    filename = f"{os.path.join(output_dir, file[:-4])}_{noise_type}_noise.wav"
    if(os.path.exists(filename)):
        os.remove(filename)
    sf.write(filename, sound, 16000)