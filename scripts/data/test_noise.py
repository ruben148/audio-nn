from audio_nn import audio as au
import random
import soundfile as sf
from playsound import playsound
import os

random.seed(7654321)

dir = 'C:\\Users\\Ruben\\Desktop\\Licenta'
input_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\49152\\pos")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\49152\\pos")

input_files = [filename for filename in os.listdir(input_dir) if filename.endswith('.wav')]

test_file = "PR_20161205_115013_1_0_7200.910905_mix_1.wav"
sound = au.load_wav_16k_mono_tf(os.path.join(input_dir, test_file))

sound_to_play = os.path.join(input_dir, test_file)
playsound(sound_to_play)

noise_types = ['white', 'pink', 'brown', 'gaussian']
noise_type = 'brown'

sound = au.add_noise(sound, 0.04, 0.10, noise_type)

filename = f"{os.path.join(dir, test_file[:-4])}_{noise_type}_noise.wav"
if(os.path.exists(filename)):
    os.remove(filename)
sf.write(filename, sound, 16000)
input("Press enter now!")
playsound(filename)