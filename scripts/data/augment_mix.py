from audio_nn import *
import random
import soundfile as sf

random.seed(7654321)

input_dir1 = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\chainsaw\\chainsaw_49152\\pos")
input_dir2 = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\esc50\\esc50_49152\\neg")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\49152\\pos")

input_files1 = [filename for filename in os.listdir(input_dir1) if filename.endswith('.wav')]
input_files2 = [filename for filename in os.listdir(input_dir2) if filename.endswith('.wav')]

counter = 0

for file1 in input_files1:
    sound1 = load_wav_16k_mono_tf(os.path.join(input_dir1, file1))
    index = random.randint(0, len(input_files2)-1)
    sound2 = load_wav_16k_mono_tf(os.path.join(input_dir2, input_files2[index]))
    ratio = random.uniform(0.5, 0.95)
    mixed = mix_advanced(sound1, sound2, ratio)
    filename = f"{os.path.join(output_dir, file1[:-4])}_mix_2.wav"
    if(os.path.exists(filename)):
        os.remove(filename)
    sf.write(filename, mixed, 16000)
    print(counter)
    counter += 1

print("Done this part.")

input_dir1 = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\chainsaw\\chainsaw_49152\\pos")
input_dir2 = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\fsc22\\fsc22_49152\\neg")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\49152\\pos")

input_files1 = [filename for filename in os.listdir(input_dir1) if filename.endswith('.wav')]
input_files2 = [filename for filename in os.listdir(input_dir2) if filename.endswith('.wav')]

counter = 0

for file1 in input_files1:
    sound1 = load_wav_16k_mono_tf(os.path.join(input_dir1, file1))
    index = random.randint(0, len(input_files2)-1)
    sound2 = load_wav_16k_mono_tf(os.path.join(input_dir2, input_files2[index]))
    ratio = random.uniform(0.5, 0.95)
    mixed = mix_advanced(sound1, sound2, ratio)
    filename = f"{os.path.join(output_dir, file1[:-4])}_mix_1.wav"
    if(os.path.exists(filename)):
        os.remove(filename)
    sf.write(filename, mixed, 16000)
    print(counter)
    counter += 1






input_dir1 = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\chainsaw\\chainsaw_24576\\pos")
input_dir2 = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\esc50\\esc50_24576\\neg")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\24576\\pos")

input_files1 = [filename for filename in os.listdir(input_dir1) if filename.endswith('.wav')]
input_files2 = [filename for filename in os.listdir(input_dir2) if filename.endswith('.wav')]

counter = 0

for file1 in input_files1:
    sound1 = load_wav_16k_mono_tf(os.path.join(input_dir1, file1))
    index = random.randint(0, len(input_files2)-1)
    sound2 = load_wav_16k_mono_tf(os.path.join(input_dir2, input_files2[index]))
    ratio = random.uniform(0.5, 0.95)
    mixed = mix_advanced(sound1, sound2, ratio)
    filename = f"{os.path.join(output_dir, file1[:-4])}_mix_2.wav"
    if(os.path.exists(filename)):
        os.remove(filename)
    sf.write(filename, mixed, 16000)
    print(counter)
    counter += 1

print("Done this part.")

input_dir1 = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\chainsaw\\chainsaw_24576\\pos")
input_dir2 = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\fsc22\\fsc22_24576\\neg")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\24576\\pos")

input_files1 = [filename for filename in os.listdir(input_dir1) if filename.endswith('.wav')]
input_files2 = [filename for filename in os.listdir(input_dir2) if filename.endswith('.wav')]

counter = 0

for file1 in input_files1:
    sound1 = load_wav_16k_mono_tf(os.path.join(input_dir1, file1))
    index = random.randint(0, len(input_files2)-1)
    sound2 = load_wav_16k_mono_tf(os.path.join(input_dir2, input_files2[index]))
    ratio = random.uniform(0.5, 0.95)
    mixed = mix_advanced(sound1, sound2, ratio)
    filename = f"{os.path.join(output_dir, file1[:-4])}_mix_1.wav"
    if(os.path.exists(filename)):
        os.remove(filename)
    sf.write(filename, mixed, 16000)
    print(counter)
    counter += 1









input_dir1 = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\chainsaw\\chainsaw_12288\\pos")
input_dir2 = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\esc50\\esc50_12288\\neg")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\12288\\pos")

input_files1 = [filename for filename in os.listdir(input_dir1) if filename.endswith('.wav')]
input_files2 = [filename for filename in os.listdir(input_dir2) if filename.endswith('.wav')]

counter = 0

for file1 in input_files1:
    sound1 = load_wav_16k_mono_tf(os.path.join(input_dir1, file1))
    index = random.randint(0, len(input_files2)-1)
    sound2 = load_wav_16k_mono_tf(os.path.join(input_dir2, input_files2[index]))
    ratio = random.uniform(0.5, 0.95)
    mixed = mix_advanced(sound1, sound2, ratio)
    filename = f"{os.path.join(output_dir, file1[:-4])}_mix_2.wav"
    if(os.path.exists(filename)):
        os.remove(filename)
    sf.write(filename, mixed, 16000)
    print(counter)
    counter += 1

print("Done this part.")

input_dir1 = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\chainsaw\\chainsaw_12288\\pos")
input_dir2 = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\fsc22\\fsc22_12288\\neg")
output_dir = os.path.join("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined\\12288\\pos")

input_files1 = [filename for filename in os.listdir(input_dir1) if filename.endswith('.wav')]
input_files2 = [filename for filename in os.listdir(input_dir2) if filename.endswith('.wav')]

counter = 0

for file1 in input_files1:
    sound1 = load_wav_16k_mono_tf(os.path.join(input_dir1, file1))
    index = random.randint(0, len(input_files2)-1)
    sound2 = load_wav_16k_mono_tf(os.path.join(input_dir2, input_files2[index]))
    ratio = random.uniform(0.5, 0.95)
    mixed = mix_advanced(sound1, sound2, ratio)
    filename = f"{os.path.join(output_dir, file1[:-4])}_mix_1.wav"
    if(os.path.exists(filename)):
        os.remove(filename)
    sf.write(filename, mixed, 16000)
    print(counter)
    counter += 1


