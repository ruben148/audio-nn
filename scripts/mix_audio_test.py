import audio_nn as ann
import librosa
import soundfile as sf

audio1 = ann.load_wav_16k_mono_tf("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\chainsaw\\chainsaw_49152\\pos\\RP6_9h30_to_9h40_195_2_528.45558.wav")
audio2 = ann.load_wav_16k_mono_tf("C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\fsc22\\fsc22_49152\\neg\\1_10106.wav")

print("Min: ", min(audio2))
print("Max: ", max(audio2))

input("Press enter now!")

mixed_audio = ann.mix_advanced(audio1, audio2, 0.1)

# librosa.output.write_wav('C:\\Users\\Ruben\\Desktop\\Licenta\\mixed_output.wav', mixed_audio, 16000)
sf.write('C:\\Users\\Ruben\\Desktop\\Licenta\\mixed_output_advanced2.wav', mixed_audio, 16000)