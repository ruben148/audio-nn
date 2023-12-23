import os
import fnmatch

def remove_noise_files(start_directory):
    for dirpath, dirnames, filenames in os.walk(start_directory):
        for filename in fnmatch.filter(filenames, '*noise*'):
            file_path = os.path.join(dirpath, filename)
            print(f"Removing: {file_path}")
            os.remove(file_path)
        for dirname in dirnames:
            remove_noise_files(os.path.join(dirpath, dirname))

start_directory = 'C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\combined'
remove_noise_files(start_directory)