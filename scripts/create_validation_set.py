import os
import shutil
import numpy as np

np.random.seed(54321)

def move_random_files(src_folder, dest_folder, proportion=0.2):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    files = os.listdir(src_folder)
    np.random.shuffle(files)
    num_files_to_move = int(proportion * len(files))

    for file in files[:num_files_to_move]:
        shutil.move(os.path.join(src_folder, file), os.path.join(dest_folder, file))

train_dir = '/home/buu3clj/radar_ws/datasets/train'
validation_dir = '/home/buu3clj/radar_ws/datasets/validation'

sub_dirs = ['12288', '24576', '49152']

for sub_dir in sub_dirs:
    move_random_files(os.path.join(train_dir, sub_dir, "neg"), 
                      os.path.join(validation_dir, sub_dir, "neg"))
    move_random_files(os.path.join(train_dir, sub_dir, "pos"), 
                      os.path.join(validation_dir, sub_dir, "pos"))