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

datasets_dir = '/home/buu3clj/radar_ws/datasets'

move_random_files(os.path.join(datasets_dir, "train", "13288", "neg"), 
                    os.path.join(datasets_dir, "validation", "13288", "neg"))
move_random_files(os.path.join(datasets_dir, "train", "13288", "pos"), 
                    os.path.join(datasets_dir, "validation", "13288", "pos"))