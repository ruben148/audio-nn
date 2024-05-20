import os
import shutil

# Directories
source_dir = "/home/buu3clj/radar_ws/datasets/fsc22"
dest_dir = "/home/buu3clj/radar_ws/datasets_2_0/chainsaw/train/7168/neg"

# Ensure both directories exist
if not os.path.exists(source_dir):
    print(f"Source directory does not exist: {source_dir}")
    exit(1)

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)  # Create the destination directory if it does not exist

# Copy files
for filename in os.listdir(source_dir):
    source_file_path = os.path.join(source_dir, filename)
    dest_file_path = os.path.join(dest_dir, filename)
    
    # Copy the file
    shutil.copy(source_file_path, dest_file_path)

print("Files copied successfully.")