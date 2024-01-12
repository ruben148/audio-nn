import os
import shutil

# Directories
source_dir = "/home/buu3clj/radar_ws/datasets/train/13288/neg"
dest_dir = "/home/buu3clj/radar_ws/datasets/esc50"

# Ensure both directories exist
if not os.path.exists(source_dir):
    print(f"Source directory does not exist: {source_dir}")
    exit(1)

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)  # Create the destination directory if it does not exist

# Copy files
for filename in os.listdir(source_dir):
    # Check if the file name starts with a digit and contains an uppercase letter
    if filename[0].isdigit() and any(char.isupper() for char in filename):
        source_file_path = os.path.join(source_dir, filename)
        dest_file_path = os.path.join(dest_dir, filename)
        
        # Copy the file
        shutil.move(source_file_path, dest_file_path)

print("Files copied successfully.")