import os
import csv

csv_file_path = 'C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\esc50\\esc50.csv'

directory_path = 'C:\\Users\\Ruben\\Desktop\\Licenta\\datasets\\esc50\\esc50_49152'

chainsaw_files = []
with open(csv_file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['category'] == 'chainsaw':
            chainsaw_files.append(row['filename'])

for filename in chainsaw_files:
    file_path = os.path.join(directory_path, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print(f"File not found: {file_path}")