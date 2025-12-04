import os
import shutil

# --- CONFIGURATION ---
# Where are the mixed images currently?
source_folder = 'mixed_images' 

# Where do you want them to go?
base_dir = 'data/training_set'
cats_dir = os.path.join(base_dir, 'cats')
dogs_dir = os.path.join(base_dir, 'dogs')

# 1. Create the directories if they don't exist
os.makedirs(cats_dir, exist_ok=True)
os.makedirs(dogs_dir, exist_ok=True)

print(f"Scanning images in '{source_folder}'...")

# 2. Loop through every file and move it
count_cats = 0
count_dogs = 0

files = os.listdir(source_folder)

for filename in files:
    # Full path to the current image
    src_path = os.path.join(source_folder, filename)
    
    # Check if it's a file (not a folder)
    if os.path.isfile(src_path):
        filename_lower = filename.lower()
        
        # LOGIC: If filename has "cat", move to cats folder
        if 'cat' in filename_lower:
            shutil.move(src_path, os.path.join(cats_dir, filename))
            count_cats += 1
            
        # LOGIC: If filename has "dog", move to dogs folder
        elif 'dog' in filename_lower:
            shutil.move(src_path, os.path.join(dogs_dir, filename))
            count_dogs += 1

print("--------------------------------")
print(f"Success! Moved {count_cats} images to {cats_dir}")
print(f"Success! Moved {count_dogs} images to {dogs_dir}")
print("You can now run train.py!")