import os
import shutil
import random
import math

src_folder = 'DataSetTypes'
dst_folder = 'DataSetTypesTesting'

if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

for subfolder in os.listdir(src_folder):
    src_subfolder = os.path.join(src_folder, subfolder)
    dst_subfolder = os.path.join(dst_folder, subfolder)

    if not os.path.isdir(src_subfolder):
        continue
    
    if not os.path.exists(dst_subfolder):
        os.makedirs(dst_subfolder)
    
    files = os.listdir(src_subfolder)
    
    num_files_to_move = int(math.ceil(len(files) * 0.2))
    
    files_to_move = random.sample(files, num_files_to_move)
    
    for file_name in files_to_move:
        src_file = os.path.join(src_subfolder, file_name)
        dst_file = os.path.join(dst_subfolder, file_name)
        shutil.move(src_file, dst_file)
