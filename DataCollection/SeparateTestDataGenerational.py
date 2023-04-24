import os
import shutil
import random

src_folder = 'DataSetGenerational'
dst_folder = 'DataSetGenerationalTesting'

if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

for i in range(1, 10):
    src_subfolder = os.path.join(src_folder, str(i))
    dst_subfolder = os.path.join(dst_folder, str(i))

    if not os.path.exists(dst_subfolder):
        os.makedirs(dst_subfolder)

    files = os.listdir(src_subfolder)

    num_files_to_move = int(len(files) * 0.2)

    files_to_move = random.sample(files, num_files_to_move)

    for file_name in files_to_move:
        src_file = os.path.join(src_subfolder, file_name)
        dst_file = os.path.join(dst_subfolder, file_name)
        shutil.move(src_file, dst_file)
