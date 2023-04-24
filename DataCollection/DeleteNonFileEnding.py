import os
from PIL import Image
root_dir = 'DataSetGeneration1'

failed = []

for subdir, dirs, files in os.walk(root_dir):
        for file in files:           
            print("Resizing " + file)
            if(not file[-4:] == '.jpg'):
                os.remove(os.path.join(subdir, file))
                failed.append(os.path.join(subdir, file))

print("Failed:")
print(failed)
