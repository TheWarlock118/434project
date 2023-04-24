import os
from PIL import Image
root_dir = 'DataSetAllMons'

failed = []

for subdir, dirs, files in os.walk(root_dir):
        for file in files:           
            print("Resizing " + file) 
            try:
                image = Image.open(os.path.join(subdir, file))
                resized = image.resize((150, 150))
                resized.save(os.path.join(subdir, file))
            except:
                failed.append(file)

print("Failed:")
print(failed)
