import os
from PIL import Image

p = os.path.abspath('.')
threshold_path = os.path.join(p, 'Thresholded')
scaled_path = os.path.join(p, 'Downsampled')


##Read files in the specified directory
files = (file for file in os.listdir(threshold_path) 
         if os.path.isfile(os.path.join(threshold_path, file)))

for name in files:
    try:
        img = Image.open(threshold_path+'/'+name)
        img.thumbnail((500, 500))
        img.save(os.path.join(scaled_path, name))
        print(f"Processed {name}")
    except:
        print(f"something is wrong with {name}")
