# DO NOT RUN THIS
# This is the code to unzip all of the image files into the GitHub repository. 
# All of the images have already been unzipped, so if you run it again, ALL OF THE IMAGES WILL BE UNZIPPED AGAIN!!

import zipfile
from PIL import Image

zip_file_path = 'flattened_images.zip'
extract_dir = "BeetleImages/"
     
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"'{zip_file_path}' contents extracted to '{extract_dir}' successfully.")