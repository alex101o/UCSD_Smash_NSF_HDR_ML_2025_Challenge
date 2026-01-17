import zipfile
from PIL import Image

zip_file_path = 'flattened_images.zip'
extract_dir = "BeetleImages/"
     
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"'{zip_file_path}' contents extracted to '{extract_dir}' successfully.")
