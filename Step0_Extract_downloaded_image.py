import zipfile
import os
from pathlib import Path

def extract_zip_files(directory):
    p = Path(directory)
    for f in p.glob('*.zip'):
        with zipfile.ZipFile(f, 'r') as archive:
            if not(os.path.isdir('sky_images_unzip/%s' % f.stem)):
                archive.extractall(path='sky_images_unzip/%s' % f.stem)
                print(f"Extracted contents from '{f.name}' to '{f.stem}' directory.")


if __name__ == "__main__":
    # Usage example
    os.makedirs("sky_images_unzip", exist_ok=True)
    extract_zip_files('sky_images')