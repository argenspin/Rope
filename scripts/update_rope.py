
import requests
from tqdm import tqdm
from pathlib import Path

import subprocess
import sys

def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def get_filename_from_link(folder, file_link):
    return f"{folder}{file_link.split(r'/')[-1]}"

def download_file(folder, url):
    filename = get_filename_from_link(folder, url)

    if Path(filename).is_file():
        print(f"Skipping {filename} as it is already downloaded ")
    else:   
        print(f"{filename}")     
        response = requests.get(url, stream=True)

        # Sizes in bytes.
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(filename, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")

#Download Models        
model_links = open(r"scripts\model_links.txt").read().splitlines()
folder = 'models/'
for link in model_links:
    download_file(folder, link)

# Download Face Editor Models
model_links = open(r"scripts\liveportrait_model_links.txt").read().splitlines()
folder = 'models/liveportrait_onnx/'
for link in model_links:
    download_file(folder, link)


''' Additional Functions to execute when updating'''
install_requirements()
pass
