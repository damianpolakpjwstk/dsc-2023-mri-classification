import os
from time import time

SOURCE_DATA_PATH = r"/home/dpolak/Downloads/IXI_dataset"
OUTPUT_DATA_PATH = r"/home/dpolak/Downloads/IXI_output"
FREESURFER_LICENCE_PATH = r"/home/dpolak/Downloads/freesurfer_license/"
ID_U = os.getuid()
ID_G = os.getgid()

BASE_COMMAND = rf"docker run --gpus all \
            -v {SOURCE_DATA_PATH}:/data \
            -v {OUTPUT_DATA_PATH}:/output \
            -v {FREESURFER_LICENCE_PATH}:/fs_license \
            --rm --user {ID_U}:{ID_G} \
            scans"

scans_to_process = os.listdir(SOURCE_DATA_PATH)

os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
for filename in scans_to_process:
    filename = filename.split('.')[0]
    if os.path.exists(os.path.join(OUTPUT_DATA_PATH, filename, "mri", "antsdn.brain_final.nii.gz")):
        print(f"Skipping {filename}")
        continue
    print(f"Processing {filename}")
    t0 = time()
    os.system(BASE_COMMAND + f" --input {filename}.nii.gz")
    print(f"Processed {filename} in {time() - t0} seconds.")
