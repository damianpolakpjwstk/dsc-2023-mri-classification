import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    filename = parser.parse_args().input
    output_path = rf"/output/{filename.split('.')[0]}"
    reference_template_path = "MNI152_T1_1mm_brain.nii.gz"

    FS_LICENSE_SET = "FS_LICENSE=/fs_license/license.txt "
    SEG_COMMAND = r"./run_fastsurfer.sh --parallel --sd /output --fs_license /fs_license/license.txt --seg_only "

    APPLY_MASK = rf"mri_mask {output_path}/mri/orig_nu.mgz {output_path}/mri/mask.mgz {output_path}/mri/brainmask.mgz"
    APPLY_MRI_CONVERT = rf"mri_convert {output_path}/mri/brainmask.mgz {output_path}/mri/brainmask.nii.gz"
    APPLY_REGISTRATION = fr"flirt -in {output_path}/mri/brainmask.nii.gz -ref {reference_template_path}\
                            -out {output_path}/mri/brainmask_registered.nii.gz -dof 6"

    ANTS_DENOISE = rf"/opt/freesurfer/bin/AntsDenoiseImageFs -i\
     {output_path}/mri/brainmask_registered.nii.gz -o {output_path}/mri/antsdn.brain_final.nii.gz"

    os.system(SEG_COMMAND + f"--t1 /data/{filename} --sid {filename.split('.')[0]}")
    print("Segmentation done")
    os.system(FS_LICENSE_SET + APPLY_MASK)
    os.system(FS_LICENSE_SET + APPLY_MRI_CONVERT)
    os.system(FS_LICENSE_SET + APPLY_REGISTRATION)
    os.system(FS_LICENSE_SET + ANTS_DENOISE)
