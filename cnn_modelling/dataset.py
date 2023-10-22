"""Dataloader for IXI dataset."""
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch.utils.data
import torchio


class IXIDataset(torch.utils.data.Dataset):
    """Loads IXI dataset."""
    def __init__(self, df: pd.DataFrame, scans_directory: str | Path, augment: bool = False):
        """
        Initialize dataloader.

        :param df: Dataframe with columns: "IXI_ID", "SEX_ID (1=m, 2=f)", "AGE".
        :param scans_directory: Path to the directory with scans.
        :param batch_size: Batch size. Default is 1.
        """

        self.scans_directory = Path(scans_directory)
        self.df = self.prepare_df(df)
        print(f"Loaded {len(self.df)} samples.")
        self.augment = augment

        self.basic_transform = torchio.Compose([
            torchio.CropOrPad(target_shape=(180, 180, 180), p=1.0),
            torchio.Resize(target_shape=(128, 128, 128)),
            ])

        self.augmentations = torchio.Compose([
            torchio.RandomFlip(axes=(0, 1, 2), p=0.5),
            torchio.RandomAffine(scales=(0.8, 1.2), degrees=5, translation=10, isotropic=True, p=0.5),
            torchio.RandomBlur(std=(0, 0.5), p=0.5),
            torchio.RandomSwap(patch_size=10, num_iterations=15, p=1.0),
        ])

        self.shuffle()

    def map_ids_to_scans_files(self):
        """
        Find scans files in the scans directory.

        :return: Dictionary mapping scan ID (IXI_ID) to the path to the scan.
        """

        scans = list(self.scans_directory.glob('*/mri/antsdn.brain_final.nii.gz'))
        return {int(str(path.parent).split(os.sep)[-2].split("-")[0].replace("IXI", "")): path
                for path in scans}

    def prepare_df(self, df):
        """
        Prepare dataframe.

        Add "path" column with paths to scans. One-hot encode ""SEX_ID (1=m, 2=f)"" column.
        :param df: Dataframe with columns: "IXI_ID", ""SEX_ID (1=m, 2=f)"", "AGE".
        :return: Modified dataframe.
        """
        df = df.dropna(subset=("IXI_ID", "SEX_ID (1=m, 2=f)", "AGE"))
        df["path"] = df["IXI_ID"].apply(lambda x: self.map_ids_to_scans_files().get(x, np.nan))
        df = df.dropna(subset=("path",))
        df["SEX_ID (1=m, 2=f)"].replace({1: "Male", 2: "Female"}, inplace=True)
        dummies = pd.get_dummies(df["SEX_ID (1=m, 2=f)"])
        df = pd.concat([df, dummies], axis=1)
        return df

    def __len__(self) -> int:
        """Get number of batches."""
        return len(self.df)

    def shuffle(self) -> None:
        """Shuffle dataframe."""
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Get single item from the dataset.

        :param idx: Index of the item.
        :return: image and tuple of ground truth tensors of age and sex for given image.
        """
        path = self.df.iloc[idx]["path"]
        image = nib.load(path).get_fdata()
        image = image.astype(np.float32)
        image /= 255.0
        image = np.expand_dims(image, axis=0)  #add channel dimension
        image = self.basic_transform(image)
        if self.augment:
            image = self.augmentations(image)
        image = torch.from_numpy(image).float().to("cuda")
        y_age = np.asarray(self.df.iloc[idx]["AGE"].astype(float))
        y_sex = self.df.iloc[idx][["Male", "Female"]].astype(float).values
        return image, (torch.from_numpy(y_age).float(), torch.from_numpy(y_sex).float())
