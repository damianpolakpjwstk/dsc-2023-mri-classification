"""Dataloader for IXI dataset."""
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch.utils.data
import torchio


class Mri3DDataLoader(torch.utils.data.Dataset):
    """Loads IXI dataset."""
    def __init__(self, df: pd.DataFrame, scans_directory: str | Path, batch_size: int = 1, augment: bool = False):
        """
        Initialize dataloader.

        :param df: Dataframe with columns: "IXI_ID", "SEX_ID (1=m, 2=f)", "AGE".
        :param scans_directory: Path to the directory with scans.
        :param batch_size: Batch size. Default is 1.
        """

        self.scans_directory = Path(scans_directory)
        self.df = self.prepare_df(df)
        print(f"Loaded {len(self.df)} samples.")
        self.batch_size = batch_size
        self.augment = augment

        self.basic_transform = torchio.Compose([
            torchio.ToCanonical(p=1.0),
            torchio.CropOrPad(target_shape=(180, 180, 180), p=1.0),
            torchio.Resize(target_shape=(128, 128, 128)),
            ])

        self.augmentations = torchio.Compose([
            #torchio.RandomFlip(axes=(0, 1, 2), p=0.5),
            #torchio.RandomAffine(scales=(1.0, 1.0), degrees=5, translation=10, isotropic=True, p=0.5),
            # torchio.RandomNoise(std=(0, 0.1), p=0.5),
            #torchio.RandomBlur(std=(0, 0.5), p=0.5),
            # torchio.RandomMotion(p=0.3),
            # torchio.RandomSpike(p=0.3),
            # torchio.RandomGhosting(p=0.3),
            #torchio.RandomSwap(patch_size=10, num_iterations=15, p=1.0),
            #torchio.Clamp(0, 1)
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
        return len(self.df) // self.batch_size

    def shuffle(self) -> None:
        """Shuffle dataframe."""
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def get_single_item(self, idx: int) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Get single item from the dataset.

        :param idx: Index of the item.
        :return: Tuple of tensors: image and label.
        """
        path = self.df.iloc[idx]["path"]
        image = nib.load(path).get_fdata()
        image = image.astype(np.float32)
        image /= 255.0
        image = np.expand_dims(image, axis=0)  #add channel dimension
        image = self.basic_transform(image)
        if self.augment:
            image = self.augmentations(image)
        image = np.expand_dims(image, axis=0)  # add batch dimension
        image = torch.from_numpy(image)
        y_age = self.df.iloc[idx]["AGE"]
        y_sex = self.df.iloc[idx][["Male", "Female"]].astype(float).values
        return image, (y_age, y_sex)

    # Use torch.DataLoader instead
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Get batch of images and labels. Returns tuple of (images, one hot encoded labels)."""
        images = []
        ys_age = []
        ys_sex = []
        for i in range(self.batch_size):
            item_idx = idx * self.batch_size + i
            image, (y_age, y_sex) = self.get_single_item(item_idx)
            images.append(image)
            ys_age.append(y_age)
            ys_sex.append(y_sex)
        return torch.cat(images, dim=0).float().cuda(), (torch.from_numpy(np.array(ys_age)).float(),
                                                         torch.from_numpy(np.array(ys_sex)).float())
