import numpy as np
import torch
from skimage import io, color
from customTypes import ImagePlusTarget
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data: pd.DataFrame, augmentations=None, random_on_error: bool = True):
        self.image_paths = data.iloc[:, 0].to_numpy()
        self.targets = data.iloc[:, 1].to_numpy()
        self.augmentations = augmentations
        self.random_on_error = random_on_error

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        image, target = None, None
        try:
            image, target = self.read_image_data(index)
        except:
            print(f"Exception occurred while reading image, {index}")
            if self.random_on_error:
                print("Replacing with random image")
                random_index = np.random.randint(0, self.__len__())
                image, target = self.read_image_data(random_index)

            else:  # todo implement return logic when self.random_on_error is false
                return

        if self.augmentations is not None:
            aug_image = self.augmentations(image=image)
            image = aug_image["image"]

        image = np.transpose(image, (2, 0, 1))

        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(target, dtype=torch.long)
        )

    def read_image_data(self, index: int) -> ImagePlusTarget:
        target = self.targets[index]
        image = io.imread(self.image_paths[index])
        if image.shape[2] > 3:
            image = color.rgba2rgb(image)

        return image, target
