import numpy as np
import torch
from skimage import io, color
from customTypes import ImagePlusTarget
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data: pd.DataFrame, augmentations=None, exit_on_error=False, random_on_error: bool = True):
        """
        :param data: Pandas dataframe with paths as first column and target as second column
        :param augmentations: Image transformations
        :param exit_on_error: Stop execution once an exception rises. Cannot be used in conjunction with random_on_error
        :param random_on_error: Upon an exception while reading an image, pick a random image and process it instead.
        Cannot be used in conjuntion with exit_on_error.
        """

        if exit_on_error and random_on_error:
            raise ValueError("Only one of 'exit_on_error' and 'random_on_error' can be true")

        if not pd.api.types.is_numeric_dtype(data.iloc[:, 1]):
            raise ValueError(f"{data.columns[1]} must be of type `int`")

        if not pd.api.types.is_string_dtype(data.iloc[:, 0]):
            raise ValueError(f"{data.columns[0]} must be of type `string`")

        self.image_paths = data.iloc[:, 0].to_numpy()
        self.targets = data.iloc[:, 1].to_numpy()
        self.augmentations = augmentations
        self.exit_on_error = exit_on_error
        self.random_on_error = random_on_error

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        try:
            image, target = self.read_image_data(index)
        except IndexError:
            raise
        except Exception:
            if self.exit_on_error:
                raise
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
        if image.ndim == 2:
            image = color.gray2rgb(image)
        if image.shape[2] > 3:
            image = color.rgba2rgb(image)
        return image, target
