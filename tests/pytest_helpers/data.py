import os
import torch
from torch.utils import data
import pytest
from skimage import io
import pandas as pd
from .nn import simple_augmentation
from utils.imageDataset import CustomDataset
from typing import Callable


@pytest.fixture()
def image():
    def get_image(image_path: str) -> Callable:
        filepath = os.path.join(os.path.dirname(__file__), image_path)
        _image = io.imread(filepath)
        _image = simple_augmentation(image=_image)['image']
        _image = torch.tensor(_image, dtype=torch.float)
        _image = _image.permute((2, 0, 1))
        _image = torch.unsqueeze(_image, 0)
        return _image

    return get_image


@pytest.fixture()
def invalidset_df():
    paths = os.path.join(os.path.dirname(__file__), '../sampleData/invalidset.csv')
    df = pd.read_csv(paths)
    return df


@pytest.fixture()
def dataloaders():
    filepath = os.path.join(os.path.dirname(__file__), '../sampleData/paths.csv')
    df = pd.read_csv(filepath)
    _dataloaders = {
        'train': data.DataLoader(CustomDataset(df.iloc[:4], simple_augmentation), batch_size=2),
        'val': data.DataLoader(CustomDataset(df.iloc[5:], simple_augmentation), batch_size=2)
    }
    return _dataloaders
