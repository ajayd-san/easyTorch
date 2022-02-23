import os

import pytest
import torch
from utils.imageDataset import CustomDataset
import pandas as pd
from tests.helpers import simple_augmentation


def test_datataset_len():
    filepath = os.path.join(os.path.dirname(__file__), '../sampleData/paths.csv')
    df = pd.read_csv(filepath)
    cd = CustomDataset(df)
    assert len(cd) == df.shape[0]


@pytest.mark.parametrize('augmentations', [None, simple_augmentation])
def test_dataset_getitem(augmentations):
    filepath = os.path.join(os.path.dirname(__file__), '../sampleData/paths.csv')
    paths_df = pd.read_csv(filepath)
    cd = CustomDataset(paths_df, augmentations)
    image, target = next(iter(cd))
    assert isinstance(image, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert image.dtype == torch.float
    assert target.dtype == torch.long

    if augmentations is not None:
        assert list(image.size()) == [3, 40, 40]

