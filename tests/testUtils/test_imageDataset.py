import os

import pytest
import torch
from utils.imageDataset import CustomDataset
import pandas as pd
from tests.helpers import simple_augmentation
from typing import Callable


@pytest.fixture()
def dataframe() -> Callable:
    def get_dataframe(rel_path: str) -> pd.DataFrame:
        filepath = os.path.join(os.path.dirname(__file__), rel_path)
        df = pd.read_csv(filepath)
        return df
    return get_dataframe


def test_random_on_error_and_exit_on_error_same_value(dataframe):
    df = dataframe('../sampleData/paths.csv')
    with pytest.raises(ValueError):
        CustomDataset(df, exit_on_error=True, random_on_error=True)


def test_exit_on_error_raises_exception(dataframe):
    df = dataframe('../sampleData/invalidset.csv')
    cd = CustomDataset(df, exit_on_error=True, random_on_error=False)
    with pytest.raises(Exception):
        for _, _ in cd:
            pass


def test_datataset_len(dataframe):
    df = dataframe('../sampleData/paths.csv')
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
