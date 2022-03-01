import pandas as pd
from utils import cleaner
import os
import pytest


@pytest.fixture()
def invalidset_df():
    paths = os.path.join(os.path.dirname(__file__), '../sampleData/invalidset.csv')
    df = pd.read_csv(paths)
    return df


def test_datasetcleaner_fit(invalidset_df):
    dc = cleaner.DatasetCleaner(invalidset_df)
    dc.fit()
    invalid_paths = dc.transform()
    assert len(invalid_paths) == 2


def test_datasetcleaner_fit_transform(invalidset_df):
    dc = cleaner.DatasetCleaner(invalidset_df)
    invalild_paths = dc.fit_transform()
    assert len(invalild_paths) == 2
