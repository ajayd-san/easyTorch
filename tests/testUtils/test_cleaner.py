from utils import cleaner
# noinspection PyUnresolvedReferences
from tests.pytest_helpers.data import invalidset_df


def test_datasetcleaner_fit(invalidset_df):
    dc = cleaner.DatasetCleaner(invalidset_df)
    dc.fit()
    invalid_paths = dc.transform()
    assert len(invalid_paths) == 2


def test_datasetcleaner_fit_transform(invalidset_df):
    dc = cleaner.DatasetCleaner(invalidset_df)
    invalild_paths = dc.fit_transform()
    assert len(invalild_paths) == 2
