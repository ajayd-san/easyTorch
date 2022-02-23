import os
from customTypes import MetricList
import pandas as pd
import pytest
import torch
from sklearn import metrics
from utils.trainer import Trainer
from torch import nn, optim
from tests.helpers import simple_augmentation, Demo_model
from utils.imageDataset import CustomDataset
from torch.utils import data
from skimage import io


@pytest.fixture()
def image():
    filepath = os.path.join(os.path.dirname(__file__), '../sampleData/images/kitten1.jpeg')
    _image = io.imread(filepath)
    _image = simple_augmentation(image=_image)['image']
    _image = torch.tensor(_image, dtype=torch.float)
    _image = _image.permute((2, 0, 1))
    _image = torch.unsqueeze(_image, 0)
    return _image


@pytest.fixture()
def sample_model():
    def _get_model(loss_func, optimizer, metrics_: MetricList):
        model = Demo_model()
        loss_func = loss_func()
        optimizer = optimizer(model.parameters())
        model_trainer = Trainer(model, metrics_, loss_func, optimizer)

        return model_trainer

    return _get_model


@pytest.fixture()
def dataloaders():
    filepath = os.path.join(os.path.dirname(__file__), '../sampleData/paths.csv')
    df = pd.read_csv(filepath)
    _dataloaders = {
        'train': data.DataLoader(CustomDataset(df.iloc[:4], simple_augmentation), batch_size=2),
        'val': data.DataLoader(CustomDataset(df.iloc[5:], simple_augmentation), batch_size=2)
    }
    return _dataloaders


def test_fit(sample_model, dataloaders):
    model = sample_model(
        nn.CrossEntropyLoss,
        optim.Adam,
        [(metrics.recall_score, {'average': 'macro'}), (metrics.accuracy_score, {})]
    )
    model.fit(dataloaders)
    assert True


def test_prediction(sample_model, image):
    model = sample_model(nn.CrossEntropyLoss, optim.Adam, [(metrics.recall_score, {'average': 'macro'})])
    predictions = model.predict(image)
    assert list(predictions.size()) == [1, 3]
