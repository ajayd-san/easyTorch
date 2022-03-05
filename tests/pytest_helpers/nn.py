from torch import nn
import albumentations as A
import pytest
from customTypes import MetricList
from utils.trainer import Trainer

simple_augmentation = A.Compose([
    A.Resize(40, 40)
])


class DemoModel(nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2 * 2 * 3, 2)
        )

    def forward(self, images):
        logits = self.model(images)
        return logits


@pytest.fixture()
def sample_model():
    def _get_model(loss_func, optimizer, metrics_: MetricList):
        model = DemoModel()
        loss_func = loss_func()
        optimizer = optimizer(model.parameters())
        model_trainer = Trainer(model, metrics_, loss_func, optimizer)

        return model_trainer

    return _get_model
