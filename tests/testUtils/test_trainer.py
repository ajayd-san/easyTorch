from sklearn import metrics
from torch import nn, optim
# noinspection PyUnresolvedReferences
from tests.pytest_helpers.data import sample_model, dataloaders, image


def test_fit(sample_model, dataloaders):
    try:
        model = sample_model(
            nn.CrossEntropyLoss,
            optim.Adam,
            [(metrics.accuracy_score, {})]
        )
        model.fit(dataloaders)
    except:
        assert False

def test_prediction(sample_model, image):
    _image = image('../sampleData/images/cat1.jpeg')
    model = sample_model(nn.CrossEntropyLoss, optim.Adam, [(metrics.recall_score, {'average': 'macro'})])
    predictions = model.predict(_image)
    assert list(predictions.size()) == [1, 3]
