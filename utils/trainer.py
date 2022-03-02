import torch
import torch.nn as nn
from tqdm import tqdm
from customTypes import TensorOrArray, MetricList


class Trainer:
    def __init__(self, model, metrics: MetricList, loss_func, optimizer, scheduler=None):
        self.model = model
        self.metrics = metrics
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def score_prediction(y, pred, metrics):
        y = y.cpu()
        pred = pred.cpu()
        for metric, params in metrics:
            score = metric(y, pred, **params)
            print(f'\n{metric.__name__}: {score}')

    def fit(self, dataloaders, epochs: int = 10) -> None:

        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}

        for epoch in range(epochs):
            self.model.train()
            running_loss, running_acc = 0, 0
            batch_targets_train, batch_class_prediction_train = torch.zeros(1), torch.zeros(1)
            batch_targets_val, batch_class_prediction_val = torch.zeros(1), torch.zeros(1)

            loop = tqdm(
                enumerate(dataloaders["train"], start=1),
                total=len(dataloaders["train"]),
            )

            for batch, (images, target) in loop:
                images, target = images.to(self.device), target.to(self.device)
                pred_proba = self.model(images)
                loss = self.loss_func(pred_proba, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.shape[0]
                pred_class = pred_proba.argmax(1)

                running_acc += (pred_class == target).sum().item()

                batch_class_prediction_train = torch.cat((batch_class_prediction_train, pred_class.cpu()), dim=0)
                batch_targets_train = torch.cat((batch_targets_train, target.cpu()), dim=0)

                loop.set_description(f"[Train]Epoch [{epoch}/{epochs}]")

            epoch_loss = running_loss / dataset_sizes["train"]
            epoch_acc = float(running_acc) / dataset_sizes["train"]
            print(f"\n running loss: {epoch_loss}, running acc: {epoch_acc}")

            if self.scheduler is not None:
                self.scheduler.step(epoch_loss)

            Trainer.score_prediction(batch_targets_train[1:], batch_class_prediction_train[1:], self.metrics)

            running_loss, running_acc = 0, 0
            self.model.eval()
            loop = tqdm(
                enumerate(dataloaders["val"], start=1),
                total=len(dataloaders["val"]),
            )

            with torch.no_grad():
                for batch, (images, target) in loop:
                    images, target = images.to(self.device), target.to(self.device)
                    pred_proba = self.model(images)
                    loss = self.loss_func(pred_proba, target)
                    running_loss += loss.item() * images.shape[0]
                    pred_class = pred_proba.argmax(1)

                    batch_class_prediction_val = torch.cat((batch_class_prediction_val, pred_class.cpu()), dim=0)
                    batch_targets_val = torch.cat((batch_targets_val, target.cpu()), dim=0)

                    loop.set_description(f"[VAL] Epoch [{epoch}/{epochs}]")
                print(f"\n running loss: {running_loss / dataset_sizes['val']}")

                Trainer.score_prediction(batch_targets_val[1:], batch_class_prediction_val[1:], self.metrics)

    def predict(self, images: TensorOrArray) -> TensorOrArray:
        """
        Predicts probability scores for inputs
        :param images: Accepts only 4d ndarray
        :return: nd-Tensor of transformed logits
        """
        # todo implement sanity checks
        if images.ndim < 4:
            pass

        pred: torch.Tensor = self.model(images)

        if pred.size(1) < 2:
            activation_func = nn.Softmax(dim=1)
        else:
            activation_func = nn.Sigmoid()

        return activation_func(pred)
