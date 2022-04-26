import torch
import torch.nn as nn
from tqdm import tqdm
from customTypes import TensorOrArray, MetricList, Dataloader
from typing import Dict


class Trainer:
    def __init__(
            self,
            model,
            metrics: MetricList,
            loss_func,
            optimizer,
            scheduler=None,
            main_metric_greater_is_better: bool = True,
            device=None
    ):
        """
        :param model: torch.nn.Module inherited class.
        :param metrics: List of metrics to be evaluated on validation set. First metric is considered the main metric.
        :param loss_func: Loss function for the model to be evaluated on. eg: Logloss, MSE.
        :param optimizer: Optimizer function. eg: Adam, SGD.
        :param scheduler: Scheduler to tune learning rate on the fly.
        :param main_metric_greater_is_better: Higher score is considered good by default, flip the bool in metrics where
        lower score is considered better. Eg: MSE.
        """
        self.model = model
        self.metrics = metrics
        self.main_metric = metrics[0][0]
        self.main_metric_parms = metrics[0][1]
        self.main_metric_greater_is_better = main_metric_greater_is_better
        self.best_score = None
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler

        # can't be tested since I don't have a stupid gpu
        if device is not None:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def score_prediction(y, pred, metrics):
        y = y.cpu()
        pred = pred.cpu()
        for metric, params in metrics:
            score = metric(y, pred, **params)
            print(f'\n{metric.__name__}: {score}')

    def fit(self, dataloaders: Dict[str, Dataloader], epochs: int = 10, save: bool = False) -> None:
        """
        Train the model and validate on validation data.
        :param dataloaders: Python dict containing train and validation dataloaders.
         Must contain 'train' and 'val' as keys.
        :param epochs: Number of epochs to train.
        :param save: Save model state dict if main metric score is better than in previous epoch.
        """
        self.model.train()
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
            print(f"\n running loss: {epoch_loss}")

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

            if save:
                current_score = self.main_metric(
                    batch_targets_val.cpu(),
                    batch_class_prediction_val.cpu(),
                    **self.main_metric_parms
                )

                if (self.main_metric_greater_is_better and current_score > self.best_score) or \
                        (self.main_metric_greater_is_better is False and current_score < self.best_score):
                    print("[New Best score, saving state dict]")
                    self.best_score = current_score
                    self.save_model(epoch)

    def save_model(self, epoch):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            f'best {self.main_metric.__name__}': self.best_score,
            'epoch': epoch
        }
        torch.save(checkpoint, 'bestModel.pkl.tar')

    def predict(self, images: TensorOrArray) -> TensorOrArray:
        """
        Predicts probability scores for inputs
        :param images: Accepts only 4d ndarray
        :return: nd-Tensor of transformed logits
        """
        self.model.eval()
        # todo implement sanity checks
        if images.ndim < 4:
            pass

        pred: torch.Tensor = self.model(images)

        if pred.size(1) < 2:
            activation_func = nn.Softmax(dim=1)
        else:
            activation_func = nn.Sigmoid()

        return activation_func(pred)
