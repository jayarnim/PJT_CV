from tqdm import tqdm
import copy
from IPython.display import clear_output
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAccuracy, 
    BinaryPrecision, 
    BinaryRecall,
    BinaryF1Score, 
    BinaryConfusionMatrix
)
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast


class TrainingLoop(nn.Module):
    def __init__(
        self, 
        model, 
        lr: float=1e-4, 
        lambda_: float=1e-2, 
        threshold: float=0.5,
    ):
        super(TrainingLoop, self).__init__()
        # device setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type=='cuda'

        # global attr
        self.model = model.to(self.device)
        self.lr = lr
        self.lambda_ = lambda_
        self.threshold = threshold

        # Loss FN
        self.criterion = nn.BCEWithLogitsLoss()

        # Metric
        self.metric = BinaryRecall(threshold=self.threshold)
        self.metric.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            params=model.parameters(), 
            lr=self.lr, 
            weight_decay=lambda_,
        )

        # gradient scaler setting
        self.scaler = GradScaler()

    def fit(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
        tst_loader: torch.utils.data.dataloader.DataLoader, 
        n_epochs: int, 
        patience: int=5,
        delta: float=1e-3,
    ):
        trn_loss_list = []
        val_loss_list = []
        val_metric_list = []

        counter = 0
        best_epoch = 0
        best_score = 0
        best_model_state = None

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print(f"EPOCH {epoch+1} START ---->>>>")

            # TRN
            trn_loss = self._trn_epoch(trn_loader, n_epochs, epoch)
            trn_loss_list.append(trn_loss)
            print(f"TRN LOSS: {trn_loss:.4f}")

            # VAL
            val_loss = self._val_epoch(val_loader, n_epochs, epoch)
            val_loss_list.append(val_loss)
            print(f"VAL LOSS: {val_loss:.4f}")

            # METRIC
            current_score = self._tst_epoch(tst_loader, n_epochs, epoch)
            val_metric_list.append(current_score)
            print(
                f"CURRENT SCORE: {current_score:.4f}",
                f"BEST SCORE: {best_score:.4f}({best_epoch})",
                sep='\t',
            )

            if current_score > best_score + delta:
                best_epoch = epoch + 1
                best_score = current_score
                best_model_state = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1
            
            if counter > patience:
                break
                
            # log reset
            if (epoch + 1) % 50 == 0:
                clear_output(wait=False)
        
        clear_output(wait=False)
        print(
            f"BEST EPOCH: {best_epoch}",
            f"BEST SCORE: {best_score:.4f}",
            sep="\n"
        )

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        history = dict(
            trn_loss=trn_loss_list,
            val_loss=val_loss_list,
            val_metric=val_metric_list,
        )

        return history

    def _trn_epoch(self, trn_loader, n_epochs, epoch):
        self.model.train()

        epoch_loss = 0.0

        iter_obj = tqdm(
            iterable=trn_loader, 
            desc=f"Epoch {epoch+1}/{n_epochs} TRN"
        )

        for videos, labels in iter_obj:
            # to gpu
            kwargs = dict(
                videos=[v.to(self.device) for v in videos],
                labels=labels.to(self.device),
            )
            # forward pass
            with autocast(self.use_amp):
                self.optimizer.zero_grad()
                batch_loss = self._batch(**kwargs)
            # accumulate loss
            epoch_loss += batch_loss.item()
            # backward pass
            self._run_opt(batch_loss)

        return epoch_loss / len(trn_loader)

    def _val_epoch(self, val_loader, n_epochs, epoch):
        self.model.eval()

        epoch_loss = 0.0

        iter_obj = tqdm(
            iterable=val_loader, 
            desc=f"Epoch {epoch+1}/{n_epochs} VAL"
        )

        with torch.no_grad():
            for videos, labels in iter_obj:
                # to gpu
                kwargs = dict(
                    videos=[v.to(self.device) for v in videos],
                    labels=labels.to(self.device),
                )
                # forward pass
                with autocast(self.use_amp):
                    batch_loss = self._batch(**kwargs)
                # accumulate loss
                epoch_loss += batch_loss.item()

        return epoch_loss / len(val_loader)


    def _tst_epoch(self, tst_loader, n_epochs, epoch):
        self.model.eval()
        
        self.metric.reset()

        iter_obj = tqdm(
            iterable=tst_loader, 
            desc=f"Epoch {epoch+1}/{n_epochs} TST"
        )

        for videos, labels in iter_obj:
            videos = [v.to(self.device) for v in videos]
            labels = labels.to(self.device)
            preds = self.model.predict(videos)
            self.metric.update(preds, labels)

        return self.metric.compute().item()

    def _batch(self, videos, labels):
        logits = self.model(videos)
        task_loss = self.criterion(logits, labels)
        return task_loss

    def _run_opt(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()



class PredictionLoop(nn.Module):
    def __init__(
        self, 
        model, 
        threshold: float=0.5,
    ):
        super(PredictionLoop, self).__init__()
        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.model = model.to(self.device)
        self.threshold = threshold

        # Metric
        self.accuracy = BinaryAccuracy(threshold=self.threshold)
        self.precision = BinaryPrecision(threshold=self.threshold)
        self.recall = BinaryRecall(threshold=self.threshold)
        self.f1 = BinaryF1Score(threshold=self.threshold)
        self.confmat = BinaryConfusionMatrix(threshold=self.threshold)

        self.accuracy.to(self.device)
        self.precision.to(self.device)
        self.recall.to(self.device)
        self.f1.to(self.device)
        self.confmat.to(self.device)


    def predict(self, tst_loader):
        self.model.eval()

        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.confmat.reset()

        iter_obj = tqdm(
            iterable=tst_loader, 
            desc=f"PREDICTION",
        )

        for videos, labels in iter_obj:
            videos = [v.to(self.device) for v in videos]
            labels = labels.to(self.device)
            preds = self.model.predict(videos)

            self.accuracy.update(preds, labels)
            self.precision.update(preds, labels)
            self.recall.update(preds, labels)
            self.f1.update(preds, labels)
            self.confmat.update(preds, labels)

        print(
            f"ACC:\t{self.accuracy.compute().item():.4f}",
            f"PRE:\t{self.precision.compute().item():.4f}",
            f"REC:\t{self.recall.compute().item():.4f}",
            f"F1:\t{self.f1.compute().item():.4f}",
            sep="\n",
        )

        self._vis_confmat()

    def _vis_confmat(self):
        scores = self.confmat.compute().cpu().numpy()

        plt.figure(figsize=(5, 4))
        
        sns.heatmap(
            data=scores, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Binary Confusion Matrix")
        
        plt.tight_layout()
        
        plt.show()
