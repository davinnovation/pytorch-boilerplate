import os.path as osp

import torch
import torch.nn as nn
import pytorch_lightning as pl


class PL(pl.LightningModule):
    def __init__(self, network, dataloader, optimizer, train_log_interval):
        super(PL, self).__init__()
        self.network = network['network']
        self.hparams = dict(network['network_option'])
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.train_log_interval = train_log_interval
        self.get_curidx = lambda x: self.current_epoch * len(self.dataloader["train"]) + x
        self.train_avg_loss = 0.0
        self.train_avg_cnt = 0

        self.val_best_score = 0.0

        self.loss = nn.CrossEntropyLoss()  # TODO HardCoded

        # related to NNI
        self.final_target = 0

    def forward(self, x):
        pred = self.network(x)
        return pred

    def training_step(self, batch, batch_nb):
        pred = self.forward(batch[0]) # == self(batch[0])
        Y = batch[1]
        loss = self.loss(pred.float(), Y.long())
        self.train_avg_loss += loss.mean()
        self.train_avg_cnt += 1

        if self.get_curidx(batch_nb) % self.train_log_interval == 0:
            tensorboard_logs = {"avg_train_loss": self.train_avg_loss / self.train_avg_cnt}

            self.train_avg_loss = 0.0
            self.train_avg_cnt = 0

            return {
                "loss": loss,
                "progress_bar": {"train_loss": loss.item()},
                "log": tensorboard_logs,
            }

        return {"loss": loss, "progress_bar": {"train_loss": loss.item()}}

    def optimizer_step(
        self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
    ):
        optimizer.step()
        optimizer.zero_grad()
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()

    def validation_step(self, batch, batch_nb):  # optional
        pred = self.forward(batch[0])
        Y = batch[1]
        loss = self.loss(pred.float(), Y.long())
        
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": logs, "progress_bar": logs}

    def test_step(self, batch, batch_nb):  # optional
        pred = self.forward(batch[0])
        Y = batch[1]
        loss = self.loss(pred.float(), Y.long())

        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        self.final_target = float(avg_loss)
        return {"test_loss": avg_loss, "log": logs, "progress_bar": logs}

    def configure_optimizers(self):  # require
        return self.optimizer

    def train_dataloader(self):
        return self.dataloader["train"]

    def val_dataloader(self):
        return self.dataloader["val"]

    def test_dataloader(self):
        return self.dataloader["test"]
