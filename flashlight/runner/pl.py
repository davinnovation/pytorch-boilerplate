import os.path as osp

import torch
import torch.nn as nn
import pytorch_lightning as pl


class PL(pl.LightningModule):  # for classification
    def __init__(self, network, optimizer):
        super(PL, self).__init__()
        self.network = network["network"]
        self.hparams = dict(network["network_option"])
        self.optimizer = optimizer

        self.val_best_score = 0.0

        self.loss = nn.CrossEntropyLoss()  # TODO HardCoded

        # related to NNI
        self.final_target = 0

    def forward(self, x):
        pred = self.network(x)
        return pred

    def training_step(self, batch, batch_nb):
        pred = self.forward(batch[0])  # == self(batch[0])
        Y = batch[1]
        loss = self.loss(pred.float(), Y.long())

        return {"loss": loss, "progress_bar": {"train_loss": loss}}

    def training_epoch_end(self, outputs):
        train_avg_loss = 0
        for output in outputs:
            train_avg_loss += output["loss"]
        train_avg_loss /= len(outputs)

        return {"log": {"train_loss": train_avg_loss.item()}, "progress_bar": {"train_loss": train_avg_loss}}

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                    second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step()

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
