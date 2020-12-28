import os.path as osp

import torch
import torch.nn as nn
import pytorch_lightning as pl


class PLModule(pl.LightningModule):  # for classification
    def __init__(self, network, optimizer):
        super(PLModule, self).__init__()
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

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch[0])  # == self(batch[0])
        Y = batch[1]
        loss = self.loss(pred.float(), Y.long())

        self.log('train_loss', loss)

        return loss

    # def training_epoch_end(self, outputs):
    #     pass

    def validation_step(self, batch, batch_idx):  # optional
        pred = self.forward(batch[0])
        Y = batch[1]
        loss = self.loss(pred.float(), Y.long())

        self.log('val_loss', loss)

    # def validation_epoch_end(self, outputs):
    #     pass

    def test_step(self, batch, batch_nb):  # optional
        pred = self.forward(batch[0])
        Y = batch[1]
        loss = self.loss(pred.float(), Y.long())

        self.log('test_loss', loss)

    # def test_epoch_end(self, outputs):
    #     pass

    def configure_optimizers(self):  # require
        return self.optimizer
