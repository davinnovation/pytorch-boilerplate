import os.path as osp
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl

from torch import optim
import torchvision

from ..network import *
from ..dataloader import get_datalaoder

from .pl import PL

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.logging import TensorBoardLogger


class MainPL:
    def __init__(self, train, val, test, hw, network, data, opt, log, seed: int = None) -> None:
        if seed:
            self.fix_seed(seed)

        self.hw_args = self._hw_intp(hw)
        self.data_args = self._data_intp(data)

        self.train_args = self._train_intp(train, self.data_args["data"], self.hw_args["num_workers"])
        self.val_args = self._val_intp(val, self.data_args["data"], self.hw_args["num_workers"])
        self.test_args = self._test_intp(test, self.data_args["data"], self.hw_args["num_workers"])

        self.network_args = self._network_intp(network)
        self.opt_args = self._opt_intp(opt, self.network_args["network"])
        self.log_args = self._log_intp(log)

    def fix_seed(self, seed=42):
        seed_everything(seed)

    def _hw_intp(self, args):
        gpu_idx = [int(gpu) for gpu in args.gpu_idx.split(",") if gpu != ""]
        return {
            "gpu_idx": gpu_idx if len(gpu_idx) > 0 else None,
            "num_workers": args.num_workers,
            "gpu_on": True if len(gpu_idx) > 0 else False,
        }

    def _data_intp(self, args):
        project_name = args.project_name
        data = get_datalaoder

        return {"data": data, "ds_name": args.ds_name}

    def _train_intp(self, args, data, num_workers):
        return {
            "dataloader": torch.utils.data.DataLoader(
                data(data=self.data_args["ds_name"], split="train"), batch_size=args.batch_size, num_workers=num_workers,
                pin_memory=self.hw_args['gpu_on']
            ),
            "epoch": args.epoch,
        }

    def _val_intp(self, args, data, num_workers):
        return {
            "dataloader": torch.utils.data.DataLoader(
                data(data=self.data_args["ds_name"], split="val"), batch_size=args.batch_size, num_workers=num_workers,
                pin_memory=self.hw_args['gpu_on']
            )
        }

    def _test_intp(self, args, data, num_workers):
        return {
            "dataloader": torch.utils.data.DataLoader(
                data(data=self.data_args["ds_name"], split="test"), batch_size=args.batch_size, num_workers=num_workers,
                pin_memory=self.hw_args['gpu_on']
            )
        }

    def _network_intp(self, args):
        network = args.network
        del args["network"]
        checkpoint = args.checkpoint
        del args["checkpoint"]

        network_option = args
        network_option_dict = check_network_option(network, network_option)
        network = get_network(network, network_option_dict)
        assert network != None
        return {"network": network, "network_option": network_option_dict}

    def _opt_intp(self, args, network):
        from ..utils import func

        opt = args.opt
        del args["opt"]

        func.function_arg_checker(optim.__dict__[opt], args)

        opt = optim.__dict__[opt](network.parameters(), **args)

        return {"opt": opt}

    def _log_intp(self, args):
        run_only_test = False
        if "test_dir" in args.keys():
            run_only_test = True
        return {
            "project_name": args.project_name,
            "train_log_freq": args.train_log_freq,
            "val_log_freq_epoch": args.val_log_freq_epoch,
            "run_only_test": run_only_test,
        }

    def run(self, profile=True):
        network = self.network_args
        optimizer = self.opt_args["opt"]
        dataloader = {
            "train": self.train_args["dataloader"],
            "val": self.val_args["dataloader"],
            "test": self.test_args["dataloader"],
        }

        pl = PL(
            network=network,
            dataloader=dataloader,
            optimizer=optimizer,
            train_log_interval=self.log_args["train_log_freq"],
        )

        trainer = Trainer(
            logger=TensorBoardLogger(save_dir="./Logs", name=self.log_args["project_name"]),
            gpus=self.hw_args["gpu_idx"],
            check_val_every_n_epoch=self.log_args["val_log_freq_epoch"],
            max_epochs=self.train_args["epoch"],
            min_epochs=self.train_args["epoch"],
            log_save_interval=1,
            row_log_interval=1,
            profiler=profile,
        )

        trainer.fit(pl)

        trainer.test(pl)

        return pl.final_target
