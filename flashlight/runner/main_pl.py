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
from ..dataloader import get_data, check_data_option

from .pl import PL

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


class MainPL:
    def __init__(self, hw, network, data, opt, log, seed: int = None) -> None:
        if seed: self.fix_seed(seed)

        self.hw_args = self._hw_intp(hw)
        self.data_args = self._data_intp(data)

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
        data_option = dict(args)
        del data_option["ds_name"]
        data_option = check_data_option(args.ds_name, data_option)
        data = get_data(args.ds_name, data_option)
        assert data != None
        
        return {"data": data}

    def _network_intp(self, args):
        network_option = dict(args)
        network = network_option["network"]
        checkpoint = network_option["checkpoint"]
        del network_option["network"]
        del network_option["checkpoint"]

        network_option_dict = check_network_option(network, network_option)
        network = get_network(network, network_option_dict)
        assert network != None
        return {"network": network, "network_option": network_option_dict}

    def _opt_intp(self, args, network):
        from ..utils import func
        args = dict(args)
        opt = args["opt"]
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
            "val_log_freq_epoch": args.val_log_freq_epoch,
            "run_only_test": run_only_test,
            "epoch": args.epoch
        }

    def run(self, profile=True):
        network = self.network_args
        optimizer = self.opt_args["opt"]

        pl = PL(network=network, optimizer=optimizer)

        trainer = Trainer(
            logger=TensorBoardLogger(save_dir="./Logs", name=self.log_args["project_name"]),
            gpus=self.hw_args["gpu_idx"],
            check_val_every_n_epoch=self.log_args["val_log_freq_epoch"],
            max_epochs=self.log_args["epoch"],
            min_epochs=self.log_args["epoch"],
            log_save_interval=1,
            row_log_interval=1,
            profiler=profile,
        )

        trainer.fit(pl, self.data_args["data"])

        trainer.test(pl, datamodule=self.data_args["data"])

        return pl.final_target
