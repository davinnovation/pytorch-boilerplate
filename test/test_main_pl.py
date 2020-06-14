import sys, os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + "/../")

import pytest

from omegaconf import OmegaConf

from config import config
from flashlight.runner import main_pl


def test_print():
    assert True


def test_mainpl():
    args = OmegaConf.structured(config.DefaultConfig)

    ml = main_pl.MainPL(args.train, args.val, args.test, args.hw, args.network, args.data, args.opt, args.log)
    assert ml.run_pretrain_routine() == True


if __name__ == "__main__":
    test_mainpl()
