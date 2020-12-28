from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class HWConfig:
    gpu_idx: str = "0"
    num_workers: int = 10


@dataclass
class NetworkConfig:  # flexible
    network: str = "squeezenet"
    checkpoint: str = ""
    num_classes: int = 11
    version: str = "1_0"


@dataclass
class DataConfig:
    ds_name: str = "MNIST"
    data_dir: str = "./"
    train_batchsize: int = 256


@dataclass
class OptConfig:  # flexible
    opt: str = "Adam"
    lr: float = 1e-3


@dataclass
class LogConfig:
    project_name: str = "with_aug"
    val_log_freq_epoch: int = 1
    epoch: int = 10

@dataclass
class DefaultConfig:
    hw: HWConfig = HWConfig()
    network: NetworkConfig = NetworkConfig()
    data: DataConfig = DataConfig()
    opt: OptConfig = OptConfig()
    log: LogConfig = LogConfig()
    seed: str = 42
