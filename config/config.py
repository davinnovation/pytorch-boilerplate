from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class TrainConfig:
    batch_size: int = 256
    epoch: int = 50


@dataclass
class ValConfig:
    batch_size: int = 256


@dataclass
class TestConfig:
    batch_size: int = 1


@dataclass
class HWConfig:
    gpu_idx: str = "0"
    num_workers: int = 10


@dataclass
class NetworkConfig:  # flexible
    network: str = "squeezenet"
    checkpoint: str = ""
    num_classes: int = 10
    version: str = "1_0"


@dataclass
class DataConfig:
    ds_name: str = "MNIST"
    project_name: str = "1st"


@dataclass
class OptConfig:  # flexible
    opt: str = "Adam"
    lr: float = 1e-3


@dataclass
class LogConfig:
    project_name: str = "with_aug"
    train_log_freq: int = 100
    val_log_freq_epoch: int = 1


@dataclass
class DefaultConfig:
    train: TrainConfig = TrainConfig()
    val: ValConfig = ValConfig()
    test: TestConfig = TestConfig()
    hw: HWConfig = HWConfig()
    network: NetworkConfig = NetworkConfig()
    data: DataConfig = DataConfig()
    opt: OptConfig = OptConfig()
    log: LogConfig = LogConfig()
    seed: str = 42
