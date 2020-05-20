from omegaconf import OmegaConf
import nni

from config import config as dc


def params_intp(params):
    ret = {}
    for param in params.keys():
        # param : "train.batch"
        spl = param.split(".")
        if len(spl) == 2:
            temp = {}
            temp[spl[1]] = params[param]
            ret[spl[0]] = temp
        elif len(spl) == 1:
            ret[spl[0]] = params[param]
        else:
            raise ValueError
    return ret


def _main(cfg=dc.DefaultConfig) -> None:
    params = nni.get_next_parameter()
    params = params_intp(params)
    cfg = OmegaConf.structured(cfg)
    cfg = OmegaConf.merge(cfg, params)
    nni.report_final_result(cfg.train.train_batch_size)


if __name__ == "__main__":
    _main()
