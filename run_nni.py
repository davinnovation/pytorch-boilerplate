import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # NOT Safe

from omegaconf import OmegaConf
import nni

from flashlight.runner import main_pl

from config import config as dc


def search_params_intp(params):
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
    params = search_params_intp(params)
    cfg = OmegaConf.structured(cfg)
    args = OmegaConf.merge(cfg, params)
    print(args)
    ml = main_pl.MainPL(
        args.train, args.val, args.test, args.hw, args.network, args.data, args.opt, args.log, args.seed
    )
    final_result = ml.run()
    nni.report_final_result(final_result)


if __name__ == "__main__":
    _main()
