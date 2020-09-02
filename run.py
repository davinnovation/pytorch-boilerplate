import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from omegaconf import OmegaConf

from config import config as dc

from flashlight.runner import main_pl


def _main(cfg=dc.DefaultConfig) -> None:
    args = OmegaConf.structured(cfg)
    args.merge_with_cli()

    ml = main_pl.MainPL(
        args.hw, args.network, args.data, args.opt, args.log, args.seed
    )
    ml.run()


if __name__ == "__main__":
    _main()
