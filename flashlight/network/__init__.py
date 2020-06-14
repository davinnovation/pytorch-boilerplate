import inspect

from ..utils import func

import torch
import torchvision


def check_network_option(network, network_option_dict):
    if len(network_option_dict.keys()) > 0:
        func.function_arg_checker(NETWORK_DICT[network], network_option_dict)
    return network_option_dict


def _get_squeezenet(num_classes, version:str="1_0", pretrained=True, progress=True):
    VERSION = {
        "1_0" : torchvision.models.squeezenet1_0,
        "1_1" : torchvision.models.squeezenet1_1
    }

    return VERSION[version](pretrained=pretrained, progress=progress, num_classes=num_classes)

def get_network(network_name, network_opt):
    return NETWORK_DICT[network_name](**network_opt)


NETWORK_DICT = {
    "squeezenet": _get_squeezenet
}
