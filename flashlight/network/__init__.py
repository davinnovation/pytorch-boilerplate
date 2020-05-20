import inspect

from ..utils import func

import torch
import torchvision


def check_network_option(network, network_option_dict):
    if len(network_option_dict.keys()) > 0:
        func.function_arg_checker(NETWORK_DICT[network], network_option_dict)
    return network_option_dict


def __get_squeezenet(num_classes):
    model = torchvision.models.squeezenet1_0(pretrained=False, progress=True, num_classes=num_classes)
    return model


def get_network(network_name, network_opt):
    return NETWORK_DICT[network_name](**network_opt)


NETWORK_DICT = {"squeezenet": __get_squeezenet}
