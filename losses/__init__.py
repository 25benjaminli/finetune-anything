import torch.nn as nn
from .losses import CustormLoss
import monai

AVAI_LOSS = {'ce': nn.CrossEntropyLoss, 'multi_label_soft_margin': nn.MultiLabelSoftMarginLoss,
             'test_custom': CustormLoss, 'mse': nn.MSELoss,
             'DiceCE': monai.losses.DiceCELoss}


def get_losses(losses):
    print(AVAI_LOSS)
    loss_dict = {}
    for name in losses:
        print(name in AVAI_LOSS)
        # assert name in AVAI_LOSS, print('{name} is not supported, please implement it first.'.format(name=name))
        if losses[name].params is not None:
            loss_dict[name] = AVAI_LOSS[name](**losses[name].params)
        else:
            loss_dict[name] = AVAI_LOSS[name]()
    return loss_dict
