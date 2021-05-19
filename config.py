""" Config class """
import argparse
import os
from functools import partial
import torch


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class TrainConfig(BaseConfig):
    @property
    def build_parser(self):
        parser = get_parser("Train config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--pretrain_model', type=str, default='train_exp/s1/checkpoint_90.pth.tar', help='pretrained model path')
        parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
        parser.add_argument('--resume', action='store_true', default=False, help='resume training')

        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--lr', type=float, default=0.01, help='lr for weights')
        parser.add_argument('--lr_min', type=float, default=0.0001, help='minimum lr for weights')
        parser.add_argument('--epochs', type=int, default=150, help='# of training epochs')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping for weights')

        parser.add_argument('--lw1', type=float, default=0.1, help='auxiliary loss weight')
        parser.add_argument('--lw2', type=float, default=0.01, help='auxiliary loss weight')
        parser.add_argument('--lw3', type=float, default=0.01, help='auxiliary loss weight')
        parser.add_argument('--lw4', type=float, default=1, help='auxiliary loss weight')
        parser.add_argument('--dropout', type=float, default=0, help='dropout prob')

        parser.add_argument('--arc', action='store_true', default=False, help='use arcmargin')
        parser.add_argument('--s', type=float, default=64, help='arcmargin parameter')
        parser.add_argument('--m', type=float, default=0.5, help='arcmargin parameter')

        parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
        parser.add_argument('--save_freq', type=int, default=30, help='save frequency')

        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma')
        parser.add_argument('--seed', type=int, default=2, help='random seed')

        return parser

    def __init__(self):
        parser = self.build_parser
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.path = os.path.join('train_exp', self.name)
        self.gpus_str = self.gpus
        self.gpus = parse_gpus(self.gpus)
