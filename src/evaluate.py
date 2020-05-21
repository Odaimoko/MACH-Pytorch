from argparse import ArgumentParser
from mach_utils import get_config


def get_args():
    p = ArgumentParser()
    return p.parse_args()


if __name__ == "__main__":
    a = get_args()
    cfg = get_config(a.cfg)

    # load model

    # load dataset

    # load ground truth

    # predict

    # evaluate using PyXCLib
