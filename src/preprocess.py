from mach_utils import label_hash, feature_hash, get_config
from argparse import ArgumentParser


def get_args():
    p = ArgumentParser()
    return p.parse_args()


if __name__ == "__main__":
    a = get_args()
    cfg = get_config(a.cfg)
    # label hashing
    # feature hashing
