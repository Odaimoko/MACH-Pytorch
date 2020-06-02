from mach_utils import label_hash, feature_hash, get_config
from argparse import ArgumentParser
import os


def get_args():
    p = ArgumentParser()
    p.add_argument("--model", dest="model", type=str, required=True)
    p.add_argument("--dataset", dest="dataset", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    a = get_args()
    data_cfg = get_config(a.dataset)
    model_cfg = get_config(a.model)
    num_labels = data_cfg['num_labels']
    ori_dim = data_cfg['ori_dim']
    dest_dim = model_cfg['dest_dim']
    b = model_cfg['b']
    R = model_cfg['r']
    record_dir = data_cfg["record_dir"]

    label_path = os.path.join(record_dir, "_".join(
        [data_cfg['name'], str(num_labels), str(b), str(R)]))  # Bibtex_159_100_32
    feat_path = os.path.join(record_dir, "_".join(
        [data_cfg['name'], str(ori_dim), str(dest_dim)]))  # Bibtex_1836_200
    for r in range(R):
        label_hash(num_labels, b, r, label_path)
        feature_hash(ori_dim, dest_dim, r, feat_path)
