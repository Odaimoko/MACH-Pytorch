import os
from collections import Counter
from multiprocessing import Pool
from typing import Dict

import numpy as np
from sklearn.utils import murmurhash3_32 as mmh3

import yaml
# ─── PREPROCESS ─────────────────────────────────────────────────────────────────


def label_hash(num_labels, b, r, dir_path):
    """
        Save label->[b] mapping results to a file
        mapping: a vector of length num_labels. Each entry is an integer in [0,b).
        counts: cum sum of the number of mapped results
        inv_mapping: mapping from int [0,b) to the original labels. 
            Since we train each model separately, we save R different files.
    """
    labels = np.arange(num_labels, dtype=int)
    mapping = mmh3(labels, seed=r) % b
    ctr = Counter(mapping)
    counts = [0]+[ctr[k] for k in range(b)]
    # we want counts[k] to be the index that the k'th mapped label starts from
    counts = np.cumsum(counts)
    rolling_counts = [0 for i in range(b)]
    inv_mapping = np.zeros(num_labels, dtype=int)
    for i in range(num_labels):
        bucket = mapping[i]
        idx = rolling_counts[bucket]+counts[bucket]
        inv_mapping[idx] = i
        rolling_counts[bucket] += 1
    name = ["counts", "mapping", "inv_mapping"]
    var = [counts, mapping, inv_mapping]
    mkdir(dir_path)
    for n, v in zip(name, var):
        np.save(os.path.join(dir_path, "_".join([n, str(r)]) + ".npy"), v)


def feature_hash(original_dim, dest_dim, r, dir_path):
    """
        Save #_original_dim->#_feat_dim mapping results to a file
    """
    features = np.arange(original_dim, dtype=int)
    mapping = mmh3(features, seed=r) % dest_dim
    mkdir(dir_path)
    np.save(os.path.join(dir_path, "_".join(
        ["feature_hash", str(r)])+".npy"), mapping)


def get_label_hash(dir_path, r):
    """
        load label mapping
        return: counts, mapping, inv_mapping
    """

    name = ["counts", "mapping", "inv_mapping"]
    return [np.load(os.path.join(dir_path, "_".join([n, str(r)]) + ".npy")) for n in name]


def get_feat_hash(dir_path, r):
    """
        load feature mapping
    """
    return np.load(os.path.join(dir_path, "_".join(["feature_hash", str(r)])+".npy"))

# ─── MISC ─────────────────────────────────────────────────────────────────────


def get_config(path) -> Dict:
    if os.path.exists(path):
        return yaml.safe_load(path)
    else:
        raise FileNotFoundError(path)


def mkdir(path):
    path = path.strip().rstrip("/")
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def create_record_dir(cfg):
    """
        Create necessary directories for training and logging
    """
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("-m", "--module", dest="module", type=str, required=True)
    a = p.parse_args()
    if a.module == "prepro":
        num_labels = 31113
        b = 1000
        R = 32
        ori_dim = 75000
        dest_dim = 3000
        suf = "random"
        dir_path = "test"
        label_path = os.path.join(dir_path, "_".join(['label', suf]))
        feat_path = os.path.join(dir_path, "_".join(['feat', suf]))
        for r in range(R):
            label_hash(num_labels, b, r, label_path)
            feature_hash(ori_dim, dest_dim, r, feat_path)
