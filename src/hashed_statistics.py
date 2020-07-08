from mach_utils import *
import logging
from argparse import ArgumentParser
from fc_network import FCNetwork
import tqdm
from dataset import XCDataset
import json
from typing import Dict, List
from trim_labels import get_discard_set
from torchnet.meter import APMeter
from xclib.data import data_utils
from matplotlib import pyplot as plt


def get_args():
    p = ArgumentParser()
    p.add_argument("--model", '-m', dest = "model", type = str, required = True,
                   help = "Path to the model config yaml file.")
    p.add_argument("--dataset", '-d', dest = "dataset", type = str, required = True,
                   help = "Path to the data config yaml file.")
    return p.parse_args()


def get_inv_hash(counts, inv_mapping, j):
    """

    :param counts:
    :param inv_mapping:
    :param j: \in [0,b), the index we want to  map back. Can be a tensor
    :return:
    """
    labels = inv_mapping[counts[j]: counts[j + 1]]
    return labels


if __name__ == "__main__":
    a = get_args()
    
    data_cfg = get_config(a.dataset)
    model_cfg = get_config(a.model)
    model_dir = os.path.join(model_cfg["model_dir"], data_cfg["prefix"])
    
    cuda = torch.cuda.is_available()
    R = model_cfg['r']
    b = model_cfg['b']
    num_labels = data_cfg["num_labels"]
    dest_dim = model_cfg['dest_dim']
    name = data_cfg['name']
    prefix = data_cfg['prefix']
    record_dir = data_cfg["record_dir"]
    data_dir = os.path.join("data", name)
    label_path = os.path.join(record_dir, "_".join(
        [prefix, str(num_labels), str(b), str(R)]))  # Bibtex_159_100_32
    
    test_file = os.path.join(data_dir, name + "_test.txt")
    features, labels, num_samples, num_features, num_labels = data_utils.read_data(
        test_file)
    instance, labels_flatten = labels.nonzero()
    count = Counter(labels_flatten)
    unsorted_count = np.zeros(num_labels, dtype = np.int)
    
    for k, v in count.items():
        unsorted_count[k] = v
    sorted_idx = np.flip(np.argsort(unsorted_count))
    thres = [5, 10, 20]
    for r in tqdm.tqdm(range(R)):
        # use feature hashing to map back
        counts, label_mapping, inv_mapping = get_label_hash(label_path, r)
        mapped_labels = label_mapping[labels_flatten]
        m_count = Counter(mapped_labels)
        zero_count = b - len(m_count)
        # hashed label -> hashed count
        mapped_unsorted_count = np.zeros(b, dtype = np.int32)
        for k, v in m_count.items():
            mapped_unsorted_count[k] = v
        # ori label -> hashed count
        ori_label_mapped_count = mapped_unsorted_count[label_mapping]
        m_sorted_count = np.flip(np.sort(mapped_unsorted_count))
        
        plt.clf()
        
        # plt.plot(range(1, len(ori_label_mapped_count) + 1), ori_label_mapped_count[sorted_idx])
        # plt.plot(range(1, len(unsorted_count) + 1), unsorted_count[sorted_idx])
        # plt.plot(range(1, len(mapped_unsorted_count) + 1), mapped_unsorted_count)
        plt.plot(range(1, len(m_sorted_count) + 1), m_sorted_count)
        plt.xlabel("Label")
        # plt.yscale('log')
        plt.ylabel("Count")
        file = os.path.join(label_path, 'rep_%02d_label_count' % (r))
        
        plt.title(file)
        plt.savefig(file + '.svg')
        
        less_than = [len([c for c in m_sorted_count if c <= t]) for t in thres]
        print(
            "%s\n\t%d (%.2f%%) out of %d labels with no instances, \n\t "
            % (file, zero_count, zero_count * 100 / b, b) +
            ", ".join(["%d (%.2f%%) fewer than %d" % (num, 100 * num / num_labels, t)
                       for num, t in zip(less_than, thres)]) +
            "\n\t Max: %d, Min: %d, Avg: %.2f, Std: %.2f" % (
                m_sorted_count[0], m_sorted_count[-1], m_sorted_count.mean(), m_sorted_count.std())
        )
