import os
from xclib.data import data_utils
import json
from mach_utils import *
import yaml
from argparse import ArgumentParser


def get_args():
    p = ArgumentParser()
    p.add_argument("--dataset", "-d", dest = "dataset", type = str, required = True,
                   help = "Dataset name. Initial should be CAPITAL.")
    p.add_argument("--type", "-t", dest = "type", type = str, required = False, default = "cumsum",
                   help = "Should be 'cumsum' or 'rank'.")
    return p.parse_args()


def get_discard_set(filepath, type, rate):
    assert type in ['cumsum', 'rank']
    # count labels -> where to? only in train
    features, labels, num_samples, num_features, num_labels = data_utils.read_data(
        filepath)
    # get labels with few instances
    instance, labels_flatten = labels.nonzero()
    count = Counter(labels_flatten)
    count_np = np.zeros(num_labels).astype(np.int32)
    for k, v in count.items():
        count_np[k] = v
    idx = np.argsort(count_np)
    sorted_count = np.sort(count_np)
    if type == 'cumsum':
        percentile = np.cumsum(sorted_count) / sorted_count.sum()
        discard_sets = [set(idx[np.nonzero(percentile < r)]) for r in rate]
    elif type == 'rank':
        discard_sets = [set(idx[0:int(len(idx) * r)]) for r in rate]
    return discard_sets, count_np


if __name__ == "__main__":
    # read in files
    a = get_args()
    name = a.dataset
    filepath = 'data/{n1}/{n1}_train.txt'.format(n1 = name)
    print(filepath)
    
    # mapping old labels to new labels. we need new labels for training, and the mapping for testing.
    rate = [0.1 * i for i in range(1, 10)]
    discard_sets, count_np = get_discard_set(filepath, a, rate)
    all_label_set = set(range(num_labels))
    rest_labels = [all_label_set - d for d in discard_sets]
    label_mapping = []
    for rest in rest_labels:
        rest = sorted(list(rest))
        mapping = {
            # label: [new label, number of instances]
            l: [i, int(count_np[l])] for i, l in enumerate(rest)
        }
        label_mapping.append(mapping)
    prefixes = ['{n2}_trim{type}{rate:.1f}'.format(
        n2 = name, rate = r, type = a.type) for r in rate]
    meta_info_filepath = [
        'data/{n1}/{pref}_meta.json'.format(n1 = name, pref = p) for p in prefixes]
    for mapping, meta_path in zip(label_mapping, meta_info_filepath):
        with open(meta_path, 'w') as f:
            json.dump(mapping, f)  # will save kv pair in the type of str:int
    
    # for each line, if there are labels which should be deleted, delete it from labels
    # if no labels are left, delete the whole line
    # write to the new train file
    trimmed_filepath = [
        'data/{n1}/{pref}_train.txt'.format(n1 = name, pref = p) for p in prefixes]
    
    fps = [open(fp, 'w') for fp in trimmed_filepath]
    num_instances = [0 for i in rate]
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            # in Delicious, there is one line which does not have labels
            # the third criterion is to remove the first meta info line
            if not line or ':' in line[0] or ':' not in line[1]:
                continue
            
            labels = [int(i)
                      for i in line[0].split(',')]
            for i, blob in enumerate(zip(label_mapping, discard_sets, fps)):
                mapping, d, fp = blob
                new_labels = [str(mapping[l][0]) for l in labels if l not in d]
                if not new_labels:  # all labels are removed
                    continue
                new_line = [",".join(new_labels)] + line[1:]
                fp.write(" ".join(new_line) + "\n")
                num_instances[i] += 1
    for fp in fps:
        fp.close()
    # write yaml config files
    config_dir = 'config/data_trim/' + name
    mkdir(config_dir)
    
    data = os.path.join(
        'config/dataset', "{n}.yaml".format(n = name.lower()))  # eurlex.yaml
    ori_data_cfg = get_config(data)
    for p, rest, n in zip(prefixes, rest_labels, num_instances):
        data_cfg = ori_data_cfg.copy()
        yaml_path = os.path.join(config_dir, "%s.yaml" % (p))
        data_cfg['trimmed'] = True
        data_cfg['ori_labels'] = data_cfg['num_labels']
        data_cfg['prefix'] = p
        data_cfg['num_labels'] = len(rest)
        data_cfg['train_size'] = int(n * .9)
        with open(yaml_path, 'w') as f:
            yaml.dump(data_cfg, f)
