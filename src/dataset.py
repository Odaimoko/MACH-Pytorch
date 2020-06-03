from torch.utils.data import Dataset
import os
import torch
import numpy as np


# TODO: for small dataset only, when we can load the txt file into mem


def get_feat_hash(dir_path, r):
    """
        load feature mapping
    """
    return np.load(os.path.join(dir_path, "_".join(["feature_hash", str(r)]) + ".npy"))


class XCDataset(Dataset):
    def __init__(self, txt_path, rep, data_cfg, model_cfg, type = 'tr'):
        """
              dont use label hashing here.
        :param txt_path:
        :param data_cfg:
        :param model_cfg:
        :param type: ['tr', 'val', 'te']. Train/val: use a portion of dataset. Test: Use all.
        """
        assert type in ['tr', 'val', 'te']
        self.name = data_cfg['name']
        self.ori_dim = data_cfg['ori_dim']
        self.dest_dim = model_cfg['dest_dim']
        self.ori_labels = data_cfg['num_labels']
        self.dest_labels = model_cfg['b']
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        # load hash results
        record_dir = data_cfg["record_dir"]
        feat_path = os.path.join(record_dir, "_".join(
            [self.name, str(self.ori_dim), str(self.dest_dim)]))  # Bibtex_1836_200
        self.feat_mapping = get_feat_hash(feat_path, rep)  # diff each rep
        
        # load data
        self.meta_info = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                n = -1
                for line in f:
                    if not line:
                        continue
                    n += 1
                    if type == 'tr':
                        if n >= data_cfg['train_size']:
                            break
                    if type == 'val':
                        if n < data_cfg['train_size']:
                            continue
                    line = line.strip().split()
                    labels = [int(i)
                              for i in line[0].split(',')]  # list of label index
                    idx_values_pair = [v.split(":") for v in line[1:]]
                    y = torch.sparse_coo_tensor([labels], torch.ones(len(labels)),
                                                size = (self.ori_labels,))
                    self.meta_info.append([y, idx_values_pair])
        else:
            print("Dataset %s does not exist." % (txt_path))
    
    def __getitem__(self, i):
        """
            Use hashed results to serve data
        """
        y, idx_values_pair = self.meta_info[i]
        idx = [int(i) for i, j in idx_values_pair]
        mapped_idx = self.feat_mapping[idx]
        mapped_idx = np.expand_dims(mapped_idx, 0)
        values = torch.tensor([float(j) for i, j in idx_values_pair])
        x = torch.sparse_coo_tensor(mapped_idx, values, size = (self.dest_dim,))
        return x, y
    
    def change_feat_map(self, rep):
        record_dir = self.data_cfg["record_dir"]
        feat_path = os.path.join(record_dir, "_".join(
            [self.name, str(self.ori_dim), str(self.dest_dim)]))  # Bibtex_1836_200
        self.feat_mapping = get_feat_hash(feat_path, rep)  # diff each rep
    
    def __len__(self):
        return len(self.meta_info)
