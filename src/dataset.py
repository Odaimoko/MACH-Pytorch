from torch.utils.data import Dataset
import os
import torch
# for small dataset only, when we can load the txt file into mem


class XCDataset(Dataset):
    def __init__(self, txt_path, data_cfg, model_cfg):
        self.ori_dim = data_cfg['ori_dim']
        self.dest_dim = model_cfg['dest_dim']
        self.ori_labels = data_cfg['num_labels']
        self.dest_labels = model_cfg['b']
        self.meta_info = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    if not line:
                        continue
                    line = line.strip().split()
                    labels = [int(i)
                              for i in line[0].split(',')]  # list of label index
                    values = [v.split(":") for v in line[1:]]
                    self.meta_info.append([labels, values])

    def __getitem__(self, i):
        """
            Use hashed results to serve data
        """
        label, idx_values_pair = self.meta_info[i]
        idx = [(0, int(i)) for i, j in idx_values_pair]
        idx_tensor = torch.LongTensor(idx).T
        values = [float(j) for i, j in idx_values_pair]
        x = torch.sparse_coo_tensor(idx_tensor, values, size=(1, self.ori_dim))
        label = [(0, j) for j in label]
        label_tensor = torch.LongTensor(label).T
        y = torch.sparse_coo_tensor(label_tensor, torch.ones(
            len(label)), size=(1, self.ori_labels))
        return x, y

    def __len__(self):
        return len(self.meta_info)
