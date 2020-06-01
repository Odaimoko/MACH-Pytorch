import os
from collections import Counter
from multiprocessing import Pool
from typing import Dict, List
import time
from functools import wraps
import numpy as np
from sklearn.utils import murmurhash3_32 as mmh3
from dataset import XCDataset
import yaml
import torch
import torch.nn.functional as F
from xclib.evaluation import xc_metrics
import scipy
from torchnet import meter


# ─── DECORATORS ─────────────────────────────────────────────────────────────────


def log_time(*text, record = None):
    def real_deco(func):
        @wraps(func)
        def impl(*args, **kw):
            start = time.perf_counter()
            func(*args, **kw)
            end = time.perf_counter()
            r = print if not record else record  # 如果没有record，默认print
            t = (func.__name__,) if not text else text
            # print(r, t)
            r(*t, "Time elapsed: %.3f" % (end - start))
        
        return impl
    
    return real_deco


# ─── TRAINING AND EVALUATION ────────────────────────────────────────────────────


class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def compute_scores(model, loader):
    """
        Get all scores. For the sake of inverse propensity, we need to first collect all labels.
        TODO: -  of course we can compute it in advance
    :param model:
    :param loader:
    :return: gt & pred: num_instances x num_labels. loss: scalar
    :return:
    """
    cuda = torch.cuda.is_available()
    if cuda and not model.is_cuda():
        model = model.cuda()
    gt = []
    scores = []
    loss_func = torch.nn.BCEWithLogitsLoss()
    loss_meter = AverageMeter()
    map_meter = meter.mAPMeter()
    for i, data in enumerate(loader):
        X, y = data
        X = X.to_dense().squeeze()
        y = y.to_dense().squeeze()
        if cuda:
            X = X.cuda()
            y = y.cuda()
        out = model(X)
        loss_meter.update(loss_func(out, y), X.shape[0])
        out = F.softmax(out, 1)
        map_meter.add(out.detach(), y)  # map_meter uses softmax scores -
        # or whatever? scoring function is monotonic
        # append cuda tensor
        gt.append(y)
        scores.append(out)
    gt = scipy.sparse.csr_matrix(torch.cat(gt).cpu().numpy())
    scores = torch.cat(scores).cpu().detach().numpy()
    mAP = map_meter.value()
    return gt, scores, loss_meter.avg, mAP


def compute_scores_all(models: List, loader):
    pass


def evaluate_single(model: torch.nn.Module, loader, model_cfg):
    """
        Return quite a few measurement scores, only for a single repetition
    """
    # p@k, psp@k, ndcg, psndcg, loss, map
    gt, pred, loss, mAP = compute_scores(model, loader)
    inv_propen = xc_metrics.compute_inv_propesity(gt, model_cfg["ps_A"], model_cfg["ps_B"])
    
    acc = xc_metrics.Metrics(true_labels = gt, inv_psp = inv_propen,
                             remove_invalid = False)
    d = {k: {} for k in model_cfg["at_k"]}
    for k in d:
        prec, ndcg, PSprec, PSnDCG = acc.eval(pred, k)
        d[k] = {
            "prec": prec,
            "ndcg": ndcg,
            "psp": PSprec,
            "psn": PSnDCG
        }
    
    return loss, d, mAP


def evaluate_all(models: List, loader):
    """
        Use all models and average their results
    """
    pass


# ─── PREPROCESS ─────────────────────────────────────────────────────────────────


@log_time("Label hash...")
def label_hash(num_labels, b, r, dir_path):
    """
        Save label->[b] mapping results to a file
        mapping: a vector of length num_labels. Each entry is an integer in [0,b).
        counts: cum sum of the number of mapped results
        inv_mapping: mapping from int [0,b) to the original labels. 
            Since we train each model separately, we save R different files.
    """
    labels = np.arange(num_labels, dtype = np.int32)
    mapping = mmh3(labels, seed = r) % b
    ctr = Counter(mapping)
    counts = [0] + [ctr[k] for k in range(b)]
    # we want counts[k] to be the index that the k'th mapped label starts from
    counts = np.cumsum(counts)
    rolling_counts = [0 for i in range(b)]
    inv_mapping = np.zeros(num_labels, dtype = int)
    for i in range(num_labels):
        bucket = mapping[i]
        idx = rolling_counts[bucket] + counts[bucket]
        inv_mapping[idx] = i
        rolling_counts[bucket] += 1
    name = ["counts", "mapping", "inv_mapping"]
    var = [counts, mapping, inv_mapping]
    mkdir(dir_path)
    for n, v in zip(name, var):
        np.save(os.path.join(dir_path, "_".join([n, str(r)]) + ".npy"), v)


@log_time("Feature hash...")
def feature_hash(original_dim, dest_dim, r, dir_path):
    """
        Save #_original_dim->#_feat_dim mapping results to a file
    """
    features = np.arange(original_dim, dtype = np.int32)
    mapping = mmh3(features, seed = r) % dest_dim
    mkdir(dir_path)
    np.save(os.path.join(dir_path, "_".join(
        ["feature_hash", str(r)]) + ".npy"), mapping)


# ─── MISC ─────────────────────────────────────────────────────────────────────


def get_config(path) -> Dict:
    if os.path.exists(path):
        with open(path, 'r', encoding = 'utf8') as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(path)


def get_loader(data_cfg, model_cfg, rep):
    """
        Return train, val and test loader 
    """
    name = data_cfg['name']
    data_dir = os.path.join("data", name)
    train_file = name + "_" + "train.txt"
    test_file = name + "_" + "test.txt"
    train_file = os.path.join(data_dir, train_file)
    test_file = os.path.join(data_dir, test_file)
    train_set = XCDataset(train_file, rep, data_cfg, model_cfg, 'tr')
    val_set = XCDataset(train_file, rep, data_cfg, model_cfg, 'val')
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size = model_cfg['batch_size'])
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size = model_cfg['batch_size'])
    test_set = XCDataset(test_file, rep, data_cfg, model_cfg, 'te')
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size = model_cfg['batch_size'])
    
    return train_loader, val_loader, test_loader


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
    dirs = ['models', 'log', ]


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    p = ArgumentParser()
    p.add_argument("-m", "--module", dest = "module", type = str, required = True)
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
