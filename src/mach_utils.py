import os
from collections import Counter
from multiprocessing import Pool
from typing import Dict, List
import time
from functools import wraps
import numpy as np
from sklearn.utils import murmurhash3_32 as mmh3
import yaml
import torch
import torch.nn.functional as F
from xclib.evaluation import xc_metrics
import scipy
from torchnet import meter
import logging


# ─── DECORATORS ─────────────────────────────────────────────────────────────────


def log_time(*text, record = None):
    def real_deco(func):
        @wraps(func)
        def impl(*args, **kw):
            start = time.perf_counter()
            res=func(*args, **kw)
            end = time.perf_counter()
            r = print if not record else record  # 如果没有record，默认print
            t = (func.__name__,) if not text else text
            # print(r, t)
            r(" ".join(t) + " " + "Time elapsed: %.3f s." % (end - start))
            return res
        return impl
    
    return real_deco


# ─── TRAINING AND EVALUATION ────────────────────────────────────────────────────
def get_model_dir(data_cfg, model_cfg, rep):
    return os.path.join(model_cfg["model_dir"], data_cfg["name"], "_".join([
        "B", str(model_cfg["b"]), "R", str(model_cfg["r"]), "feat", str(model_cfg["dest_dim"]),
        "hidden", str(model_cfg['hidden']),
        "rep", "%02d" % rep
    ]))


def get_mapped_labels(y, label_mapping, b):
    if y.is_sparse:
        nz = y.coalesce().indices()  # get indices of nonzero entries
    else:
        nz = y.nonzero().T
    nz[1, :] = label_mapping[nz[1, :]]
    y = torch.sparse_coo_tensor(nz, torch.ones(nz.shape[1]), size = (y.shape[0], b)).to_dense().clamp_max(1)
    return y


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


@log_time( record = logging.info)
def compute_scores(model, loader, label_mapping = None, b = None):
    """
        Get all scores. For the sake of inverse propensity, we need to first collect all labels.
        TODO: -  of course we can compute it in advance
    :param model:
    :param loader:
    :return: gt & pred: num_instances x num_labels. loss: scalar
    :return:
    """
    
    cuda = torch.cuda.is_available()
    if cuda and not isinstance(model, torch.nn.DataParallel) and not model.is_cuda():
        model = model.cuda()
    gt = []
    scores = []
    loss_func = torch.nn.BCEWithLogitsLoss()
    loss_meter = AverageMeter()
    map_meter = meter.mAPMeter()
    for i, data in enumerate(loader):
        X, y = data
        X = X.to_dense()
        if label_mapping is not None:
            y = get_mapped_labels(y, label_mapping, b)
        if cuda:
            X = X.cuda()
            y = y.cuda()
        out = model(X)
        if label_mapping is not None:
            loss_meter.update(loss_func(out, y), X.shape[0])
        out = F.softmax(out, 1)
        if label_mapping is not None:
            map_meter.add(out.detach(), y)  # map_meter uses softmax scores -
        # or whatever? scoring function is monotonic
        # append cuda tensor
        gt.append(y)
        scores.append(out)
    gt = torch.cat(gt)
    scores = torch.cat(scores)
    if gt.is_sparse:
        gt = gt.to_dense()
    if scores.is_sparse:
        scores = scores.to_dense()
    gt = scipy.sparse.csr_matrix(gt.cpu().numpy())
    scores = scores.cpu().detach().numpy()
    mAP = map_meter.value()
    return gt, scores, loss_meter.avg, mAP


def evaluate_scores(gt, scores, model_cfg):
    inv_propen = xc_metrics.compute_inv_propesity(gt, model_cfg["ps_A"], model_cfg["ps_B"])
    
    acc = xc_metrics.Metrics(true_labels = gt, inv_psp = inv_propen,
                             remove_invalid = False)
    map_meter = meter.mAPMeter()
    
    map_meter.add(scores,gt.todense())
    prec, ndcg, PSprec, PSnDCG = acc.eval(scores, model_cfg["at_k"])
    d = {
        "prec": prec,
        "ndcg": ndcg,
        "psp": PSprec,
        "psndcg": PSnDCG,
        "mAP": [map_meter.value()]
    }
    return d


def format_evaluation(d: Dict):
    if len(d) == 0:
        print("Empty Evaluation Dictionary.")
    
    s = ""
    for k in d.keys():
        s += k + '\t' + '\t'.join([str("%.2f" % (score * 100)) for score in d[k]]) + '\n'
    at_k = len(d[k])
    s = '\t\t' + '\t\t'.join(['@%d' % (i + 1) for i in range(at_k)]) + '\n' + s
    return s


def log_eval_results(d):
    s = format_evaluation(d)
    for l in s.split('\n'):
        logging.info(l)


def evaluate_single(model: torch.nn.Module, loader, model_cfg, label_mapping):
    """
        Return quite a few measurement scores, only for a single repetition
    """
    # p@k, psp@k, ndcg, psndcg, loss, map
    gt, pred, loss, mAP = compute_scores(model, loader, label_mapping, model_cfg["b"])
    d = evaluate_scores(gt, pred, model_cfg)
    return loss, d, mAP


def evaluate_all(models: List, loader, model_cfg):
    """
        Use all models and average their results.
        For now, sequentially run.
    """
    pred = []
    gt = None
    for m in models:
        gt, p, _, _ = compute_scores(m, loader)
        pred.append(p)  # each = num_instances x num_labels
    if gt is None:
        raise Exception("You must have at least one model.")
    else:
        pred = np.stack(pred)  # R x num_ins x num_lab
        scores = pred.mean(axis = 0)
        d = evaluate_scores(gt, scores, model_cfg)
        return d


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


def get_label_hash(dir_path, r):
    """
        load label mapping
        return: counts, mapping, inv_mapping
    """
    
    name = ["counts", "mapping", "inv_mapping"]
    return [np.load(os.path.join(dir_path, "_".join([n, str(r)]) + ".npy")) for n in name]


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
