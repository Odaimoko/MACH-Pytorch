import yaml
from typing import Dict
import os
# ─── PREPROCESS ─────────────────────────────────────────────────────────────────


def label_hash(r, path):
    """
        Save label->[R] mapping results to a file
    """
    pass


def feature_hash(r, path):
    """
        Save #_original_dim->#_feat_dim mapping results to a file
    """
    pass


def get_label_hash_dict(path):
    """
        load label mapping
    """
    pass


def get_feat_hash_dict(path):
    """
        load feature mapping
    """
    pass

# ─── MISC ─────────────────────────────────────────────────────────────────────


def get_config(path) -> Dict:
    return {}


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
