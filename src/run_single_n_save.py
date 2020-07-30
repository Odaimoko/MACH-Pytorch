from mach_utils import *
import logging
from argparse import ArgumentParser
from fc_network import FCNetwork
import tqdm
from dataset import XCDataset, XCDataset_massive
import json
from typing import Dict, List
from trim_labels import get_discard_set
from xclib.evaluation import xc_metrics
from xclib.data import data_utils
from torchnet import meter
import time


def get_args():
    p = ArgumentParser()
    
    p.add_argument("--rep", '-r', dest = "rep", type = int, default = 0,
                   help = "Which reptition to train. Default 0.")
    p.add_argument("--model", '-m', dest = "model", type = str, required = True,
                   help = "Path to the model config yaml file.")
    p.add_argument("--dataset", '-d', dest = "dataset", type = str, required = True,
                   help = "Path to the data config yaml file.")
    p.add_argument("--gpus", '-g', dest = "gpus", type = str, required = False, default = "0",
                   help = "A string that specifies which GPU you want to use, split by comma. Eg 0,1")
    
    p.add_argument("--cost", '-c', dest = "cost", type = str, required = False, default = '',
                   help = "Use cost-sensitive model or not. Should be in [hashed, original]. "
                          "Default empty string, which indicates that no cost-sensitive is used.")
    p.add_argument("--batch_size", '-bs', dest = "bs", type = int, required = False, default = "32",
                   help = """Evaluation batch size.""")
    
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


def single_rep(data_cfg, model_cfg, r):
    return


def map_trimmed_back(scores, data_dir, prefix, ori_labels):
    mapping_file = os.path.join(data_dir, prefix + "_meta.json")
    with open(mapping_file, 'r') as f:
        trim_mapping: Dict = json.load(f)
    reverse_mapping = {v[0]: int(k) for k, v in trim_mapping.items()}
    reverse_mapping_tensor = torch.tensor(
        [reverse_mapping[k] for k in sorted(reverse_mapping.keys())])
    
    num_ins = scores.shape[0]
    ori_scores = np.zeros([num_ins, ori_labels])
    ori_scores[:, reverse_mapping_tensor] = scores
    scores = ori_scores
    return scores


def sanity_check(a):
    assert a.type in ['all', 'trim_eval', 'only_tail']


if __name__ == "__main__":
    a = get_args()
    r = a.rep
    gpus = [int(i) for i in a.gpus.split(",")]
    
    data_cfg = get_config(a.dataset)
    model_cfg = get_config(a.model)
    log_file = data_cfg['prefix'] + "_eval.log"
    model_dir = os.path.join(model_cfg["model_dir"], data_cfg["prefix"])
    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s %(levelname)-8s %(message)s', datefmt = '%Y-%m-%d %H:%M:%S',
                        handlers = [
                            logging.FileHandler(
                                os.path.join(model_dir, log_file)),
                            logging.StreamHandler()
                        ])
    
    cuda = torch.cuda.is_available()
    R = model_cfg['r']
    b = model_cfg['b']
    num_labels = data_cfg["num_labels"]
    ori_dim = data_cfg['ori_dim']
    dest_dim = model_cfg['dest_dim']
    name = data_cfg['name']
    prefix = data_cfg['prefix']
    record_dir = data_cfg["record_dir"]
    data_dir = os.path.join("data", name)
    K = model_cfg['at_k']
    feat_path = os.path.join(record_dir, "_".join(
        [prefix, str(ori_dim), str(dest_dim)]))
    
    # load dataset
    test_file = os.path.join(data_dir, prefix + "_test.txt")
    # this will take a lot of space!!!!!!
    test_set = XCDataset_massive(test_file, 0, data_cfg, model_cfg, 'te')
    # test_sets = [XCDataset(test_file, r, data_cfg, model_cfg, 'te') for r in range(R)]
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size = a.bs)
    # construct model
    layers = [dest_dim] + model_cfg['hidden'] + [b]
    model = FCNetwork(layers)
    model = torch.nn.DataParallel(model, device_ids = gpus)
    if cuda:
        model = model.cuda()
    label_path = os.path.join(record_dir, "_".join(
        [prefix, str(num_labels), str(b), str(R)]))  # Bibtex_159_100_32
    
    pred_avg_meter = AverageMeter()
    
    # load model
    model_dir = get_model_dir(data_cfg, model_cfg, a)
    best_param = os.path.join(model_dir, model_cfg["best_file"])
    preload_path = model_cfg["pretrained"] if model_cfg["pretrained"] else best_param
    if os.path.exists(preload_path):
        start = time.perf_counter()
        if cuda:
            meta_info = torch.load(preload_path)
        else:
            meta_info = torch.load(
                preload_path, map_location = lambda storage, loc: storage)
        meta_info['model'] = dict(meta_info['model'])
        for k in list(meta_info['model'].keys()):  # load un-dataparallel model into dataparallel
            if "module." not in k:
                meta_info['model']["module." + k] = meta_info['model'][k]
                del meta_info['model'][k]
        model.load_state_dict(meta_info['model'])
        end = time.perf_counter()
        logging.info("Load model time: %.3f s." % (end - start))
    
    else:
        raise FileNotFoundError(
            "Model {} does not exist.".format(preload_path))
    logging.info("Evaluating config %s" % (a.model))
    logging.info("Dataset config %s" % (a.dataset))
    if a.cost:
        logging.info("Evaluating cost-sensitive method: %s" % (a.cost))
    
    # get inverse propensity
    
    outs = []
    gts = []
    feat_mapping = get_feat_hash(feat_path, r)
    
    num_keep = int(b * .1)
    for i, data in enumerate(tqdm.tqdm(test_loader)):
        X, gt = data
        bs = X.shape[0]
        x = X
        if model_cfg['is_feat_hash']:
            x = x.coalesce()
            ind = x.indices()
            v = x.values()
            ind[1] = torch.from_numpy(feat_mapping[ind[1]])
            x = torch.sparse_coo_tensor(ind, values = v, size = (bs, dest_dim))
        else:
            pass
        x = x.to_dense()
        if cuda:
            x = x.cuda()
        start = time.perf_counter()
        
        model.eval()
        with torch.no_grad():
            out = model(x)
            out = torch.sigmoid(out)
        val, ind = torch.sort(out, descending = True, dim = 1)
        val=val[:,:num_keep].flatten().numpy()
        ind_label=ind[:,:num_keep].numpy()
        ind_instances=np.arange(bs).repeat(num_keep)
        indices = np.stack((ind_instances,ind_label.flatten()))
        out_sparse = scipy.sparse.coo_matrix((val,
                                         indices),
                                        shape = (bs, b))
        # out = out.detach().cpu().numpy()
        outs.append(out_sparse)
        
        if gt.is_sparse:
            gt_np = gt.coalesce()
            gt_np = scipy.sparse.coo_matrix((gt_np.values().cpu().numpy(),
                                             gt_np.indices().cpu().numpy()),
                                            shape = (bs, num_labels))
        else:
            gp_np = gt.numpy()
        gts.append(gt_np)
        end = time.perf_counter()
        # logging.info("Single model running time: %.3f s." % (end - start))
    outs = scipy.sparse.vstack(outs).tocsc()
    scipy.sparse.save_npz(os.path.join(model_dir, "pred.npz"), outs)
    gts = scipy.sparse.vstack(gts)
    scipy.sparse.save_npz(os.path.join(model_dir, "gt.npz"), gts)
