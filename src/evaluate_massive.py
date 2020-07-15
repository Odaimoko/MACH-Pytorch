from mach_utils import *
import logging
from argparse import ArgumentParser
from fc_network import FCNetwork
import tqdm
from dataset_massive import XCDataset
import json
from typing import Dict, List
from trim_labels import get_discard_set
from xclib.evaluation import xc_metrics
from xclib.data import data_utils
from torchnet import meter


def get_args():
    p = ArgumentParser()
    p.add_argument("--model", '-m', dest = "model", type = str, required = True,
                   help = "Path to the model config yaml file.")
    p.add_argument("--dataset", '-d', dest = "dataset", type = str, required = True,
                   help = "Path to the data config yaml file.")
    p.add_argument("--gpus", '-g', dest = "gpus", type = str, required = False, default = "0",
                   help = "A string that specifies which GPU you want to use, split by comma. Eg 0,1")
    
    p.add_argument("--cost", '-c', dest = "cost", type = str, required = False, default = '',
                   help = "Use cost-sensitive model or not. Should be in [hashed, original]. "
                          "Default empty string, which indicates that no cost-sensitive is used.")
    p.add_argument("--type", '-t', dest = "type", type = str, required = False, default = "all",
                   help = """Evaluation type. Should be 'all'(default) and/or 'trim_eval', split by comma. Eg. 'all,trim_eval'. If it is 'trim_eval', the rate parameter should be specified.
                   'all': Evaluate normally. If the 'trimmed' field in data config file is true, the code will automatically map the rest of the labels back to the orginal ones.
                   'trim_eval': Trim labels when evaluating. The scores with tail labels will be set to 0 in order not to predict these ones. This checks how much tail labels affect final evaluation metrics. Plus it will evaluate average precision on tail and head labels only. 
                   """)
    p.add_argument("--rate", '-r', dest = "rate", type = str, required = False, default = "0.1",
                   help = """If evaluation needs trimming, this parameter specifies how many labels will be trimmed, decided by cumsum.
                   Should be a string containing trimming rates split by comma. Eg '0.1,0.2'. Default '0.1'.""")
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
    # load ground truth
    a.__dict__['rep'] = r
    model_dir = get_model_dir(data_cfg, model_cfg, a)
    # load mapping
    counts, label_mapping, inv_mapping = get_label_hash(label_path, r)
    label_mapping = torch.from_numpy(label_mapping)
    # load models
    best_param = os.path.join(model_dir, model_cfg["best_file"])
    preload_path = model_cfg["pretrained"] if model_cfg["pretrained"] else best_param
    if os.path.exists(preload_path):
        meta_info = torch.load(preload_path)
        model.load_state_dict(meta_info['model'])
    else:
        raise FileNotFoundError(
            "Model {} does not exist.".format(preload_path))
    # predict. gt: original label. p: hashed.
    gt, p, _, _ = compute_scores(model, test_loader)
    return gt, p[:, label_mapping]


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
    feat_path = os.path.join(record_dir, "_".join([prefix, str(ori_dim), str(dest_dim)]))
    
    # load dataset
    test_file = os.path.join(data_dir, prefix + "_test.txt")
    # this will take a lot of space!!!!!!
    test_set = XCDataset(test_file, 0, data_cfg, model_cfg, 'te')
    # test_sets = [XCDataset(test_file, r, data_cfg, model_cfg, 'te') for r in range(R)]
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size = a.bs)
    # construct model
    layers = [dest_dim] + model_cfg['hidden'] + [b]
    model = FCNetwork(layers)
    if cuda:
        model = torch.nn.DataParallel(model, device_ids = gpus).cuda()
    label_path = os.path.join(record_dir, "_".join(
        [prefix, str(num_labels), str(b), str(R)]))  # Bibtex_159_100_32
    
    pred_avg_meter = AverageMeter()
    gt = None
    logging.info("Evaluating config %s" % (a.model))
    logging.info("Dataset config %s" % (a.dataset))
    if a.cost:
        logging.info("Evaluating cost-sensitive method: %s" % (a.cost))
    
    # get inverse propensity
    
    _, labels, _, _, _ = data_utils.read_data(test_file)
    inv_propen = xc_metrics.compute_inv_propesity(labels, model_cfg["ps_A"], model_cfg["ps_B"])
    gts = []
    scaled_eval_flags = []
    eval_flags = []
    ps_eval_flags = []
    map_meter = meter.mAPMeter()
    
    for i, data in enumerate(tqdm.tqdm(test_loader)):
        print(i, 'th data')
        pred_avg_meter = AverageMeter()
        X, gt = data
        bs = X.shape[0]
        for r in range(R):
            print("REP", r, end = '\t')
            x = X
            feat_mapping = get_feat_hash(feat_path, r)
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
            # load model
            a.__dict__['rep'] = r
            model_dir = get_model_dir(data_cfg, model_cfg, a)
            # load mapping
            counts, label_mapping, inv_mapping = get_label_hash(label_path, r)
            label_mapping = torch.from_numpy(label_mapping)
            # load models
            best_param = os.path.join(model_dir, model_cfg["best_file"])
            preload_path = model_cfg["pretrained"] if model_cfg["pretrained"] else best_param
            if os.path.exists(preload_path):
                meta_info = torch.load(preload_path)
                model.load_state_dict(meta_info['model'])
            else:
                raise FileNotFoundError(
                    "Model {} does not exist.".format(preload_path))
            # the r_th output
            model.eval()
            with torch.no_grad():
                out = model(x)
                out = torch.sigmoid(out)
            out = out.detach().cpu().numpy()[:, label_mapping]
            pred_avg_meter.update(out, 1)
        
        if gt.is_sparse:
            gt = gt.coalesce()
            gt = scipy.sparse.coo_matrix((gt.values().cpu().numpy(),
                                          gt.indices().cpu().numpy()),
                                         shape = (bs, num_labels))
        else:
            gt = scipy.sparse.coo_matrix(gt.cpu().numpy())
        # only a batch of eval flags
        scores = pred_avg_meter.avg
        # map_meter.add(scores, gt.todense())
        
        indices, true_labels, ps_indices, inv_psp = xc_metrics. \
            _setup_metric(scores, gt, inv_propen)
        eval_flag = xc_metrics._eval_flags(indices, true_labels, None)
        ps_eval_flag = xc_metrics._eval_flags(ps_indices, true_labels, inv_psp)
        # gts.append(gt)
        scaled_eval_flag = np.multiply(inv_psp[indices], eval_flag)
        eval_flags.append(eval_flag)
        ps_eval_flags.append(ps_eval_flag)
        scaled_eval_flags.append(scaled_eval_flag)
    
    # eval all
    # gts = np.concatenate(gts)
    scaled_eval_flags = np.concatenate(scaled_eval_flags)
    eval_flags = np.concatenate(eval_flags)
    ps_eval_flags = np.concatenate(ps_eval_flags)
    
    ndcg_denominator = np.cumsum(
        1 / np.log2(np.arange(1, num_labels + 1) + 1))
    _total_pos = np.asarray(
        labels.sum(axis = 1),
        dtype = np.int32)
    
    n = ndcg_denominator[_total_pos - 1]
    prec = xc_metrics._precision(eval_flags, K)
    ndcg = xc_metrics._ndcg(eval_flags, n, K)
    
    PSprec = xc_metrics._precision(scaled_eval_flags, K) / xc_metrics._precision(ps_eval_flags, K)
    PSnDCG = xc_metrics._ndcg(scaled_eval_flags, n, K) / xc_metrics._ndcg(ps_eval_flags, n, K)
    d = {
        "prec": prec,
        "ndcg": ndcg,
        "psp": PSprec,
        "psn": PSnDCG,
        "mAP": [map_meter.value()]
    }
    log_eval_results(d)
    # map trimmed labels back to original ones
    # scores = pred_avg_meter.avg
    # types = a.type.split(',')
    # if 'all' in types:
    #     if data_cfg['trimmed']:
    #         # if use trim_eval or only_tail, data_cfg['trimmed'] should be false
    #         scores = map_trimmed_back(
    #             scores, data_dir, prefix, data_cfg['ori_labels'])
    #
    #     if gt is None:
    #         raise Exception("You must have at least one model.")
    #     else:
    #         #  Sum of avg is larger than 1 -> that is the feature, no problem
    #         d = evaluate_scores(gt, scores, model_cfg)
    #         log_eval_results(d)
    #
    # if 'trim_eval' in types or 'only_tail' in types:
    #     #   find tail labels using  training set.
    #     filepath = 'data/{n1}/{n1}_train.txt'.format(n1 = name)
    #     print(filepath)
    #     rate = [float(f) for f in a.rate.split(',')]
    #     discard_sets, count_np = get_discard_set(filepath, 'cumsum', rate)
    #     all_label_set = set(range(num_labels))
    #     rest_labels = [all_label_set - d for d in discard_sets]
    #     if 'trim_eval' in types:
    #         for r, dis_set, rest in zip(rate, discard_sets, rest_labels):
    #             logging.info(
    #                 "Evaluate when trimming off {num_dis} labels (cumsum rate: {rate:.2f}%%, actual rate: {r2:.2f}%%)".format(
    #                     num_dis = len(dis_set), rate = r * 100, r2 = len(dis_set) / num_labels * 100))
    #             dis_list = sorted(list(dis_set))
    #             rest_list = sorted(list(rest))
    #             new_score = np.copy(scores)
    #             new_score[:, dis_list] = 0
    #             log_eval_results(evaluate_scores(gt, new_score, model_cfg))
    #
    #             # eval on head and tail labels, using original scores
    #             ap = APMeter()
    #             ap.add(scores, gt.todense())
    #             logging.info("AP of tail labels and head labels: %.2f, %.2f.\n" % (
    #                 ap.value()[dis_list].mean() * 100, ap.value()[rest_list].mean() * 100))
