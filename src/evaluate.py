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


def get_args():
    p = ArgumentParser()
    p.add_argument("--model", '-m', dest="model", type=str, required=True,
                   help="Path to the model config yaml file.")
    p.add_argument("--dataset", '-d', dest="dataset", type=str, required=True,
                   help="Path to the data config yaml file.")
    p.add_argument("--gpus", '-g', dest="gpus", type=str, required=False, default="0",
                   help="A string that specifies which GPU you want to use, split by comma. Eg 0,1")

    p.add_argument("--cost", '-c', dest="cost", type=str, required=False, default='',
                   help="Use cost-sensitive model or not. Should be in [hashed, original]. "
                   "Default empty string, which indicates that no cost-sensitive is used.")
    p.add_argument("--type", '-t', dest="type", type=str, required=False, default="all",
                   help="""Evaluation type. Should be 'all'(default) and/or 'trim_eval', split by comma. Eg. 'all,trim_eval'. If it is 'trim_eval', the rate parameter should be specified.
                   'all': Evaluate normally. If the 'trimmed' field in data config file is true, the code will automatically map the rest of the labels back to the orginal ones.
                   'trim_eval': Trim labels when evaluating. The scores with tail labels will be set to 0 in order not to predict these ones. This checks how much tail labels affect final evaluation metrics. Plus it will evaluate average precision on tail and head labels only. 
                   """)
    p.add_argument("--rate", '-r', dest="rate", type=str, required=False, default="0.1",
                   help="""If evaluation needs trimming, this parameter specifies how many labels will be trimmed, decided by cumsum.
                   Should be a string containing trimming rates split by comma. Eg '0.1,0.2'. Default '0.1'.""")
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
    test_set.change_feat_map(r)
    a.__dict__['rep'] = r
    model_dir = get_model_dir(data_cfg, model_cfg, a)
    # load mapping
    counts, label_mapping, inv_mapping = get_label_hash(label_path, r)
    label_mapping = torch.from_numpy(label_mapping)
    # load models
    best_param = os.path.join(model_dir, model_cfg["best_file"])
    preload_path = model_cfg["pretrained"] if model_cfg["pretrained"] else best_param
    if os.path.exists(preload_path):
        if cuda:
            meta_info = torch.load(preload_path)
        else:
            meta_info = torch.load(
                preload_path, map_location=lambda storage, loc: storage)
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
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(
                                os.path.join(model_dir, log_file)),
                            logging.StreamHandler()
                        ])

    cuda = torch.cuda.is_available()
    R = model_cfg['r']
    b = model_cfg['b']
    num_labels = data_cfg["num_labels"]
    dest_dim = model_cfg['dest_dim']
    name = data_cfg['name']
    prefix = data_cfg['prefix']
    record_dir = data_cfg["record_dir"]
    data_dir = os.path.join("data", name)

    # load dataset
    test_file = os.path.join(data_dir, prefix + "_test.txt")
    test_set = XCDataset(test_file, 0, data_cfg, model_cfg, 'te')
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=model_cfg['batch_size'])
    # construct model
    layers = [dest_dim] + model_cfg['hidden'] + [b]
    model = FCNetwork(layers)
    model = torch.nn.DataParallel(model, device_ids=gpus)
    if cuda:
        model = model.cuda()
    label_path = os.path.join(record_dir, "_".join(
        [prefix, str(num_labels), str(b), str(R)]))  # Bibtex_159_100_32

    pred_avg_meter = AverageMeter()
    gt = None
    logging.info("Evaluating config %s" % (a.model))
    logging.info("Dataset config %s" % (a.dataset))
    if a.cost:
        logging.info("Evaluating cost-sensitive method: %s" % (a.cost))

    for r in tqdm.tqdm(range(R)):
        gt, ori_pred = single_rep(data_cfg, model_cfg, r)
        # use feature hashing to map back
        pred_avg_meter.update(ori_pred, 1)

    # map trimmed labels back to original ones
    scores = pred_avg_meter.avg
    types = a.type.split(',')
    if 'all' in types:
        if data_cfg['trimmed']:
            # if use trim_eval or only_tail, data_cfg['trimmed'] should be false
            scores = map_trimmed_back(
                scores, data_dir, prefix, data_cfg['ori_labels'])

        if gt is None:
            raise Exception("You must have at least one model.")
        else:
            #  Sum of avg is larger than 1 -> that is the feature, no problem
            d = evaluate_scores(gt, scores, model_cfg)
            log_eval_results(d)

    if 'trim_eval' in types or 'only_tail' in types:
        #   find tail labels using  training set.
        filepath = 'data/{n1}/{n1}_train.txt'.format(n1=name)
        print(filepath)
        rate = [float(f) for f in a.rate.split(',')]
        discard_sets, count_np = get_discard_set(filepath, 'cumsum', rate)
        all_label_set = set(range(num_labels))
        rest_labels = [all_label_set - d for d in discard_sets]
        if 'trim_eval' in types:
            for r, dis_set, rest in zip(rate, discard_sets, rest_labels):
                logging.info(
                    "Evaluate when trimming off {num_dis} labels (cumsum rate: {rate:.2f}%%, actual rate: {r2:.2f}%%)".format(
                        num_dis=len(dis_set), rate=r * 100, r2=len(dis_set) / num_labels * 100))
                dis_list = sorted(list(dis_set))
                rest_list = sorted(list(rest))
                new_score = np.copy(scores)
                new_score[:, dis_list] = 0
                log_eval_results(evaluate_scores(gt, new_score, model_cfg))

                # eval on head and tail labels, using original scores
                ap = APMeter()
                ap.add(scores, gt.todense())
                logging.info("AP of tail labels and head labels: %.2f, %.2f.\n" % (
                    ap.value()[dis_list].mean() * 100, ap.value()[rest_list].mean() * 100))
