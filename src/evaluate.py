from mach_utils import *
import logging
from argparse import ArgumentParser
from fc_network import FCNetwork
import tqdm
from dataset import XCDataset


def get_args():
    p = ArgumentParser()
    p.add_argument("--model", dest="model", type=str, required=True,
                   help="Path to the model config yaml file.")
    p.add_argument("--dataset", dest="dataset", type=str, required=True,
                   help="Path to the data config yaml file.")
    p.add_argument("--gpus", dest="gpus", type=str, required=False, default="0",
                   help="A string that specifies which GPU you want to use, split by comma. Eg 0,1")
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


def evaluate():
    pass


if __name__ == "__main__":
    a = get_args()
    gpus = [int(i) for i in a.gpus.split(",")]

    data_cfg = get_config(a.dataset)
    model_cfg = get_config(a.model)
    log_file = "eval.log"
    model_dir = os.path.join(model_cfg["model_dir"], data_cfg["name"])
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(
                                os.path.join(model_dir, log_file)),
                            logging.StreamHandler()
                        ])

    # load models
    cuda = torch.cuda.is_available()
    R = model_cfg['r']
    b = model_cfg['b']
    ori_labels = data_cfg["num_labels"]
    dest_dim = model_cfg['dest_dim']
    name = data_cfg['name']
    record_dir = data_cfg["record_dir"]
    data_dir = os.path.join("data", name)

    # load dataset
    test_file = os.path.join(data_dir, name + "_" + "test.txt")
    test_set = XCDataset(test_file, 0, data_cfg, model_cfg, 'te')

    pred_avg_meter = AverageMeter()
    gt = None

    logging.info("Evaluating config %s" % (a.model))
    for r in tqdm.tqdm(range(R)):
        # load ground truth
        test_set.change_feat_map(r)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=model_cfg['batch_size'])
        model_dir = get_model_dir(data_cfg, model_cfg, r)
        best_param = os.path.join(model_dir, model_cfg["best_file"])

        layers = [dest_dim] + model_cfg['hidden'] + [b]
        model = FCNetwork(layers)

        # load mapping
        label_path = os.path.join(record_dir, "_".join(
            [name, str(ori_labels), str(b), str(R)]))  # Bibtex_159_100_32
        counts, label_mapping, inv_mapping = get_label_hash(label_path, r)
        label_mapping = torch.from_numpy(label_mapping)

        if cuda:
            model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        preload_path = model_cfg["pretrained"] if model_cfg["pretrained"] else best_param
        if os.path.exists(preload_path):
            meta_info = torch.load(preload_path)
            model.load_state_dict(meta_info['model'])
        else:
            raise FileNotFoundError(
                "Model {} does not exist.".format(preload_path))
        # predict
        # gt: original label. p: hashed.
        gt, p, _, _ = compute_scores(model, test_loader)

        # use feature hashing to map back
        pred_avg_meter.update(p[:, label_mapping], 1)
    if gt is None:
        raise Exception("You must have at least one model.")
    else:
        #  Sum of avg is larger than 1 -> that is the feature, no problem
        d = evaluate_scores(gt, pred_avg_meter.avg, model_cfg)
        log_eval_results(d)
