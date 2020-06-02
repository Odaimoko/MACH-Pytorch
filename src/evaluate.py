from mach_utils import *
import logging
from argparse import ArgumentParser
from fc_network import FCNetwork


def get_args():
    p = ArgumentParser()
    p.add_argument("--model", dest = "model", type = str, required = True)
    p.add_argument("--dataset", dest = "dataset", type = str, required = True)
    
    p.add_argument("--gpus", dest = "gpus", type = str, required = False, default = "0",
                   help = "A string that specifies which GPU you want to use, split by comma. Eg 0,1")
    return p.parse_args()


def evaluate():
    pass


if __name__ == "__main__":
    a = get_args()
    gpus = [int(i) for i in a.gpus.split(",")]
    
    data_cfg = get_config(a.dataset)
    model_cfg = get_config(a.model)
    log_file = "eval.log"
    model_dir = os.path.join(model_cfg["model_dir"], data_cfg["name"])
    logging.basicConfig(filename = os.path.join(model_dir, log_file), level = logging.INFO)
    
    # load models
    cuda = torch.cuda.is_available()
    R = model_cfg['r']
    pred = []
    gt = None
    name = data_cfg['name']
    
    data_dir = os.path.join("data", name)
    
    test_file = name + "_" + "test.txt"
    
    test_set = XCDataset(test_file, 0, data_cfg, model_cfg, 'te')
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size = model_cfg['batch_size'])
    for r in range(R):
        model_dir = get_model_dir(data_cfg, model_cfg, r)
        best_param = os.path.join(model_dir, model_cfg["best_file"])
        
        layers = [dest_dim] + model_cfg['hidden'] + [b]
        model = FCNetwork(layers)
        
        if cuda:
            model = torch.nn.DataParallel(model, device_ids = gpus).cuda()
        preload_path = model_cfg["pretrained"] if model_cfg["pretrained"] else best_param
        if os.path.exists(preload_path):
            meta_info = torch.load(preload_path)
            model.load_state_dict(meta_info['model'])
        else:
            raise FileNotFoundError("Model {} does not exist.".format(preload_path))
        gt, p, _, _ = compute_scores(model, test_loader)
        
        # use feature hashing to map back
        pred.append(p)  # each = num_instances x num_labels
    if gt is None:
        raise Exception("You must have at least one model.")
    else:
        pred = np.stack(pred)  # R x num_ins x num_lab
        scores = pred.mean(axis = 0)
        d = evaluate_scores(gt, scores, model_cfg)
    
    # load dataset
    
    # load ground truth
    
    # predict
    
    # evaluate using PyXCLib
