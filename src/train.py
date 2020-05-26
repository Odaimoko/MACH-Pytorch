from argparse import ArgumentParser
from mach_utils import get_config, create_record_dir, mkdir, get_loader, evaluate
import os
from fc_network import FCNetwork
import torch
import tqdm

def get_args():
    p = ArgumentParser()
    p.add_argument("--rep", dest="rep", type=int, default=0,
                   help="Which reptition to train")
    p.add_argument("--model", dest="model", type=str, required=True)
    p.add_argument("--dataset", dest="dataset", type=str, required=True)
    p.add_argument("--gpu", dest="gpu", type=str, required=False, default="0",
                   help="A string that specifies which GPU you want to use, split by comma. Eg 0,1")
    return p.parse_args()


def train(data_cfg, model_cfg, rep, gpus, train_loader, val_loader):
    """
        Train one division
    """
    cuda = torch.cuda.is_available()
    dest_dim = model_cfg['dest_dim']
    b = model_cfg['b']
    R = model_cfg['r']
    model_dir = os.path.join("models", "_".join([
        data_cfg["name"], str(b), str(R), str(dest_dim), rep
    ]))  # each repetition has its own dir
    mkdir(model_dir)
    latest_param = os.path.join(model_dir, "final_ckpt.pkl")
    best_param = os.path.join(model_dir, "best_ckpt.pkl")
    # build model and optimizers
    layers = [data_cfg['ori_dim']]+model_cfg['hidden']+[model_cfg['dest_dim']]
    model = FCNetwork(layers)
    if cuda:
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=model_cfg['lr'])
    lr_sch = torch.optim.lr_scheduler.MultiStepLR(
        opt, model_cfg['lr_step'], model_cfg['lr_factor'])
    loss_func = torch.nn.CrossEntropyLoss()
    # load pretrained parameters/checkpoint
    begin = model_cfg['begin_epoch']
    preload_path = model_cfg["pretrained"] if model_cfg["pretrained"] else latest_param
    if os.path.exists(preload_path):
        meta_info = torch.load(preload_path)
        model.load_state_dict(meta_info['model'])
        opt.load_state_dict(meta_info['opt'])
        lr_sch.load_state_dict(meta_info['lr_sch'])
        begin = meta_info['epoch']
    end = model_cfg['end_epoch']
    # train
    for ep in range(begin, end):
        model.train()
        for i, sample in enumerate(train_loader):
            X, y = sample
            if cuda:
                X = X.cuda()
                y = y.cuda()
            opt.zero_grad()
            out = model(X)
            loss = loss_func(out, y)
            loss.backward()
            opt.step()
            lr_sch.step()
        # evaluate on val set
        evaluate()


if __name__ == "__main__":
    a = get_args()
    data_cfg = get_config(a.dataset)
    create_record_dir(data_cfg)
    # load dataset
    gpus = [int(i) for i in a.gpus.split(",")]
    train_loader, val_loader, test_loader = get_loader(data_cfg)
