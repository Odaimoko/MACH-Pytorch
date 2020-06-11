from argparse import ArgumentParser
from mach_utils import get_config, create_record_dir, mkdir, evaluate_single, get_model_dir, get_label_hash, \
    get_mapped_labels, log_eval_results
import os
import multiprocessing
import pathlib
import yaml
import subprocess


def get_args():
    p = ArgumentParser()
    p.add_argument("--dataset", '-d', dest="dataset", type=str, required=True,
                   help="Dataset name. Initial should be lower-case.")
    p.add_argument("--process", '-p', dest="processes", type=int, required=False, default=4,
                   help="Number of processes to run in parallel")
    return p.parse_args()


def create_config(model_cfg, dir_path):
    p = pathlib.PurePosixPath(dir_path)
    bs = [500, 1000, 2000]
    lrs = [.1, .01, .001]
    hiddens = [
        [32, 32], [128, 128], [512, 512],
        [128, 128, 128], [512, 512, 512]
    ]
    d = model_cfg.copy()
    for lr in lrs:
        for h in hiddens:
            for b in bs:
                d['b'] = b
                d['lr'] = lr
                d['hidden'] = h
                file_name = "B_%d_lr_%s-hidden_%s.yaml" % (
                    b, str(lr), str(h[0]) + "_" + str(len(h)))
                with open(os.path.join(dir_path, file_name), 'w') as f:
                    yaml.dump(d, f)


if __name__ == "__main__":
    py = subprocess.check_output(['which', 'python']).decode().strip()
    a = get_args()
    dataset = os.path.join("config/dataset", a.dataset + ".yaml")
    model = os.path.join("config/model", a.dataset + ".yaml")
    model_cfg = get_config(model)
    config_temp = os.path.join("config/temp/", a.dataset)
    mkdir(config_temp)
    create_config(model_cfg, dir_path=config_temp)
    R = model_cfg['r']

    cli_args = "--model %s --dataset %s" % (model, dataset)
    for c in sorted(os.listdir(config_temp)):
        current_model = os.path.join(config_temp, c)
        cli_args = "--model %s --dataset %s" % (current_model, dataset)
        os.system(py + " src/preprocess.py " + cli_args)

        cmds = []
        k = 0
        for r in range(R):
            cmd = "export CUDA_VISIBLE_DEVICES=%d; %s -W ignore::Warning src/train.py %s --rep %d --gpus 0" % (
                k, py, cli_args, r)
            k = (k+1) % 4
            cmds.append(cmd)
        with multiprocessing.pool.Pool(processes=a.processes) as p:
            print("Running... %s" % current_model, cmd)
            p.imap(os.system, cmds)
            p.close()
            p.join()
        os.system(py+" src/evaluate.py " + cli_args)
