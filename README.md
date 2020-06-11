# Pytorch implementation for MACH



MACH (Merged-Average Classiﬁers via Hashing) is a method to reduce time and space cost with extreme classification. [Paper](https://arxiv.org/abs/1910.13830). [Dataset](http://manikvarma.org/downloads/XC/XMLRepository.html).

In short, MACH use hash functions to map `L` labels into `B` buckets. In total there are `R` diﬀerent hash functions, which indicates there are `R` groups of buckets. Then we train on classification model for each group. When predicting, it computes the output of `R` models, each predicting `B` labels. Then it maps `B` labels back to `L` labels. At last the scores in `R` groups are averaged to create the final score for each labels, generating a vector of length `L`.

**Terminologies**: 

- Bucket: A hash function f: N->[0,B) hashes any integer into bucket 0~B-1. 
- Repetition: each of `R` label groups is called a repetition.



## TODO

- [x] Test on CPU.
- [x] Debug Evaluation in xclib.
- [x] Test on GPU.
- [x] Add mAP  to final evaluation.
- [x] Add do not use feature hash.
- [x] Test on large datasets.
- [x] Decide A and B for each dataset.[cannot]
- [x] Tune hyper-parameters.
    - [x] bibtex
    - [x] delicious
    - [ ] mediamill
    - [ ] Eurlex
    - [x] wiki10
    - [ ] Amazon670k 
- [x] Trim label. 
    - [x] Train and Evaluate trimmed datasets code.
    - [x] Document for trimmed datasets.
    - [ ] Does trimming labels break MACH's theoretical guarantee?
    - [x] bibtex
    - [x] delicious
    - [ ] mediamill
    - [ ] Eurlex 
    - [ ] wiki10
    - [ ] Amazon670k 


## Currently supporting ...

- Train a fully connected neural network for multilabel classification, using BCE loss.
- Evaluate  sequentially.

## Prerequisite

For evaluation, [pyxclib](https://github.com/kunaldahiya/pyxclib) is need. Currently it has a bug which cause precision to be incorrectly calculated. See this [issue](https://github.com/kunaldahiya/pyxclib/issues/5) and change the code.

> Change `indices` to `indices[:, ::-1]` in [_get_top_k](https://github.com/kunaldahiya/pyxclib/blob/e1100a4013ad9edbfde1524093916a4d870d9a3e/xclib/evaluation/xc_metrics.py#L117).

For calculate mAP (mean average precision), torchnet is used. It has a bug ([issue](https://github.com/pytorch/tnt/issues/134)). Please change `return 0` [here](https://github.com/pytorch/tnt/blob/013e9fed1bebb6b0e9ef89fd47e5edd017c60cfe/torchnet/meter/apmeter.py#L109) to `return torch.tensor([0.])`.

Other prerequisites are specified in `requirements.txt`.



## Usage

Please run code from the project root directory, not from `src`.

### Data preparation

The structure of `data` directory is as follows:

```
data/
├── Delicious -> ../../dataset_uncompressed/Delicious
│   ├── Delicious_test.txt
│   └── Delicious_train.txt
└── bibtex -> ../../dataset_uncompressed/Bibtex/
    ├── bibtex_train.txt
    └── bibtex_test.txt
```

Note: The data directory\'s name should be the same as the `name` field in the config file, and should be case-sensitive. Eg. for *Delicious* dataset, the `name` field in `config/data/delicious.yaml`, the name of the subdirectory under `data` both should be *Delicious*. Additionally, the prefix of `txt` files in the subdirectory can be anything, and should be specified in dataset config files. The training file under `bibtex` directory can be `Delicious_train.txt`, as long as the `prefix` field in the config file is `Delicious`.

`*_test.txt` contains the test dataset and `*_train.txt` contains both training and validation datasets. The number of training instances are specified by `train_size` in data config file. Assume it is `m` . The first `m` training instances in `*_train.txt` are used in training, with the rest being validation set.

The format of the text files can be found [here](http://manikvarma.org/downloads/XC/XMLRepository.html)  (search `README`). The first line in the original dataset contains 3 numbers indicating the **number of instances, features and labels**. These should also be included in the dataset config manually. 

### Configuration

Configuration files are placed in `config/data` (dataset meta-information) and `config/models` (model hyperparameters). There is one file called `test.yaml` in each directory with comments, serving as an example.

The initial of config files should be lower-case.

### Preprocess

After setting up the config files,  preprocess to generate auxiliary files in `record` directory.

```bash
python src/preprocess.py -h
usage: preprocess.py [-h] --model MODEL --dataset DATASET

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Path to the model config yaml file.
  --dataset DATASET, -d DATASET
                        Path to the data config yaml file.
```



### Train

```bash
python src/train.py -h
usage: train.py [-h] [--rep REP] --model MODEL --dataset DATASET [--gpus GPUS]

optional arguments:
  -h, --help            show this help message and exit
  --rep REP, -r REP     Which reptition to train. Default 0.
  --model MODEL, -m MODEL
                        Path to the model config yaml file.
  --dataset DATASET, -d DATASET
                        Path to the data config yaml file.
  --gpus GPUS, -g GPUS  A string that specifies which GPU you want to use,
                        split by comma. Eg 0,1. Default 0.
```

Example: `python src/train.py --rep 0 --model $MODEL_CONFIG $DATASET_CONFIG --gpus 0`

Training script trains model for only one repetition. It creates a `models` directory in which trained models are saved.  The structure of the directory is 

```
models/
└── bibtex
    ├── B_100_R_32_feat_1000_hidden_[32, 32]_rep_00
    │   ├── best_ckpt.pkl
    │   ├── final_ckpt.pkl
    │   └── train.log
    ├── B_100_R_32_feat_1000_hidden_[32, 32]_rep_01
    │   ├── best_ckpt.pkl
    │   ├── final_ckpt.pkl
    │   └── train.log
    ...
    └── eval.log
```




### Evaluation



```bash
python src/evaluate.py -h
usage: evaluate.py [-h] --model MODEL --dataset DATASET [--gpus GPUS]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Path to the model config yaml file.
  --dataset DATASET, -d DATASET
                        Path to the data config yaml file.
  --gpus GPUS, -g GPUS  A string that specifies which GPU you want to use,
                        split by comma. Eg 0,1
```

After training all `R` repetitions, running `evaluate.py` provides the following metrics: Precision, nDCG, PSPrecision, PSnDCG, which are described in [XMLrepo](http://manikvarma.org/downloads/XC/XMLRepository.html). It also logs them into `models/[dataset]/eval.log` (see above).

### Trim labels

```bash
python src/trim_labels.py -h
usage: trim_labels.py [-h] --dataset DATASET --type TYPE

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        Dataset name. Initial should be CAPITAL.
  --type TYPE, -t TYPE  Should be 'cumsum' or 'rank'.
```

This script reads in config files in `config/dataset/`, creates 9 more config files under `config/data_trim/$DATASET/`, and creates 9 more dataset text files in `data/$DATASET/`, as follows. They are datasets which only retain the major instances and labels.

```
data_trim/
├── Bibtex
│   ├── Bibtex_trimcumsum0.1.yaml
│   ├── Bibtex_trimcumsum0.2.yaml
│   ├── ...
│   ├── Bibtex_trimrank0.1.yaml
│   ├── Bibtex_trimrank0.2.yaml
│   ├── ...

data/Bibtex/
├── Bibtex_trimcumsum0.1_meta.json
├── Bibtex_trimcumsum0.1_train.txt
├── Bibtex_trimcumsum0.2_meta.json
├── Bibtex_trimcumsum0.2_train.txt
├── ...
```

There are two modes for trimming off tail labels: `cumsum` and `rank`. Let the ratio of labels to be cut off `r`, and the number of instances `N` the number of total labels `L`. In `rank` mode, `r*L` labels with the fewest instances will be cut. In `cumsum` mode, the fewest labels whose numbers add up to `r*N` will be cut.

After running trimming script, train and evaluate models as usual using these config files. 