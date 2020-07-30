#!/usr/bin/env bash
while getopts 'm:d:c:a:' c; do
    case $c in
    m) MODEL=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    c) COST=$OPTARG ;;
    a) APPROX=$OPTARG ;;
    esac
done
MODEL_CONFIG=config/model/$MODEL.yaml
DATASET_CONFIG=config/dataset/$DATASET.yaml

echo $MODEL_CONFIG
echo $DATASET_CONFIG

for ((i = 0; i < 8; i++)); do
    for ((j = 0; j < 4; j++)); do
        part=$(($i * 4 + $j))
        echo $part
        export CUDA_VISIBLE_DEVICES=$j
        if [ -z $COST ]; then
            echo nocost
            python -W ignore::Warning src/run_single_n_save.py --rep $part --model $MODEL_CONFIG \
                --dataset $DATASET_CONFIG --gpus 0 -a $APPROX
        else
            echo heicost
            python -W ignore::Warning src/run_single_n_save.py --rep $part --model $MODEL_CONFIG \
                --dataset $DATASET_CONFIG --gpus 0 --cost $COST -a $APPROX
        fi
    done
    wait
done

if [ -z $COST ]; then
    echo noevalcost python -W ignore::Warning src/eval_map_ensemble.py --model $MODEL_CONFIG --dataset $DATASET_CONFIG
    python -W ignore::Warning src/eval_map_ensemble.py --model $MODEL_CONFIG --dataset $DATASET_CONFIG
else
    echo evalcost python -W ignore::Warning src/eval_map_ensemble.py --model $MODEL_CONFIG --dataset $DATASET_CONFIG --cost $COST
    python -W ignore::Warning src/eval_map_ensemble.py --model $MODEL_CONFIG --dataset $DATASET_CONFIG --cost $COST 
fi
