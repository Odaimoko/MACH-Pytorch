#!/usr/bin/env bash
echo $1
python -W ignore::Warning src/preprocess.py --model config/model/$1.yaml --dataset config/dataset/$1.yaml
for ((i = 0; i < 8; i++)); do
    for ((j = 0; j < 4; j++)); do
        part=$(($i * 4 + $j))
        echo $part
        export CUDA_VISIBLE_DEVICES=$j
        python -W ignore::Warning src/train.py --rep $part --model config/model/$1.yaml \
            --dataset config/dataset/$1.yaml --gpus 0 &
    done
    wait
done
python -W ignore::Warning src/evaluate.py --model config/model/$1.yaml --dataset config/dataset/$1.yaml
