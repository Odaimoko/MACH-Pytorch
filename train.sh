#!/usr/bin/env bash
for ((i = 0; i < 8; i++)); do
    for ((j = 0; j < 4; j++)); do
        part=$(($i * 4 + $j))
        echo $part
        export CUDA_VISIBLE_DEVICES=$j
        python -W ignore::Warning src/train.py --rep $part --model config/model/wiki10.yaml \
            --dataset config/dataset/wiki10.yaml --gpus $j &
    done
    wait

done
