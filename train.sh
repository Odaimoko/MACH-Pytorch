#!/usr/bin/env bash
for ((i = 0; i < 8; i++)); do
    for ((j = 0; j < 4; j++)); do
        part=$(($i*4+$j))
        echo $part
         python -W ignore::Warning src/train.py --rep $part --model config/model/test.yaml \
            --dataset config/dataset/test.yaml --gpu 0 &
    done
    wait

done
