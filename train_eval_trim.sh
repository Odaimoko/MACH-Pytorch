#!/usr/bin/env bash
model=bibtex
data=Bibtex
for (( j = 1; j < 10; ++j)); do
    MODEL=$model
    DATA=${data}_trimcumsum0.$j
    ./train.sh -m $MODEL -d $DATA
done
