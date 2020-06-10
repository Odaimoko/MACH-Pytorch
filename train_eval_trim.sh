#!/usr/bin/env bash
model=bibtex
data=Bibtex
type=rank
for (( j = 1; j < 10; ++j)); do
    MODEL=$model
    DATA=${data}_trimrank0.$j
    ./train.sh -m $MODEL -d $DATA
done
