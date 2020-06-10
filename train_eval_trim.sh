#!/usr/bin/env bash
#model=bibtex
#data=Bibtex
#type=rank
while getopts 'm:d:' c; do
    case $c in
    m) model=$OPTARG ;;
    d) data=$OPTARG ;;
    t) type=$OPTARG ;;
    esac
done

for (( j = 1; j < 10; ++j)); do
    MODEL=$model
    DATA=${data}_trimrank0.$j
    ./train.sh -m $MODEL -d $DATA
done
