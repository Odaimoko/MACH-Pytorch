#!/usr/bin/env bash
#model=bibtex
#data=Bibtex
#type=rank
while getopts 'm:d:t:' c; do
    case $c in
    m) model=$OPTARG ;;
    d) data=$OPTARG ;;
    t) type=$OPTARG ;;
    esac
done

python -W ignore::Warning src/tune.py -d $data