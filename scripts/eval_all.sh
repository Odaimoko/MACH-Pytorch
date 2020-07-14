#!/usr/bin/env bash

for name in bibtex wiki10-paper delicious eurlex; do
    echo $name
    echo $name
    echo $name
    echo =====================
    python -W ignore::Warning src/evaluate.py -m config/model/$name.yaml -d config/dataset/$name.yaml -c hashed -t all,trim_eval -r 0.1,.2,.3,.4,.5,.6,.7,.8,.9 
    python -W ignore::Warning src/evaluate.py -m config/model/$name.yaml -d config/dataset/$name.yaml -t all,trim_eval -r 0.1,.2,.3,.4,.5,.6,.7,.8,.9 
    echo =====================
done
