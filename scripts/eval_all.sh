#!/usr/bin/env bash

for name in bibtex wiki10-paper delicious eurlex; do
    python src/evaluate.py -m config/model/$name.yaml -d config/dataset/$name.yaml -c hashed -type all,trim_eval -r 0.1,.2,.3,.4,.5,.6,.7,.8,.9 >logs/${name}_hashed.log
    python src/evaluate.py -m config/model/$name.yaml -d config/dataset/$name.yaml -type all,trim_eval -r 0.1,.2,.3,.4,.5,.6,.7,.8,.9 >logs/${name}.log
done
