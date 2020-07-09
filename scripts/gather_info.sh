#!/usr/bin/env bash
echo $1
grep $1 -e "prec"
grep $1 -e "ndcg"
grep $1 -e "psp"
grep $1 -e "psn"
grep $1 -e "mAP"
grep $1 -e "AP of"