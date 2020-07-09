#!/usr/bin/env bash
while getopts 'm:d:c:' c; do
    case $c in
    m) MODEL=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    c) COST=$OPTARG ;;
    esac
done
if [ -z $MODEL ] || [ -z $DATASET ]; then
    echo "Need to input model and dataset"
    exit
fi
# Capital initial for DATASET, lower for MODEL
# ./train.sh -m bibtex -d Bibtex_trimcumsum0.1
MODEL_CONFIG=config/model/$MODEL.yaml
DATASET_CONFIG1=config/dataset/$DATASET.yaml
if [ -e $DATASET_CONFIG1 ]; then
    DATASET_CONFIG=$DATASET_CONFIG1
else
    # trimmed
    IFS='_' read -r -a array <<<"$DATASET"
    DATASET_CONFIG2=config/data_trim/${array[0]}/$DATASET.yaml
    if [ -e $DATASET_CONFIG2 ]; then
        DATASET_CONFIG=$DATASET_CONFIG2
    else
        echo "Config files do not exist, $DATASET_CONFIG1 or $DATASET_CONFIG2."
    fi
fi
echo $MODEL_CONFIG
echo $DATASET_CONFIG
python -W ignore::Warning src/preprocess.py --model $MODEL_CONFIG --dataset $DATASET_CONFIG
for ((i = 0; i < 8; i++)); do
   for ((j = 0; j < 4; j++)); do
       part=$(($i * 4 + $j))
       echo $part
       export CUDA_VISIBLE_DEVICES=$j
       if [ -z $COST ]; then
           echo nocost
           python -W ignore::Warning src/train.py --rep $part --model $MODEL_CONFIG \
               --dataset $DATASET_CONFIG --gpus 0 &
       else
           echo heicost
           python -W ignore::Warning src/train.py --rep $part --model $MODEL_CONFIG \
               --dataset $DATASET_CONFIG --gpus 0  --cost $COST &
       fi
   done
   wait
done
if [ -z $COST ]; then
   echo noevalcost
   python -W ignore::Warning src/evaluate.py --model $MODEL_CONFIG --dataset $DATASET_CONFIG
else
   echo eval cost
   python -W ignore::Warning src/evaluate.py --model $MODEL_CONFIG --dataset $DATASET_CONFIG --cost $COST
fi