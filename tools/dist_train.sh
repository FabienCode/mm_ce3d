#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-9610}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --work-dir log/vote_label_gtpara_fploss_1.11_v1
