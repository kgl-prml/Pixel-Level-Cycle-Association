#!/bin/bash

export PYTHONPATH="../../:$PYTHONPATH"
if [ $# != 4 ]
then
  echo "Please specify the parameters: 1) gpus; 2) cfg; 3) weights; 4) exp_name."
  exit 1
fi

gpus=${1}
gpu_array=(`echo ${gpus} | sed "s/,/ /g"`)
num_gpus=${#gpu_array[@]}

cfg=${2}
weights=${3}
exp_name=${4}

logpath=./experiments/ckpt/${exp_name}/
#if [ -d ${logpath} ]
#then
#  rm -i -r ${logpath}
#fi

if [ ! -d ${logpath} ]
then
  mkdir -p ${logpath}
fi

CUDA_VISIBLE_DEVICES=${gpus} python \
        ./tools/test.py --cfg ${cfg} --weights ${weights} --exp_name ${exp_name} 2>&1 | tee ${logpath}/log.txt

