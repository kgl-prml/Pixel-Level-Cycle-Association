#!/bin/bash

export PYTHONPATH="../../:$PYTHONPATH"
if [ $# != 4 ]
then
  echo "Please specify the parameters: 1) gpus; 2) cfg; 3) exp_name; 4) master_port."
  exit 1
fi

gpus=${1}
gpu_array=(`echo ${gpus} | sed "s/,/ /g"`)
num_gpus=${#gpu_array[@]}

cfg=${2}
exp_name=${3}
logpath=./experiments/ckpt/${exp_name}/
#if [ -d ${logpath} ]
#then
#  rm -i -r ${logpath}
#fi

if [ ! -d ${logpath} ]
then
  mkdir -p ${logpath}
fi
master_port=${4}

# optional: --master_port
CUDA_VISIBLE_DEVICES=${gpus} python -m torch.distributed.launch --nproc_per_node=${num_gpus} --use_env --master_port=${master_port} ./tools/train.py --cfg ${cfg} --exp_name ${exp_name} 2>&1 | tee ${logpath}/log.txt
