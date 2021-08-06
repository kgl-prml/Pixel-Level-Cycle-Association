#!/bin/bash

export PYTHONPATH="../../:$PYTHONPATH"
if [ $# != 5 ]
then
  echo "Please specify the parameters: 1) gpus; 2) cfg; 3) weights; 4) exp_name; 5) master_port."
  exit 1
fi

gpus=${1}
gpu_array=(`echo ${gpus} | sed "s/,/ /g"`)
num_gpus=${#gpu_array[@]}

cfg=${2}
weights=${3}
exp_name=${4}
master_port=${5}

logpath=./experiments/ckpt/${exp_name}/
#if [ -d ${logpath} ]
#then
#  rm -i -r ${logpath}
#fi

if [ ! -d ${logpath} ]
then
  mkdir -p ${logpath}
fi

# optional: --master_port
CUDA_VISIBLE_DEVICES=${gpus} python -m torch.distributed.launch --nproc_per_node=${num_gpus} --use_env --master_port=${master_port} \
        ./tools/test.py --cfg ${cfg} --weights ${weights} --exp_name ${exp_name} 2>&1 | tee ${logpath}/log.txt

