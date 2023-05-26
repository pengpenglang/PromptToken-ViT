#!/bin/bash

# 检查是否提供了要执行的Python脚本名称
if [ -z "$1" ]; then
  echo "Need to provide the name of the Python script to execute."
  exit 1
fi

# 设置要使用的GPU设备
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 将所有参数作为字符串拼接到Python命令的末尾
PYTHON_ARGS="${@:2}"

# 执行Python脚本
python -m torch.distributed.run --nproc_per_node=4 --master_port=25642 "$1" $PYTHON_ARGS