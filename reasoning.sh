#!/bin/bash
export CUDA_VISIBLE_DEVICES="1,2,3,4"

# 设置主节点地址和端口
export MASTER_ADDR="localhost"  # 你也可以设置为主节点的 IP 地址
export MASTER_PORT="29500"      # 设置一个不冲突的端口

# 设置世界规模（总进程数，例如两个 GPU 的话是 2）
export WORLD_SIZE=4

# 在分布式训练中，RANK 是每个进程的唯一标识，范围是 0 到 WORLD_SIZE-1
# 如果你有两个进程，可以分别运行下面两个命令：
export RANK=0  # 对于进程 0
# 或者对于进程 1，运行另一个脚本：
# export RANK=1

# 启动分布式训练脚本
# python evaluate/ddp_reasoning.py
# python test.py
python3 -m vllm.entrypoints.openai.api_server --model checkpoints/qwen-72/ --served-model-name Qwen-72B --tensor-parallel-size 4 --gpu-memory-utilization 0.9
