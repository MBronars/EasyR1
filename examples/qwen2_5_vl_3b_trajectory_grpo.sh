#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
python -m verl.trainer.main \
  config=examples/trajectory_config.yaml \
  trainer.n_gpus_per_node=2 \
  trainer.max_steps=6000 \
  trainer.val_freq=-1 \
  trainer.save_freq=500 \
  data.rollout_batch_size=64 \
  worker.actor.global_batch_size=64 \
  worker.rollout.n=5 \
  worker.actor.micro_batch_size_per_device_for_experience=16 \
  worker.actor.micro_batch_size_per_device_for_update=16 

