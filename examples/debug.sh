#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

python -m verl.trainer.main \
  config=examples/trajectory_config.yaml \
  trainer.n_gpus_per_node=1 \
  worker.rollout.tensor_parallel_size=1 \
  trainer.max_steps=1 \
  worker.rollout.n=2 \
  worker.actor.micro_batch_size_per_device_for_experience=2 \
  worker.actor.micro_batch_size_per_device_for_update=2 \
  worker.actor.fsdp.enable_full_shard=false \
  worker.actor.offload.offload_params=false \
  worker.actor.offload.offload_optimizer=false
