#!/bin/bash

set -x
set -o pipefail

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

CONFIG_PATH=$1
OUTPUT_DIR=$2
PRETRAINED_MODEL=$3

# Preparation
mkdir -p ${OUTPUT_DIR}
RUN_NAME=$(basename ${OUTPUT_DIR})


NNODES=${NNODES:=1}
if command -v nvidia-smi &> /dev/null && nvidia-smi --list-gpus &> /dev/null; then
  # GPU
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}
  else
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(nvidia-smi --list-gpus | wc -l)}
  fi
  export NCCL_DEBUG=WARN
else
  # NPU
  if [[ -n "${ASCEND_RT_VISIBLE_DEVICES}" ]]; then
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(echo "${ASCEND_RT_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}
  else
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(ls -l /dev/davinci* | grep -v "davinci_manager" | wc -l)}
  fi
  # NPU env that may optimize performance
  export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:='expandable_segments:True'}
fi
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}

if [[ "$NNODES" == "1" ]]; then
  additional_args="$additional_args --standalone"
else
  additional_args="--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

torchrun \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  $additional_args \
  tasks/omni/train_qwen_vl.py $CONFIG_PATH \
  --train.output_dir ${OUTPUT_DIR} \
  --train.wandb_name ${RUN_NAME} \
  --model.model_path ${PRETRAINED_MODEL} \
   2>&1 | tee ${OUTPUT_DIR}/log.txt


# e.g.
# bash scripts/train.sh configs/qwen3vl/llava-next_packing.yaml output/qwen3vl/qwen3vl-4b_llava-next_packing16384_1x8x1_lr1e-5 pretrained_models/Qwen3-VL-4B-LLaVAOV-Stage1.5-New
