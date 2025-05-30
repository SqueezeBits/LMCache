#!/bin/bash

kill_gpu_processes() {
  pkill -f python
  pkill -f python3
  pkill -f tritonserver
  pkill -f pt_main_thread
  pkill -f text-generation
  pkill -f lmdeploy

  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)" -ge 1000 ]; do
    sleep 1
  done
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  timeout 1200 bash -c '
    until curl -s localhost:8000/v1/completions > /dev/null; do
      sleep 1
    done' && return 0 || return 1
}

MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"


python3 multi-round-qa-org.py \
    --sharegpt \
    --num-users 20 \
    --num-rounds 5 \
    --shared-system-prompt 1000 \
    --user-history-prompt 2000 \
    --max-answer-len 256 \
    --min-answer-len 0 \
    --model $MODEL_ID \
    --base-url http://localhost:8000/v1 \
    --qps 0.5 \
    --time 30 \
    --output test.csv
  
kill_gpu_processes