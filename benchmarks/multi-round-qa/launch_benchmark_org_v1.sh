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

MODEL_ID="Qwen/Qwen3-14B"

for type in cpu cxl disk; do

    for qps in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0; do
        . launch_server_v1.sh --use-lmcache --type=$type &
        wait_for_server

        OUTPUT_FILE="results/org/lmcache_u160_r5_m256_vllm_v1/${type}_qps_${qps}.csv"
        python3 multi-round-qa-org.py \
            --sharegpt \
            --num-users 160 \
            --num-rounds 5 \
            --shared-system-prompt 1000 \
            --user-history-prompt 2000 \
            --max-answer-len 256 \
            --model $MODEL_ID \
            --base-url http://localhost:8000/v1 \
            --qps $qps \
            --output $OUTPUT_FILE \
            --time 900

        rm lmcache_disk/*
        kill_gpu_processes
        sleep 60
    done

done

