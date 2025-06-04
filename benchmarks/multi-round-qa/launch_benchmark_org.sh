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

for qps in 0.5 1.0 1.5 2.0; do
    . launch_server.sh &
    wait_for_server

    OUTPUT_FILE="results/org/default/qps_${qps}.csv"
    python3 multi-round-qa-org.py \
        --sharegpt \
        --num-users 40 \
        --num-rounds 16 \
        --shared-system-prompt 1000 \
        --user-history-prompt 2000 \
        --max-answer-len 512 \
        --min-answer-len 256 \
        --model $MODEL_ID \
        --base-url http://localhost:8000/v1 \
        --qps $qps \
        --output $OUTPUT_FILE \
        --time 300

    kill_gpu_processes
    sleep 60
done

for type in cpu cxl disk; do

    for qps in 0.5 1.0 1.5 2.0; do
        . launch_server.sh --use-lmcache --type=$type &
        wait_for_server

        OUTPUT_FILE="results/org/lmcache/${type}_qps_${qps}.csv"
        python3 multi-round-qa-org.py \
            --sharegpt \
            --num-users 40 \
            --num-rounds 16 \
            --shared-system-prompt 1000 \
            --user-history-prompt 2000 \
            --max-answer-len 512 \
            --min-answer-len 256 \
            --model $MODEL_ID \
            --base-url http://localhost:8000/v1 \
            --qps $qps \
            --output $OUTPUT_FILE \
            --time 300 \
            --use-full-dialogue

        rm lmcache_disk/*
        kill_gpu_processes
        sleep 60
    done

done