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

for maxconcurrency in 1 2 4 8 16 32; do
    . launch_server.sh &
    wait_for_server

    OUTPUT_FILE="results/default/concur_${maxconcurrency}.csv"
    python3 multi-round-qa.py \
        --sharegpt \
        --num-users 40 \
        --num-rounds 5 \
        --shared-system-prompt 1000 \
        --user-history-prompt 2000 \
        --max-answer-len 256 \
        --min-answer-len 128 \
        --model $MODEL_ID \
        --base-url http://localhost:8000/v1 \
        --max-concurrency $maxconcurrency \
        --output $OUTPUT_FILE

    kill_gpu_processes
    sleep 60
done

for type in cpu disk; do

    for maxconcurrency in 1 2 4 8 16 32; do
        . launch_server.sh --use-lmcache --type=$type &
        wait_for_server

        OUTPUT_FILE="results/lmcache/${type}_concur_${maxconcurrency}.csv"
        python3 multi-round-qa.py \
            --sharegpt \
            --num-users 40 \
            --num-rounds 5 \
            --shared-system-prompt 1000 \
            --user-history-prompt 2000 \
            --max-answer-len 256 \
            --min-answer-len 128 \
            --model $MODEL_ID \
            --base-url http://localhost:8000/v1 \
            --max-concurrency $maxconcurrency \
            --output $OUTPUT_FILE

        rm lmcache_disk/*
        kill_gpu_processes
        sleep 60
    done

done