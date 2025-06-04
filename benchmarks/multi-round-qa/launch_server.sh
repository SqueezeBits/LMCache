#!/bin/bash

MODEL_ID="Qwen/Qwen3-14B"
KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}'

USE_LMCACHE=false
STORAGE_TYPE=""
LMCACHE_ENVS=""
KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}'

for arg in "$@"; do
    if [[ "$arg" == "--use-lmcache" ]]; then
        USE_LMCACHE=true
        LMCACHE_ENVS="$LMCACHE_ENVS LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CHUNK_SIZE=256"
        set -- "${@/--use-lmcache/}"
    elif [[ "$arg" == --type=* ]]; then
        STORAGE_TYPE="${arg#--type=}"
        set -- "${@/$arg/}"
    fi
done

if [[ "$STORAGE_TYPE" == "cpu" ]]; then
    LMCACHE_ENVS="$LMCACHE_ENVS LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=60.0"
elif [[ "$STORAGE_TYPE" == "cxl" ]]; then
    LMCACHE_ENVS="$LMCACHE_ENVS LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=120.0"
elif [[ "$STORAGE_TYPE" == "disk" ]]; then
    DISK_PATH="lmcache_disk/"
    LMCACHE_ENVS="$LMCACHE_ENVS LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=60.0 LMCACHE_MAX_LOCAL_DISK_SIZE=60.0 LMCACHE_LOCAL_DISK=$DISK_PATH"
fi


CMD="VLLM_USE_V1=0 \
python -m vllm.entrypoints.openai.api_server \
--model $MODEL_ID \
--tokenizer $MODEL_ID \
--disable-log-requests \
--uvicorn-log-level warning"

if [ "$USE_LMCACHE" = true ]; then
    CMD="$LMCACHE_ENVS $CMD --kv-transfer-config '$KV_TRANSFER_CONFIG'"
fi

echo $CMD
eval $CMD