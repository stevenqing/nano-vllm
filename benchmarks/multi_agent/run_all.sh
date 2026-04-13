#!/bin/bash
# Run all multi-agent benchmarks: nano-vllm vs vLLM
# Usage: bash run_all.sh <model_size> (e.g., 0.6B or 8B)

MODEL_SIZE=${1:-"8B"}
MODEL_PATH="$HOME/huggingface/Qwen3-${MODEL_SIZE}/"
NANO_PORT=8100
VLLM_PORT=8200
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "Multi-Agent Benchmark Suite"
echo "Model: Qwen3-${MODEL_SIZE}"
echo "============================================"
echo ""

# ---- Step 1: Start nano-vllm server ----
echo "[1/6] Starting nano-vllm server on port ${NANO_PORT}..."
CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/nano_api_server.py" \
    --model "${MODEL_PATH}" --port ${NANO_PORT} &
NANO_PID=$!
sleep 30  # Wait for model to load

# Test nano-vllm server
curl -s http://localhost:${NANO_PORT}/health && echo " nano-vllm ready" || { echo "nano-vllm failed"; kill $NANO_PID; exit 1; }

# ---- Step 2: Run benchmarks on nano-vllm ----
echo ""
echo "[2/6] Running MetaGPT pipeline on nano-vllm..."
python "${SCRIPT_DIR}/bench_metagpt.py" --url "http://localhost:${NANO_PORT}/v1" --model "nano-vllm" --label "nano-vllm"

echo ""
echo "[3/6] Running CAMEL role-playing on nano-vllm..."
python "${SCRIPT_DIR}/bench_camel.py" --url "http://localhost:${NANO_PORT}/v1" --model "nano-vllm" --label "nano-vllm"

# Kill nano-vllm
kill $NANO_PID 2>/dev/null
wait $NANO_PID 2>/dev/null
sleep 5

# ---- Step 3: Start vLLM server ----
echo ""
echo "[4/6] Starting vLLM server on port ${VLLM_PORT}..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" --port ${VLLM_PORT} \
    --max-model-len 4096 --enable-prefix-caching \
    --disable-log-stats &
VLLM_PID=$!
sleep 60  # vLLM takes longer to start

curl -s http://localhost:${VLLM_PORT}/health && echo " vLLM ready" || { echo "vLLM failed"; kill $VLLM_PID; exit 1; }

# ---- Step 4: Run benchmarks on vLLM ----
echo ""
echo "[5/6] Running MetaGPT pipeline on vLLM..."
python "${SCRIPT_DIR}/bench_metagpt.py" --url "http://localhost:${VLLM_PORT}/v1" --model "Qwen3-${MODEL_SIZE}" --label "vLLM"

echo ""
echo "[6/6] Running CAMEL role-playing on vLLM..."
python "${SCRIPT_DIR}/bench_camel.py" --url "http://localhost:${VLLM_PORT}/v1" --model "Qwen3-${MODEL_SIZE}" --label "vLLM"

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo ""
echo "============================================"
echo "All benchmarks complete!"
echo "============================================"
