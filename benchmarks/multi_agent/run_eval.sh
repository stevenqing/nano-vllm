#!/bin/bash
# Run vLLM multi-turn benchmark + MoA on both nano-vllm and vLLM servers.
# Usage: bash run_eval.sh [model_path] [gpu_id]
set -e

MODEL=${1:-~/huggingface/Qwen3-0.6B}
GPU=${2:-0}
NANO_PORT=8100
VLLM_PORT=8200
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
VLLM_BENCH_DIR="/home/aiscuser/shuqing/vllm/benchmarks/multi_turn"
MOA_SCRIPT="$SCRIPT_DIR/bench_moa.py"
RESULTS_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$RESULTS_DIR"

MODEL_NAME="$(basename $MODEL)"

eval "$(conda shell.bash hook)"
conda activate nano-vllm

echo "================================================================"
echo "  Multi-Agent Evaluation: nano-vllm vs vLLM"
echo "  Model: $MODEL_NAME  GPU: $GPU"
echo "================================================================"

# ── Helper functions ──
kill_servers() {
    fuser -k ${NANO_PORT}/tcp ${VLLM_PORT}/tcp 2333/tcp 2>/dev/null || true
    sleep 2
}

wait_for_server() {
    local url=$1
    local name=$2
    echo "  Waiting for $name at $url ..."
    for i in $(seq 1 60); do
        if curl -s "$url/health" > /dev/null 2>&1 || curl -s "$url/v1/models" > /dev/null 2>&1; then
            echo "  $name is ready."
            return 0
        fi
        sleep 2
    done
    echo "  ERROR: $name did not start in time."
    return 1
}

# ── 1. Run nano-vllm server ──
echo ""
echo "=== Starting nano-vllm server ==="
kill_servers
CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT_DIR/nano_api_server.py" \
    --model "$MODEL" --port $NANO_PORT --served-model-name "$MODEL_NAME" &
NANO_PID=$!
wait_for_server "http://localhost:$NANO_PORT" "nano-vllm"

# ── 2. vLLM multi-turn benchmark on nano-vllm ──
echo ""
echo "=== vLLM Multi-Turn Benchmark → nano-vllm ==="
cd "$VLLM_BENCH_DIR"
# Download text file for synthetic conversation generation if not present
if [ ! -f pg1184.txt ]; then
    wget -q https://www.gutenberg.org/ebooks/1184.txt.utf-8 -O pg1184.txt 2>/dev/null || echo "skip download"
fi
python benchmark_serving_multi_turn.py \
    --model "$MODEL_NAME" \
    --url "http://localhost:$NANO_PORT" \
    --input-file generate_multi_turn.json \
    --num-clients 1 \
    --max-active-conversations 4 \
    --no-stream \
    --request-timeout-sec 300 \
    2>&1 | tee "$RESULTS_DIR/multiturn_nano.txt"

# ── 3. MoA benchmark on nano-vllm ──
echo ""
echo "=== MoA Benchmark → nano-vllm ==="
cd "$REPO_DIR"
python "$MOA_SCRIPT" \
    --url "http://localhost:$NANO_PORT/v1" \
    --model "$MODEL_NAME" \
    --label "nano-vllm" \
    --layers 2 \
    --agents 3 \
    2>&1 | tee "$RESULTS_DIR/moa_nano.txt"

# ── 4. Stop nano-vllm, start vLLM ──
echo ""
echo "=== Starting vLLM server ==="
kill $NANO_PID 2>/dev/null || true
kill_servers
CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL_NAME" \
    --port $VLLM_PORT \
    --max-model-len 4096 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.9 \
    --disable-log-requests \
    2>&1 &
VLLM_PID=$!
wait_for_server "http://localhost:$VLLM_PORT" "vLLM"

# ── 5. vLLM multi-turn benchmark on vLLM ──
echo ""
echo "=== vLLM Multi-Turn Benchmark → vLLM ==="
cd "$VLLM_BENCH_DIR"
python benchmark_serving_multi_turn.py \
    --model "$MODEL_NAME" \
    --url "http://localhost:$VLLM_PORT" \
    --input-file generate_multi_turn.json \
    --num-clients 1 \
    --max-active-conversations 4 \
    --no-stream \
    --request-timeout-sec 300 \
    2>&1 | tee "$RESULTS_DIR/multiturn_vllm.txt"

# ── 6. MoA benchmark on vLLM ──
echo ""
echo "=== MoA Benchmark → vLLM ==="
cd "$REPO_DIR"
python "$MOA_SCRIPT" \
    --url "http://localhost:$VLLM_PORT/v1" \
    --model "$MODEL_NAME" \
    --label "vllm" \
    --layers 2 \
    --agents 3 \
    2>&1 | tee "$RESULTS_DIR/moa_vllm.txt"

# ── Cleanup ──
kill $VLLM_PID 2>/dev/null || true
kill_servers

echo ""
echo "================================================================"
echo "  All evaluations complete. Results in $RESULTS_DIR/"
echo "================================================================"
