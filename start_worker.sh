#!/bin/bash
echo "=== OLMoE Worker ==="
read -rp "Coordinator IP (e.g. 192.168.1.100): " COORDINATOR_HOST
export COORDINATOR_HOST
export COORDINATOR_PORT=8000
export WORKER_HTTP_PORT=8001
export ZMQ_IN_PORT=5555
export ZMQ_RESULT_PORT=5556
export ZMQ_RECV_TIMEOUT_MS=30000
export SIMULATE_VRAM_GB=4.0
# export MY_IP=192.168.1.X  # uncomment if auto-detection fails
source venv/bin/activate
python -m worker.main