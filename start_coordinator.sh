#!/bin/bash
echo "=== OLMoE Coordinator ==="
source venv/bin/activate
python -m uvicorn coordinator.main:app --host 0.0.0.0 --port 8000