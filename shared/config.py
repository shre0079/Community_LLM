import os

COORDINATOR_HOST = os.getenv("COORDINATOR_HOST", "localhost")
COORDINATOR_PORT = int(os.getenv("COORDINATOR_PORT", "8000"))
COORDINATOR_URL  = f"http://{COORDINATOR_HOST}:{COORDINATOR_PORT}"

MODEL_ID        = "allenai/OLMoE-1B-7B-0125"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")

TOTAL_LAYERS = 16

HEARTBEAT_INTERVAL = 5    # seconds between heartbeats
HEARTBEAT_TIMEOUT  = 20   # seconds before coordinator marks worker dead

WORKER_HTTP_PORT    = int(os.getenv("WORKER_HTTP_PORT",    "8001"))
ZMQ_IN_PORT         = int(os.getenv("ZMQ_IN_PORT",         "5555"))  # workers 2,3 bind here
ZMQ_RESULT_PORT     = int(os.getenv("ZMQ_RESULT_PORT",     "5556"))  # worker 1 binds here
ZMQ_RECV_TIMEOUT_MS = int(os.getenv("ZMQ_RECV_TIMEOUT_MS", "30000")) # 30 s

# Set to 4.0 to cap VRAM on high-VRAM cards (e.g. A6000 → simulate 4 GB).
# 0 = no cap.
SIMULATE_VRAM_GB = float(os.getenv("SIMULATE_VRAM_GB", "0"))

# ── Single-node-first placement (Section 4 of the design doc) ─────────────────
# Rough, intentionally crude feasibility threshold used ONLY to decide whether
# a single registering worker can serve the whole model with zero inter-node
# hops. This is NOT the Phase 3 greedy_pack profiler — it doesn't account for
# fragmentation, per-node runtime overhead variance, or concurrent requests.
# Treat it as a configurable heuristic, not a guarantee (same caveat the
# design doc applies to the 25% KV reserve) — Phase 3 should replace this
# with real per-node capacity accounting.
EST_MODEL_VRAM_GB    = float(os.getenv("EST_MODEL_VRAM_GB", "6.0"))  # 4-bit NF4 weights + CUDA/bnb overhead
EST_KV_ACTIVATION_GB = float(os.getenv("EST_KV_ACTIVATION_GB", "2.0"))  # KV-cache + activations headroom
SINGLE_NODE_VRAM_GB  = EST_MODEL_VRAM_GB + EST_KV_ACTIVATION_GB