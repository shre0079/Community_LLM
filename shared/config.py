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