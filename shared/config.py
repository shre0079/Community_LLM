import os

COORDINATOR_HOST = os.getenv("COORDINATOR_HOST", "localhost")
COORDINATOR_PORT = int(os.getenv("COORDINATOR_PORT", "8000"))
COORDINATOR_URL = f"http://{COORDINATOR_HOST}:{COORDINATOR_PORT}"

MODEL_ID = "allenai/OLMoE-1B-7B"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")

TOTAL_LAYERS = 16
HIDDEN_DIM = 2048

HEARTBEAT_INTERVAL = 5       # seconds between heartbeats
HEARTBEAT_TIMEOUT = 20       # seconds before coordinator marks worker dead

WORKER_HTTP_PORT = int(os.getenv("WORKER_HTTP_PORT", "8001"))
ZMQ_IN_PORT = int(os.getenv("ZMQ_IN_PORT", "5555"))       # hidden states in
ZMQ_RESULT_PORT = int(os.getenv("ZMQ_RESULT_PORT", "5556")) # tokens back (Worker 1 only)