"""
Pre-download OLMoE-1B-7B to local cache.
Run this once on every machine before starting workers.
The model is ~14 GB in float16; bitsandbytes quantizes it to ~3.5 GB in memory.
"""
from huggingface_hub import snapshot_download
from shared.config import MODEL_ID, MODEL_CACHE_DIR

print(f"Downloading {MODEL_ID} → {MODEL_CACHE_DIR}")
print("Expected size: ~14 GB.  This runs once and is cached.")

snapshot_download(
    repo_id=MODEL_ID,
    cache_dir=MODEL_CACHE_DIR,
    ignore_patterns=["*.gguf"],   # skip GGUF variants if any
)

print("\nDownload complete. You can now start workers.")