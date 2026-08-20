# test_shard.py  — run with: python test_shard.py
from worker.model import ShardRunner
import torch

runner = ShardRunner(layer_start=0, layer_end=5, is_first=True, is_last=False)
runner.load()

ids = runner.encode("Hello, world!")
out = runner.forward(ids)
print("Output shape:", out.shape)  # should be [1, seq_len, 2048]
print("VRAM used:", runner.vram_used_gb(), "GB")