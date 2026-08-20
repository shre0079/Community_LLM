# vram_monitor.py
import time
import torch

while True:
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {used:.2f} / {total:.2f} GB ({100*used/total:.1f}%)")
    time.sleep(2)