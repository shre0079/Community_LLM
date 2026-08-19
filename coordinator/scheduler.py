from typing import Tuple
from coordinator.registry import WorkerRegistry

# Fixed layer assignments — adjust if VRAM differs across machines
LAYER_RANGES = [
    (0,  5,  True,  False),  # 6 layers + embedding
    (6,  10, False, False),  # 5 layers
    (11, 15, False, True),   # 5 layers + norm + lm_head
]


class Scheduler:
    def __init__(self, registry: WorkerRegistry):
        self.registry = registry

    def assign(self, worker_id: str) -> Tuple[int, int, bool, bool]:
        used = {
            (w.layer_start, w.layer_end)
            for w in self.registry.all()
            if w.layer_start is not None and w.worker_id != worker_id
        }
        for start, end, is_first, is_last in LAYER_RANGES:
            if (start, end) not in used:
                w = self.registry.get(worker_id)
                w.layer_start = start
                w.layer_end = end
                w.is_first = is_first
                w.is_last = is_last
                return start, end, is_first, is_last
        raise ValueError("All shards are already assigned.")