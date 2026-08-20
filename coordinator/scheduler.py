from coordinator.registry import WorkerRegistry
from shared.config import ZMQ_IN_PORT, ZMQ_RESULT_PORT

# (layer_start, layer_end, is_first, is_last)
LAYER_RANGES = [
    (0,   5, True,  False),   # 6 layers — embedding lives here
    (6,  10, False, False),   # 5 layers
    (11, 15, False, True),    # 5 layers — norm + lm_head live here
]


class Scheduler:
    def __init__(self, registry: WorkerRegistry):
        self.registry = registry

    def assign(self, worker_id: str) -> dict:
        """
        Find the next unclaimed shard, stamp it onto the worker, and
        return a complete assignment dict including the ZMQ port the
        coordinator decided this worker should bind on.
        """
        occupied = {
            (w.layer_start, w.layer_end)
            for w in self.registry.all()
            if w.layer_start is not None and w.worker_id != worker_id
        }

        for start, end, is_first, is_last in LAYER_RANGES:
            if (start, end) not in occupied:
                w             = self.registry.get(worker_id)
                w.layer_start = start
                w.layer_end   = end
                w.is_first    = is_first
                w.is_last     = is_last
                # Worker 1 receives *tokens* on the result port;
                # workers 2 & 3 receive *hidden states* on the in-port.
                w.zmq_in_port = ZMQ_RESULT_PORT if is_first else ZMQ_IN_PORT

                return {
                    "layer_start": start,
                    "layer_end":   end,
                    "is_first":    is_first,
                    "is_last":     is_last,
                    "zmq_in_port": w.zmq_in_port,
                }

        raise ValueError("All layer shards are already assigned.")