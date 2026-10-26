import logging

from coordinator.registry import WorkerRegistry
from shared.config import (
    SINGLE_NODE_VRAM_GB,
    TOTAL_LAYERS,
    ZMQ_IN_PORT,
    ZMQ_RESULT_PORT,
)

log = logging.getLogger(__name__)

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
        Single-node-first: if this is the ONLY worker registered so far and
        no shard has been claimed by anyone yet, try giving it the WHOLE
        model (zero inter-node hops) when its reported VRAM clears a rough
        feasibility threshold. Only when that doesn't apply — not enough
        VRAM, or another worker already registered — fall back to the fixed
        3-stage LAYER_RANGES split below.

        NOTE: this intentionally does not handle a second worker joining
        AFTER a single-node pipeline is already active/loading — collapsing
        or re-sharding a live pipeline is drain-then-form territory
        (Phase 6), out of scope here. A late-arriving worker in that
        situation just proceeds to the sharded fallback and waits.
        """
        w = self.registry.get(worker_id)
        occupied = {
            (ww.layer_start, ww.layer_end)
            for ww in self.registry.all()
            if ww.layer_start is not None and ww.worker_id != worker_id
        }
        others = [ww for ww in self.registry.all() if ww.worker_id != worker_id]

        if not others and not occupied:
            if w.vram_gb >= SINGLE_NODE_VRAM_GB:
                w.layer_start = 0
                w.layer_end   = TOTAL_LAYERS - 1
                w.is_first    = True
                w.is_last     = True
                w.zmq_in_port = ZMQ_RESULT_PORT  # unused in local mode; kept for schema consistency
                log.info(
                    f"Placement: LOCAL (single-node, zero hops). worker={worker_id} "
                    f"vram_gb={w.vram_gb:.1f} >= threshold {SINGLE_NODE_VRAM_GB:.1f} GB "
                    f"layers=0-{TOTAL_LAYERS - 1}"
                )
                return {
                    "layer_start": 0,
                    "layer_end":   TOTAL_LAYERS - 1,
                    "is_first":    True,
                    "is_last":     True,
                    "zmq_in_port": w.zmq_in_port,
                }
            else:
                log.info(
                    f"Placement: SHARDED fallback. worker={worker_id} "
                    f"vram_gb={w.vram_gb:.1f} < threshold {SINGLE_NODE_VRAM_GB:.1f} GB "
                    f"— this node alone can't hold the model; waiting for "
                    f"additional node(s), minimum pipeline depth={len(LAYER_RANGES)}"
                )

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