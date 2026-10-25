import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class WorkerStatus(str, Enum):
    REGISTERING = "registering"
    LOADING     = "loading"
    LOADED      = "loaded"   # model ready; ZMQ not yet wired
    ACTIVE      = "active"   # fully connected; serving inference
    DEAD        = "dead"


@dataclass
class WorkerInfo:
    worker_id:  str
    hostname:   str
    ip:         str
    http_port:  int
    gpu_name:   str
    vram_gb:    float
    zmq_in_port: int          = 0
    layer_start: Optional[int] = None
    layer_end:   Optional[int] = None
    is_first:    bool          = False
    is_last:     bool          = False
    status:      WorkerStatus  = WorkerStatus.REGISTERING
    last_heartbeat: float      = field(default_factory=time.time)


class WorkerRegistry:
    def __init__(self):
        self._workers: Dict[str, WorkerInfo] = {}

    def add(self, **kwargs) -> WorkerInfo:
        wid = str(uuid.uuid4())
        w   = WorkerInfo(worker_id=wid, **kwargs)
        self._workers[wid] = w
        return w

    def get(self, wid: str) -> Optional[WorkerInfo]:
        return self._workers.get(wid)

    def all(self) -> List[WorkerInfo]:
        return list(self._workers.values())

    def heartbeat(self, wid: str, status: WorkerStatus):
        if wid in self._workers:
            self._workers[wid].last_heartbeat = time.time()
            self._workers[wid].status = status

    def mark_dead(self, wid: str):
        if wid in self._workers:
            self._workers[wid].status = WorkerStatus.DEAD

    def remove(self, wid: str):
        self._workers.pop(wid, None)

    # ── Pipeline helpers ──────────────────────────────────────────────────────

    def pipeline(self) -> List[WorkerInfo]:
        """
        Loaded OR active workers ordered by layer_start.
        Used by /topology so workers can wire ZMQ before going active.
        """
        eligible = [
            w for w in self._workers.values()
            if w.status in (WorkerStatus.LOADED, WorkerStatus.ACTIVE)
            and w.layer_start is not None
        ]
        return sorted(eligible, key=lambda w: w.layer_start)

    def pipeline_ready(self) -> bool:
        """True when every layer 0-15 is covered by a loaded/active worker."""
        from shared.config import TOTAL_LAYERS
        covered: set[int] = set()
        for w in self.pipeline():
            for i in range(w.layer_start, w.layer_end + 1):
                covered.add(i)
        return len(covered) == TOTAL_LAYERS